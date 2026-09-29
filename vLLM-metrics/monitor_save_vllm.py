#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean


DEFAULT_URL = "http://127.0.0.1:8000/metrics"
DEFAULT_DURATION = 600
DEFAULT_INTERVAL = 2

LABEL_RE = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"'
)


# ============================================================
# Prometheus metrics parser
# ============================================================

def fetch_metrics(url: str) -> dict:
    """Fetch and parse Prometheus metrics."""

    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode("utf-8")

    metrics: dict[str, list[tuple[dict[str, str], float]]] = {}

    for line in text.splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        try:
            left, value_text = line.rsplit(None, 1)
            value = float(value_text)
        except (ValueError, IndexError):
            continue

        if "{" in left:
            metric_name, labels_part = left.split("{", 1)
            labels_part = labels_part.rstrip("}")

            labels = {
                match.group(1): match.group(2)
                for match in LABEL_RE.finditer(labels_part)
            }

        else:
            metric_name = left
            labels = {}

        metrics.setdefault(metric_name, []).append(
            (labels, value)
        )

    return metrics


def sum_metric(
    metrics: dict,
    name: str,
    filters: dict[str, str] | None = None,
) -> float:

    total = 0.0
    filters = filters or {}

    for labels, value in metrics.get(name, []):

        if all(
            labels.get(k) == v
            for k, v in filters.items()
        ):
            total += value

    return total


def metric_by_engine(
    metrics: dict,
    name: str,
) -> dict[str, float]:

    result = {}

    for labels, value in metrics.get(name, []):

        engine = labels.get("engine")

        if engine is not None:
            result[engine] = value

    return result


def counter_delta(
    start: dict,
    end: dict,
    name: str,
    filters: dict[str, str] | None = None,
) -> float:

    before = sum_metric(start, name, filters)
    after = sum_metric(end, name, filters)

    # Counter 正常情况下只增不减。
    # 如果 server 中途重启，避免返回负数。
    return max(0.0, after - before)


# ============================================================
# Histogram utilities
# ============================================================

def histogram_buckets(
    metrics: dict,
    base_name: str,
) -> dict[float, float]:

    result = {}

    metric_name = base_name + "_bucket"

    for labels, value in metrics.get(metric_name, []):

        le = labels.get("le")

        if le is None:
            continue

        try:
            boundary = float(le)
        except ValueError:
            continue

        result[boundary] = (
            result.get(boundary, 0.0)
            + value
        )

    return result


def histogram_delta_buckets(
    start: dict,
    end: dict,
    base_name: str,
) -> dict[float, float]:

    start_buckets = histogram_buckets(
        start,
        base_name,
    )

    end_buckets = histogram_buckets(
        end,
        base_name,
    )

    result = {}

    for boundary, end_value in end_buckets.items():

        start_value = start_buckets.get(
            boundary,
            0.0,
        )

        result[boundary] = max(
            0.0,
            end_value - start_value,
        )

    return result


def histogram_quantile(
    buckets: dict[float, float],
    q: float,
) -> float | None:
    """
    根据 Prometheus cumulative histogram buckets
    近似计算 quantile。

    使用 bucket 内线性插值。
    """

    if not buckets:
        return None

    ordered = sorted(
        buckets.items(),
        key=lambda x: x[0],
    )

    # +Inf bucket 一般就是总样本数量
    total = ordered[-1][1]

    if total <= 0:
        return None

    target = total * q

    prev_boundary = 0.0
    prev_count = 0.0

    for boundary, cumulative_count in ordered:

        if cumulative_count >= target:

            # 如果落进 +Inf bucket，
            # 只能返回前一个有限边界。
            if boundary == float("inf"):
                return prev_boundary

            bucket_count = (
                cumulative_count
                - prev_count
            )

            if bucket_count <= 0:
                return boundary

            fraction = (
                (target - prev_count)
                / bucket_count
            )

            return (
                prev_boundary
                + fraction
                * (boundary - prev_boundary)
            )

        prev_boundary = boundary
        prev_count = cumulative_count

    return None


def histogram_stats(
    start: dict,
    end: dict,
    base_name: str,
) -> dict:

    count = counter_delta(
        start,
        end,
        base_name + "_count",
    )

    total_sum = counter_delta(
        start,
        end,
        base_name + "_sum",
    )

    buckets = histogram_delta_buckets(
        start,
        end,
        base_name,
    )

    avg = (
        total_sum / count
        if count > 0
        else None
    )

    return {
        "samples": int(count),
        "mean": avg,
        "p50": histogram_quantile(
            buckets,
            0.50,
        ),
        "p95": histogram_quantile(
            buckets,
            0.95,
        ),
        "p99": histogram_quantile(
            buckets,
            0.99,
        ),
    }


# ============================================================
# Gauge sampling
# ============================================================

GAUGE_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv": "vllm:kv_cache_usage_perc",
}


def extract_gauge_sample(metrics: dict) -> dict:

    return {
        name: metric_by_engine(
            metrics,
            metric_name,
        )
        for name, metric_name
        in GAUGE_METRICS.items()
    }


def summarize_gauges(samples: list[dict]) -> dict:

    engines = set()

    for sample in samples:
        for metric_data in sample.values():
            engines.update(metric_data.keys())

    result = {
        "engines": {},
    }

    # ------------------------
    # Per-engine
    # ------------------------

    for engine in sorted(engines):

        engine_result = {}

        for metric_name in GAUGE_METRICS:

            values = [
                sample[metric_name][engine]
                for sample in samples
                if engine
                in sample[metric_name]
            ]

            if not values:
                continue

            if metric_name == "kv":

                engine_result[
                    "kv_avg_pct"
                ] = mean(values) * 100

                engine_result[
                    "kv_max_pct"
                ] = max(values) * 100

            else:

                engine_result[
                    f"{metric_name}_avg"
                ] = mean(values)

                engine_result[
                    f"{metric_name}_max"
                ] = max(values)

        result["engines"][engine] = (
            engine_result
        )

    # ------------------------
    # Global Running / Waiting
    # ------------------------

    running_totals = []
    waiting_totals = []
    all_kv_values = []

    for sample in samples:

        running_totals.append(
            sum(
                sample["running"].values()
            )
        )

        waiting_totals.append(
            sum(
                sample["waiting"].values()
            )
        )

        all_kv_values.extend(
            sample["kv"].values()
        )

    result["running_avg_total"] = (
        mean(running_totals)
        if running_totals
        else 0
    )

    result["running_peak_total"] = (
        max(running_totals)
        if running_totals
        else 0
    )

    result["waiting_avg_total"] = (
        mean(waiting_totals)
        if waiting_totals
        else 0
    )

    result["waiting_peak_total"] = (
        max(waiting_totals)
        if waiting_totals
        else 0
    )

    result["kv_avg_pct"] = (
        mean(all_kv_values) * 100
        if all_kv_values
        else 0
    )

    result["kv_peak_pct"] = (
        max(all_kv_values) * 100
        if all_kv_values
        else 0
    )

    return result


# ============================================================
# Helpers
# ============================================================

def safe_round(value, digits=3):

    if value is None:
        return None

    return round(value, digits)


def sanitize_label(label: str) -> str:

    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        label,
    )


# ============================================================
# Main benchmark
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect one-minute vLLM serving metrics."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="Measurement duration in seconds.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=(
            "Internal gauge sampling interval. "
            "Samples are NOT saved individually."
        ),
    )

    parser.add_argument(
        "--label",
        default="run",
        help=(
            "Experiment label, e.g. "
            "workers64_api2"
        ),
    )

    parser.add_argument(
        "--wait-for-load",
        action="store_true",
        help=(
            "Wait until at least one request is "
            "running before starting the timer."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="metrics_results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Wait for workload
    # ========================================================

    print()
    print("=" * 70)
    print("vLLM 1-Minute Metrics Collector")
    print("=" * 70)

    print(f"Metrics URL : {args.url}")
    print(f"Duration    : {args.duration}s")
    print(f"Interval    : {args.interval}s")
    print(f"Label       : {args.label}")

    print()

    if args.wait_for_load:

        print(
            "Waiting for vLLM workload..."
        )

        while True:

            current = fetch_metrics(
                args.url
            )

            running = sum_metric(
                current,
                "vllm:num_requests_running",
            )

            if running > 0:
                print(
                    f"Workload detected "
                    f"(running={running:.0f})."
                )
                break

            time.sleep(0.5)

    else:

        current = fetch_metrics(
            args.url
        )

    # ========================================================
    # Start snapshot
    # ========================================================

    start_metrics = current

    start_wall_time = datetime.now()
    start_monotonic = time.monotonic()

    gauge_samples = [
        extract_gauge_sample(
            start_metrics
        )
    ]

    print()
    print(
        f"Measurement started at "
        f"{start_wall_time.isoformat(timespec='seconds')}"
    )

    print(
        f"Collecting for "
        f"{args.duration} seconds..."
    )

    # ========================================================
    # Sample Gauges
    # ========================================================

    while True:

        elapsed = (
            time.monotonic()
            - start_monotonic
        )

        if elapsed >= args.duration:
            break

        remaining = (
            args.duration - elapsed
        )

        time.sleep(
            min(
                args.interval,
                remaining,
            )
        )

        if (
            time.monotonic()
            - start_monotonic
            >= args.duration
        ):
            break

        metrics = fetch_metrics(
            args.url
        )

        gauge_samples.append(
            extract_gauge_sample(
                metrics
            )
        )

    # ========================================================
    # End snapshot
    # ========================================================

    end_metrics = fetch_metrics(
        args.url
    )

    gauge_samples.append(
        extract_gauge_sample(
            end_metrics
        )
    )

    actual_duration = (
        time.monotonic()
        - start_monotonic
    )

    # ========================================================
    # Token counters
    # ========================================================

    prompt_tokens = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:prompt_tokens_total",
    )

    prefill_compute_tokens = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:prompt_tokens_by_source_total",
        {
            "source": "local_compute"
        },
    )

    cached_prompt_tokens = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:prompt_tokens_by_source_total",
        {
            "source": "local_cache_hit"
        },
    )

    generation_tokens = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:generation_tokens_total",
    )

    prefix_queries = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:prefix_cache_queries_total",
    )

    prefix_hits = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:prefix_cache_hits_total",
    )

    preemptions = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:num_preemptions_total",
    )

    requests_completed = counter_delta(
        start_metrics,
        end_metrics,
        "vllm:request_success_total",
    )

    # ========================================================
    # Throughput
    # ========================================================

    prompt_tok_s = (
        prompt_tokens
        / actual_duration
    )

    prefill_compute_tok_s = (
        prefill_compute_tokens
        / actual_duration
    )

    cached_prompt_tok_s = (
        cached_prompt_tokens
        / actual_duration
    )

    decode_tok_s = (
        generation_tokens
        / actual_duration
    )

    request_per_s = (
        requests_completed
        / actual_duration
    )

    prefix_hit_rate = (
        prefix_hits
        / prefix_queries
        * 100
        if prefix_queries > 0
        else 0
    )

    # ========================================================
    # Latency Histograms
    # ========================================================

    ttft = histogram_stats(
        start_metrics,
        end_metrics,
        "vllm:time_to_first_token_seconds",
    )

    tpot = histogram_stats(
        start_metrics,
        end_metrics,
        "vllm:request_time_per_output_token_seconds",
    )

    queue = histogram_stats(
        start_metrics,
        end_metrics,
        "vllm:request_queue_time_seconds",
    )

    e2e = histogram_stats(
        start_metrics,
        end_metrics,
        "vllm:e2e_request_latency_seconds",
    )

    prefill_time = histogram_stats(
        start_metrics,
        end_metrics,
        "vllm:request_prefill_time_seconds",
    )

    decode_time = histogram_stats(
        start_metrics,
        end_metrics,
        "vllm:request_decode_time_seconds",
    )

    # ========================================================
    # Gauge summary
    # ========================================================

    gauges = summarize_gauges(
        gauge_samples
    )

    # ========================================================
    # Final result
    # ========================================================

    result = {

        "experiment": {
            "label": args.label,
            "start_time": (
                start_wall_time.isoformat()
            ),
            "duration_seconds": (
                safe_round(
                    actual_duration,
                    2,
                )
            ),
            "gauge_samples": len(
                gauge_samples
            ),
        },

        "tokens": {
            "prompt_tokens": int(
                prompt_tokens
            ),
            "prefill_compute_tokens": int(
                prefill_compute_tokens
            ),
            "cached_prompt_tokens": int(
                cached_prompt_tokens
            ),
            "generation_tokens": int(
                generation_tokens
            ),
        },

        "throughput": {
            "prompt_tok_s": safe_round(
                prompt_tok_s,
                2,
            ),
            "prefill_compute_tok_s": (
                safe_round(
                    prefill_compute_tok_s,
                    2,
                )
            ),
            "cached_prompt_tok_s": (
                safe_round(
                    cached_prompt_tok_s,
                    2,
                )
            ),
            "decode_tok_s": safe_round(
                decode_tok_s,
                2,
            ),
            "requests_per_s": safe_round(
                request_per_s,
                4,
            ),
        },

        "prefix_cache": {
            "query_tokens": int(
                prefix_queries
            ),
            "hit_tokens": int(
                prefix_hits
            ),
            "hit_rate_pct": safe_round(
                prefix_hit_rate,
                2,
            ),
        },

        "scheduler": {
            "preemptions": int(
                preemptions
            ),
            "requests_completed": int(
                requests_completed
            ),
            **gauges,
        },

        "latency": {
            "ttft_seconds": {
                k: safe_round(v, 4)
                if isinstance(v, float)
                else v
                for k, v in ttft.items()
            },
            "tpot_seconds_per_token": {
                k: safe_round(v, 5)
                if isinstance(v, float)
                else v
                for k, v in tpot.items()
            },
            "queue_seconds": {
                k: safe_round(v, 4)
                if isinstance(v, float)
                else v
                for k, v in queue.items()
            },
            "e2e_seconds": {
                k: safe_round(v, 3)
                if isinstance(v, float)
                else v
                for k, v in e2e.items()
            },
            "prefill_seconds": {
                k: safe_round(v, 3)
                if isinstance(v, float)
                else v
                for k, v in prefill_time.items()
            },
            "decode_seconds": {
                k: safe_round(v, 3)
                if isinstance(v, float)
                else v
                for k, v in decode_time.items()
            },
        },
    }

    # ========================================================
    # Save JSON
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_label = sanitize_label(
        args.label
    )

    json_path = (
        output_dir
        / f"{safe_label}_{timestamp}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Append one row to CSV
    # ========================================================

    csv_path = (
        output_dir
        / "vllm_benchmark_summary.csv"
    )

    csv_row = {
        "timestamp": timestamp,
        "label": args.label,
        "duration_s": safe_round(
            actual_duration,
            2,
        ),

        "prompt_tokens": int(
            prompt_tokens
        ),

        "prompt_tok_s": safe_round(
            prompt_tok_s,
            2,
        ),

        "prefill_compute_tokens": int(
            prefill_compute_tokens
        ),

        "prefill_compute_tok_s": safe_round(
            prefill_compute_tok_s,
            2,
        ),

        "cached_prompt_tokens": int(
            cached_prompt_tokens
        ),

        "cached_prompt_tok_s": safe_round(
            cached_prompt_tok_s,
            2,
        ),

        "generation_tokens": int(
            generation_tokens
        ),

        "decode_tok_s": safe_round(
            decode_tok_s,
            2,
        ),

        "requests_completed": int(
            requests_completed
        ),

        "requests_per_s": safe_round(
            request_per_s,
            4,
        ),

        "prefix_hit_rate_pct": safe_round(
            prefix_hit_rate,
            2,
        ),

        "preemptions": int(
            preemptions
        ),

        "running_avg": safe_round(
            gauges[
                "running_avg_total"
            ],
            2,
        ),

        "running_peak": safe_round(
            gauges[
                "running_peak_total"
            ],
            2,
        ),

        "waiting_avg": safe_round(
            gauges[
                "waiting_avg_total"
            ],
            2,
        ),

        "waiting_peak": safe_round(
            gauges[
                "waiting_peak_total"
            ],
            2,
        ),

        "kv_avg_pct": safe_round(
            gauges[
                "kv_avg_pct"
            ],
            2,
        ),

        "kv_peak_pct": safe_round(
            gauges[
                "kv_peak_pct"
            ],
            2,
        ),

        "ttft_samples": ttft[
            "samples"
        ],

        "ttft_mean_s": safe_round(
            ttft["mean"],
            3,
        ),

        "ttft_p50_s": safe_round(
            ttft["p50"],
            3,
        ),

        "ttft_p95_s": safe_round(
            ttft["p95"],
            3,
        ),

        "tpot_samples": tpot[
            "samples"
        ],

        "tpot_mean_ms": (
            safe_round(
                tpot["mean"] * 1000,
                2,
            )
            if tpot["mean"]
            is not None
            else None
        ),

        "tpot_p50_ms": (
            safe_round(
                tpot["p50"] * 1000,
                2,
            )
            if tpot["p50"]
            is not None
            else None
        ),

        "tpot_p95_ms": (
            safe_round(
                tpot["p95"] * 1000,
                2,
            )
            if tpot["p95"]
            is not None
            else None
        ),

        "queue_mean_s": safe_round(
            queue["mean"],
            3,
        ),

        "queue_p95_s": safe_round(
            queue["p95"],
            3,
        ),

        "e2e_mean_s": safe_round(
            e2e["mean"],
            3,
        ),

        "e2e_p95_s": safe_round(
            e2e["p95"],
            3,
        ),

        "prefill_mean_s": safe_round(
            prefill_time["mean"],
            3,
        ),

        "decode_mean_s": safe_round(
            decode_time["mean"],
            3,
        ),
    }

    file_exists = csv_path.exists()

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                csv_row.keys()
            ),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            csv_row
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 70)
    print("60-SECOND RESULT")
    print("=" * 70)

    print()
    print("TOKEN THROUGHPUT")
    print("-" * 70)

    print(
        f"Prompt              : "
        f"{prompt_tok_s:10.1f} tok/s "
        f"({prompt_tokens:,.0f} tokens)"
    )

    print(
        f"Prefill compute     : "
        f"{prefill_compute_tok_s:10.1f} tok/s "
        f"({prefill_compute_tokens:,.0f} tokens)"
    )

    print(
        f"Prefix cached       : "
        f"{cached_prompt_tok_s:10.1f} tok/s "
        f"({cached_prompt_tokens:,.0f} tokens)"
    )

    print(
        f"Decode              : "
        f"{decode_tok_s:10.1f} tok/s "
        f"({generation_tokens:,.0f} tokens)"
    )

    print()
    print("CACHE / SCHEDULER")
    print("-" * 70)

    print(
        f"Prefix hit rate     : "
        f"{prefix_hit_rate:10.2f}%"
    )

    print(
        f"Running avg / peak  : "
        f"{gauges['running_avg_total']:.1f}"
        f" / "
        f"{gauges['running_peak_total']:.0f}"
    )

    print(
        f"Waiting avg / peak  : "
        f"{gauges['waiting_avg_total']:.1f}"
        f" / "
        f"{gauges['waiting_peak_total']:.0f}"
    )

    print(
        f"KV avg / peak       : "
        f"{gauges['kv_avg_pct']:.1f}%"
        f" / "
        f"{gauges['kv_peak_pct']:.1f}%"
    )

    print(
        f"Preemptions         : "
        f"{preemptions:.0f}"
    )

    print()
    print("LATENCY")
    print("-" * 70)

    if ttft["samples"] > 0:
        print(
            f"TTFT mean / P95     : "
            f"{ttft['mean']:.3f}s"
            f" / "
            f"{ttft['p95']:.3f}s"
        )

    if tpot["samples"] > 0:
        print(
            f"TPOT mean / P95     : "
            f"{tpot['mean'] * 1000:.1f}ms"
            f" / "
            f"{tpot['p95'] * 1000:.1f}ms"
        )

    if queue["samples"] > 0:
        print(
            f"Queue mean / P95    : "
            f"{queue['mean']:.3f}s"
            f" / "
            f"{queue['p95']:.3f}s"
        )

    print()
    print("PER ENGINE")
    print("-" * 70)

    for (
        engine,
        data,
    ) in gauges[
        "engines"
    ].items():

        print(
            f"Engine {engine}: "
            f"running avg={data.get('running_avg', 0):.1f}, "
            f"max={data.get('running_max', 0):.0f}; "
            f"waiting avg={data.get('waiting_avg', 0):.1f}, "
            f"max={data.get('waiting_max', 0):.0f}; "
            f"KV avg={data.get('kv_avg_pct', 0):.1f}%, "
            f"max={data.get('kv_max_pct', 0):.1f}%"
        )

    print()
    print(
        f"JSON saved : {json_path}"
    )

    print(
        f"CSV updated: {csv_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()