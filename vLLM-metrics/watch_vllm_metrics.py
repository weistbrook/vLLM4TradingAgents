import re
import time
import urllib.request

URL = "http://127.0.0.1:8000/metrics"
INTERVAL = 1.0


def fetch_metrics():
    with urllib.request.urlopen(URL, timeout=5) as response:
        return response.read().decode("utf-8")


def get_metric(text, metric_name, extra_label=None):
    """
    返回:
    {
        "0": value,
        "1": value,
    }
    """
    values = {}

    for line in text.splitlines():
        if not line.startswith(metric_name + "{"):
            continue

        if extra_label is not None and extra_label not in line:
            continue

        engine_match = re.search(r'engine="([^"]+)"', line)
        if not engine_match:
            continue

        engine = engine_match.group(1)

        try:
            value = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue

        values[engine] = value

    return values


def rate(current, previous, dt):
    result = {}

    for engine, value in current.items():
        old = previous.get(engine)

        if old is None:
            result[engine] = 0.0
        else:
            result[engine] = max(0.0, (value - old) / dt)

    return result


def fmt(values):
    return " | ".join(
        f"E{engine}: {value:8.1f}"
        for engine, value in sorted(values.items())
    )


print("Watching vLLM metrics...")
print("Ctrl+C to stop\n")

previous = None
previous_time = None

try:
    while True:
        text = fetch_metrics()
        now = time.time()

        prompt = get_metric(
            text,
            "vllm:prompt_tokens_total"
        )

        prefill_compute = get_metric(
            text,
            "vllm:prompt_tokens_by_source_total",
            'source="local_compute"',
        )

        cache_hit = get_metric(
            text,
            "vllm:prompt_tokens_by_source_total",
            'source="local_cache_hit"',
        )

        generation = get_metric(
            text,
            "vllm:generation_tokens_total"
        )

        running = get_metric(
            text,
            "vllm:num_requests_running"
        )

        waiting = get_metric(
            text,
            "vllm:num_requests_waiting"
        )

        kv = get_metric(
            text,
            "vllm:kv_cache_usage_perc"
        )

        if previous is not None:
            dt = now - previous_time

            prompt_rate = rate(
                prompt,
                previous["prompt"],
                dt,
            )

            prefill_rate = rate(
                prefill_compute,
                previous["prefill_compute"],
                dt,
            )

            cache_rate = rate(
                cache_hit,
                previous["cache_hit"],
                dt,
            )

            decode_rate = rate(
                generation,
                previous["generation"],
                dt,
            )

            total_prompt = sum(prompt_rate.values())
            total_prefill = sum(prefill_rate.values())
            total_cache = sum(cache_rate.values())
            total_decode = sum(decode_rate.values())

            print("\033[2J\033[H", end="")

            print(
                f"vLLM realtime metrics "
                f"(interval={dt:.2f}s)"
            )
            print("=" * 72)

            print("\nTOKEN THROUGHPUT")
            print("-" * 72)

            print(
                f"Prompt throughput : "
                f"{total_prompt:9.1f} tok/s"
            )

            print(
                f"Prefill compute   : "
                f"{total_prefill:9.1f} tok/s"
            )

            print(
                f"Prefix cache hit  : "
                f"{total_cache:9.1f} tok/s"
            )

            print(
                f"Decode throughput : "
                f"{total_decode:9.1f} tok/s"
            )

            print(
                f"Total compute tok : "
                f"{total_prefill + total_decode:9.1f} tok/s"
            )

            print("\nPER ENGINE")
            print("-" * 72)

            print(
                "Prompt :",
                fmt(prompt_rate)
            )

            print(
                "Prefill:",
                fmt(prefill_rate)
            )

            print(
                "Decode :",
                fmt(decode_rate)
            )

            print("\nSCHEDULER")
            print("-" * 72)

            print(
                "Running:",
                fmt(running)
            )

            print(
                "Waiting:",
                fmt(waiting)
            )

            kv_percent = {
                engine: value * 100
                for engine, value in kv.items()
            }

            print(
                "KV %   :",
                fmt(kv_percent)
            )

        previous = {
            "prompt": prompt,
            "prefill_compute": prefill_compute,
            "cache_hit": cache_hit,
            "generation": generation,
        }

        previous_time = now

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nStopped.")