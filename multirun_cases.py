from __future__ import annotations

import copy
import json
import os
import signal
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.main import save_report_to_disk


# ── Signal handling ──────────────────────────────────────────────────────────

def _force_exit(signum, frame):
    """
    Ctrl+C / kill 时立即结束整个 batch runner。

    ThreadPoolExecutor 中正在运行的线程无法被 Python 安全地强制 kill，
    因此这里使用 os._exit()，避免等待网络请求 / LLM 调用结束。
    """
    signal_name = signal.Signals(signum).name

    print(
        f"\n\n⚠️  Received {signal_name}. "
        "Force stopping all case runners...",
        flush=True,
    )

    # 130 = 128 + SIGINT(2)
    # SIGTERM 对应 143 = 128 + 15
    os._exit(128 + signum)


signal.signal(signal.SIGINT, _force_exit)
signal.signal(signal.SIGTERM, _force_exit)

# Linux / macOS 下让阻塞系统调用也尽量被 SIGINT 中断
if hasattr(signal, "siginterrupt"):
    signal.siginterrupt(signal.SIGINT, True)
    signal.siginterrupt(signal.SIGTERM, True)


# ── Config ───────────────────────────────────────────────────────────────────

CASES_DIR = Path(__file__).parent / "cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)

TRADE_DATE = "2026-09-28"


# 64 A-share tickers across different sectors
# fmt: off
TICKERS = {

    # ── 白酒 / 消费 / 家电 / 食品 ──────────────────────────────────────────
    "600519": "贵州茅台 (主板·白酒)",
    "000858": "五粮液 (主板·白酒)",
    "000568": "泸州老窖 (主板·白酒)",
    "600809": "山西汾酒 (主板·白酒)",
    "600887": "伊利股份 (主板·乳制品)",
    "000651": "格力电器 (主板·家电)",
    "000333": "美的集团 (主板·家电)",
    "603288": "海天味业 (主板·调味品)",

    # ── 新能源 / 电池 / 光伏 ──────────────────────────────────────────────
    "300750": "宁德时代 (创业板·动力电池)",
    "002594": "比亚迪 (主板·新能源汽车)",
    "601012": "隆基绿能 (主板·光伏)",
    "600732": "爱旭股份 (主板·光伏)",
    "002460": "赣锋锂业 (主板·锂矿)",
    "300274": "阳光电源 (创业板·光伏逆变器)",
    "300014": "亿纬锂能 (创业板·锂电池)",
    "600438": "通威股份 (主板·光伏)",
    "601865": "福莱特 (主板·光伏玻璃)",
    "002812": "恩捷股份 (主板·锂电隔膜)",

    # ── 半导体 / 芯片 / 消费电子 ──────────────────────────────────────────
    "688981": "中芯国际 (科创板·晶圆代工)",
    "002475": "立讯精密 (主板·消费电子)",
    "002600": "领益智造 (主板·消费电子)",
    "300782": "卓胜微 (创业板·射频芯片)",
    "603501": "豪威集团 (主板·图像传感器)",
    "688012": "中微公司 (科创板·半导体设备)",
    "688008": "澜起科技 (科创板·芯片设计)",
    "688256": "寒武纪 (科创板·AI芯片)",
    "600584": "长电科技 (主板·半导体封测)",
    "603986": "兆易创新 (主板·存储芯片)",
    "002371": "北方华创 (主板·半导体设备)",

    # ── 医药 / 医疗器械 ──────────────────────────────────────────────────
    "300760": "迈瑞医疗 (创业板·医疗器械)",
    "600276": "恒瑞医药 (主板·创新药)",
    "300015": "爱尔眼科 (创业板·医疗服务)",
    "603259": "药明康德 (主板·医药研发服务)",
    "300122": "智飞生物 (创业板·疫苗)",
    "000661": "长春高新 (主板·生物医药)",

    # ── 券商 / 银行 / 保险 ────────────────────────────────────────────────
    "300059": "东方财富 (创业板·互联网券商)",
    "600030": "中信证券 (主板·券商)",
    "601318": "中国平安 (主板·保险)",
    "600036": "招商银行 (主板·银行)",
    "601166": "兴业银行 (主板·银行)",
    "601398": "工商银行 (主板·银行)",
    "601288": "农业银行 (主板·银行)",

    # ── 电力 / 石油 / 煤炭 / 能源 ────────────────────────────────────────
    "600578": "京能电力 (主板·火电)",
    "601991": "大唐发电 (主板·电力)",
    "600900": "长江电力 (主板·水电)",
    "601088": "中国神华 (主板·煤炭)",
    "601857": "中国石油 (主板·石油天然气)",
    "600028": "中国石化 (主板·石油化工)",
    "600886": "国投电力 (主板·电力)",

    # ── 工业自动化 / 机器人 / 高端制造 ──────────────────────────────────
    "300124": "汇川技术 (创业板·工业自动化)",
    "688017": "绿的谐波 (科创板·谐波减速器)",
    "002654": "万润科技 (主板·LED/科技产业)",
    "002050": "三花智控 (主板·热管理/机器人)",
    "601100": "恒立液压 (主板·液压设备)",
    "600031": "三一重工 (主板·工程机械)",
    "000157": "中联重科 (主板·工程机械)",

    # ── 面板 / 显示 / 消费电子 ───────────────────────────────────────────
    "000725": "京东方A (主板·显示面板)",
    "002241": "歌尔股份 (主板·消费电子)",

    # ── 化工 / 有色 / 材料 ───────────────────────────────────────────────
    "600691": "潞化科技 (主板·化工)",
    "600309": "万华化学 (主板·化工)",
    "002466": "天齐锂业 (主板·锂矿)",
    "601899": "紫金矿业 (主板·有色金属)",
    "600547": "山东黄金 (主板·黄金)",
    "600585": "海螺水泥 (主板·建材)",
}
# fmt: on

assert len(TICKERS) == 64, f"Expected 64 tickers, got {len(TICKERS)}"


def build_config() -> dict:
    """Build the TradingAgents config for case runs."""
    config = DEFAULT_CONFIG.copy()

    config["llm_provider"] = "openai_compatible"
    config["backend_url"] = "http://127.0.0.1:8000/v1"

    config["deep_think_llm"] = "Qwen3.8-27B"
    config["quick_think_llm"] = "Qwen3.8-27B"

    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }

    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = "Chinese"

    return config


def run_single(ticker: str, label: str, config: dict) -> None:
    """Run one ticker and save the complete analysis report."""

    print(f"\n{'=' * 60}")
    print(f"Analysing {ticker} — {label}")
    print(f"Trade date: {TRADE_DATE}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # 每个线程使用独立 config，避免 TradingAgentsGraph 内部修改 config
    # 时多个 worker 相互影响。
    ta = TradingAgentsGraph(
        debug=True,
        config=copy.deepcopy(config),
    )

    final_state = None
    decision = ""

    try:
        final_state, decision = ta.propagate(ticker, TRADE_DATE)
    except Exception as e:
        decision = f"ERROR: {type(e).__name__}: {e}"

    elapsed = time.time() - start_time

    if final_state is None:
        print(f"\n❌ Failed {ticker}: {decision}")
        return

    stock_name = label.split("(")[0].strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    dir_name = f"{ticker}_{stock_name}_{ts}"
    ticker_dir = CASES_DIR / dir_name

    report_path = save_report_to_disk(
        final_state,
        ticker,
        ticker_dir,
    )

    renamed_report = ticker_dir / f"{dir_name}.md"
    report_path.rename(renamed_report)

    print(
        f"\n✅ {ticker} Report saved to "
        f"{renamed_report} ({elapsed:.0f}s)"
    )

    summary_path = ticker_dir / "summary.json"

    _save_json_summary(
        summary_path,
        ticker,
        label,
        elapsed,
        final_state,
        decision,
    )


def _save_json_summary(
    summary_path: Path,
    ticker: str,
    label: str,
    elapsed: float,
    final_state: dict,
    decision: str,
) -> None:

    summary = {
        "ticker": ticker,
        "label": label,
        "trade_date": TRADE_DATE,
        "run_time": datetime.now().isoformat(),
        "duration_seconds": round(elapsed),
        "signal": decision,
        "reports": {},
    }

    report_keys = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "policy_report",
        "hot_money_report",
        "lockup_report",
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    ]

    for key in report_keys:
        val = final_state.get(key, "")
        if val:
            summary["reports"][key] = val[:3000]

    debate = final_state.get("investment_debate_state", {})

    if debate:
        summary["reports"]["bull_history"] = (
            debate.get("bull_history", "")[:2000]
        )
        summary["reports"]["bear_history"] = (
            debate.get("bear_history", "")[:2000]
        )
        summary["reports"]["research_manager"] = (
            debate.get("judge_decision", "")[:2000]
        )

    risk = final_state.get("risk_debate_state", {})

    if risk:
        summary["reports"]["aggressive_analyst"] = (
            risk.get("aggressive_history", "")[:2000]
        )
        summary["reports"]["conservative_analyst"] = (
            risk.get("conservative_history", "")[:2000]
        )
        summary["reports"]["neutral_analyst"] = (
            risk.get("neutral_history", "")[:2000]
        )
        summary["reports"]["portfolio_manager"] = (
            risk.get("judge_decision", "")[:2000]
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:

    config = build_config()

    parser = ArgumentParser(
        description="Run A-stock analysis cases concurrently."
    )

    parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker codes to run; omit to run all entries in TICKERS.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of concurrent stock analyses (default: 1).",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    # 检查传入 ticker
    unknown_tickers = [
        ticker
        for ticker in args.tickers
        if ticker not in TICKERS
    ]

    if unknown_tickers:
        print(
            "⚠️ Unknown ticker(s), using ticker code as label: "
            + ", ".join(unknown_tickers)
        )

    selected_tickers = args.tickers or list(TICKERS)

    cases = [
        (ticker, TICKERS.get(ticker, ticker))
        for ticker in selected_tickers
    ]

    print(
        f"\n🚀 Running {len(cases)} cases "
        f"with {args.workers} worker(s)..."
    )

    print(
        f"vLLM backend: {config['backend_url']} | "
        f"model: {config['deep_think_llm']}"
    )

    print("Press Ctrl+C to force stop.\n")

    executor = ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="stock-worker",
    )

    futures = {
        executor.submit(
            run_single,
            ticker,
            label,
            config,
        ): ticker
        for ticker, label in cases
    }

    try:
        for future in as_completed(futures):

            ticker = futures[future]

            try:
                future.result()

            except Exception as exc:
                print(
                    f"\n❌ Worker failed for {ticker}: "
                    f"{type(exc).__name__}: {exc}"
                )

    except KeyboardInterrupt:
        # 正常情况下 SIGINT handler 会先 os._exit()
        # 这里作为额外 fallback。
        print(
            "\n⚠️ KeyboardInterrupt fallback triggered.",
            flush=True,
        )
        os._exit(130)

    else:
        executor.shutdown(wait=True)

    print(f"\n{'=' * 60}")
    print(
        f"✅ All {len(cases)} cases complete. "
        f"Results in {CASES_DIR}/"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()