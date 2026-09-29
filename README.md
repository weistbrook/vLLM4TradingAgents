# vLLM4TradingAgents

使用 A 股多 Agent 分析流程，为 **vLLM OpenAI 兼容服务**生成包含长上下文、工具调用和多轮推理的在线请求负载。本仓库基于 [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) 改造；此处关注的是推理服务的并发与延迟，**不是股票策略收益评测**。

> **当前阶段：负载生成器。** `multirun_cases.py` 可以并发运行股票分析并保存报告和单任务总耗时；它尚未自动采集逐请求 TTFT、ITL、token 数、请求失败率或 vLLM `/metrics`。请不要把 `--workers`、`duration_seconds` 或完成的股票数直接称作 vLLM QPS、吞吐或 P99 延迟。

## 请求链路

```text
multirun_cases.py (多个股票分析任务)
  └─ TradingAgentsGraph (分析师 → 辩论 → 交易员 → 风险评估)
      ├─ A 股数据源 / 本地缓存
      └─ LangChain ChatOpenAI → /v1/chat/completions → vLLM
```

一个 `worker` 执行一个完整股票分析任务；单个任务会发出多次模型请求，也可能同时运行多个 Agent 节点和数据工具。实际同时在 vLLM 中执行的请求数要通过服务端指标确认。

## 快速开始

### 1. 安装客户端

需要 Python 3.10+、可访问的 A 股数据源，以及一台单独安装了 vLLM 并已准备好模型的推理服务器。客户端机器不需要安装 vLLM。

```bash
git clone https://github.com/weistbrook/vLLM4TradingAgents.git
cd vLLM4TradingAgents
uv sync
cp .env.example .env
```

在 `.env` 中增加（或通过环境变量设置）：

```dotenv
OPENAI_COMPATIBLE_API_KEY=EMPTY
```

当前 `openai_compatible` 客户端需要一个非空 API Key。若 vLLM 启动时配置了 `--api-key`，此处必须填写相同的值；未配置服务端鉴权时可使用示例占位值。不要把真实密钥提交到仓库。

### 2. 启动 vLLM 服务

下面以一台有 4 张可用 GPU、模型已保存在服务器上的机器为例。根据显存和模型调整 GPU、并行拓扑、模型路径；模型对外名称要与下文脚本中的名称一致。

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve /path/to/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --dtype bfloat16 \
  --host 127.0.0.1 --port 8000
```

此示例是 **TP=2、DP=2** 的部署方式，不代表仓库已验证所有模型及 vLLM 版本上的性能。若客户端在另一台机器，可用 SSH 隧道把本机 `8000` 端口转发到服务器（保持隧道运行）：

```bash
ssh -L 8000:127.0.0.1:8000 user@server
```

也可以使用受控网络中的服务地址：将 `multirun_cases.py` 的 `build_config()` 内 `backend_url` 改为 `http://SERVER_IP:8000/v1`，并设置合适的服务端监听地址和访问控制。脚本目前使用硬编码 URL，不会从 `.env` 中的 `BACKEND_URL` 自动读取。

在客户端确认路由与模型名：

```bash
curl -s -H "Authorization: Bearer EMPTY" http://127.0.0.1:8000/v1/models
```

### 3. 从单任务逐步增加负载

```bash
uv run python multirun_cases.py 600519 --workers 1
uv run python multirun_cases.py 600519 000858 300750 002594 --workers 2
uv run python multirun_cases.py --workers 8
```

不传股票代码时，脚本运行内置的 64 个股票任务。`--workers` 是同时运行的股票任务上限，而不是并发 HTTP 请求数。还可以指定任意股票代码，但未列入 `TICKERS` 的代码仅使用代码本身作为显示名称。完整参数见 `uv run python multirun_cases.py --help`。

运行前检查脚本顶部的 `TRADE_DATE`，以及 `build_config()` 中的 `backend_url`、`deep_think_llm` 和 `quick_think_llm`；日期和模型目前都是硬编码的。数据供应商均设为 `a_stock`，辩论轮数各为 1。运行会访问外部财经数据源，外部源响应时间也会进入单任务耗时。

运行结果保存在仓库根目录的 `cases/<股票代码>_<名称>_<时间戳>/`，其中包含完整 Markdown 报告与 `summary.json`。`summary.json` 的 `duration_seconds` 是**整个 Agent 任务**的墙钟耗时，含数据抓取、LLM、工具和文件写入；`reports` 中的内容经过截断。`cases/` 被 `.gitignore` 忽略。

其他入口：`run_selected.py` 是使用 DeepSeek 的顺序运行脚本；`examples/run_cases.py` 默认使用 MiniMax。vLLM 压测请从 `multirun_cases.py` 开始。

## 如何开展可复现的服务端评测

建议先固定模型和输入，再逐级增加压力。下面是实验设计建议，不是仓库已实现的自动化基准：

| 阶段 | 控制变量 | 记录内容 |
| --- | --- | --- |
| 冒烟测试 | 1 个股票、`--workers 1` | 模型名、鉴权、工具调用和完整报告是否正常 |
| 并发阶梯 | 同一批股票，`--workers 1/2/4/8` | 总运行时间、成功与失败任务数、vLLM 请求量与排队情况 |
| 部署对比 | 相同输入与并发，分别部署 TP/DP 或调整引擎参数 | 输出 token/s、请求延迟、TTFT/ITL、GPU/KV 使用率 |
| 稳定性 | 固定并发持续运行或重复实验 | 错误、超时、队列积压、显存变化 |

每次实验记录 commit SHA、vLLM/模型版本、GPU 型号、启动命令、日期、股票集合、`--workers`、缓存状态和重复次数。先预热并说明是否保留数据缓存及 prefix cache；不同日期、新闻内容和模型输出长度会改变请求形状。比较两组配置时使用同一输入集合并报告重复实验的波动。

vLLM 的 OpenAI 兼容 API 服务通过 `http://127.0.0.1:8000/metrics` 提供 Prometheus 指标。可在服务器侧保存原始样本并结合 GPU 监控观察 `vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`、token 计数及延迟直方图；具体指标名随版本可能变化，以所用版本的 [vLLM 指标文档](https://docs.vllm.ai/en/latest/usage/metrics/)和实际 `/metrics` 输出为准。

**逐请求的 TTFT/ITL/P99** 需要增加流式请求时间戳或可靠的服务端直方图采集，并明确定义采样范围和百分位算法。当前 `summary.json` 无法计算这些数值。若只想得到纯模型服务能力基线，另用 `vllm bench serve` 测量；本项目用于评估 Agent 工作流带来的真实负载及其系统表现。

## 当前限制与待办

- 缺少自动化的 run ID、全局汇总、逐请求 token/延迟采集和 `/metrics` 快照；完成消息不等于全部成功。
- 股票列表、交易日期、模型与地址写在脚本中；`BACKEND_URL` 环境变量不能覆盖 `multirun_cases.py` 的地址。
- `debug=True` 会打印 Agent 消息；高并发时终端输出交错，也增加客户端开销。
- 所有任务共享本地结果目录、数据缓存和交易记忆日志；重复运行的上下文及热缓存状态可能改变请求规模，并发写入应另行验证。
- 股票分析依赖外部数据源，端到端耗时不能单独归因于 vLLM；需要服务端指标与客户端调用日志一起分析。

## 来源与许可

本仓库由 [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) 改造，其上游为 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)。保留上游版权与声明，参见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
