# Stock Analyzer - AI 股票智能分析系统

输入一个股票代码，AI 自动获取数据 + 生成多维度投资分析报告。

支持 A 股 / 港股 / 美股，可选本地大模型（Ollama）或云端 API（DeepSeek / 通义千问 / Claude 等）。

## 功能概览

### 9 大分析维度

| 维度 | 说明 | LLM 梯度 |
|------|------|---------|
| 公司基础 | 股价、估值、主营业务、产业链位置、风险因子 | T1 (免费) |
| 行业分析 | 行业空间、竞争格局、政策影响 | T2 |
| 商业模式 | 盈利模式、护城河来源、可持续性 | T2 |
| 财务质量 | 营收/利润/现金流、杜邦分析、财务健康度 | T2 |
| 估值分析 | PE/PB/DCF 多维估值、历史分位、合理区间 | T2 |
| 护城河 | 品牌/技术/规模/转换成本/网络效应 | T2 |
| 产业链 | 上下游关系、议价能力、受益标的 | T2 |
| 技术面 | MA/MACD/RSI/KDJ/布林带 + 量价分析 + 操作计划 | T1 (免费) |
| 综合决策 | 汇总所有维度，给出买入/持有/卖出建议 | T3 |

### 5 种预设场景

| 场景 | 维度组合 | 耗时 | 成本 |
|------|---------|------|------|
| **快速扫描** | 公司基础 + 技术面 | ~30s | 免费 |
| **价值挖掘** | 基础 + 行业 + 商业模式 + 财务 + 估值 + 护城河 | ~2min | ~¥0.10 |
| **深度研报** | 全部 9 维度 | ~3min | ~¥0.50 |
| **产业链投资** | 基础 + 行业 + 产业链 | ~1min | ~¥0.06 |
| **买卖时机** | 估值 + 技术面 + 综合决策 | ~1min | ~¥0.08 |

### LLM 三梯度路由

| 梯度 | 模型 | 定位 |
|------|------|------|
| T1 | 百度 ERNIE-Speed / 智谱 GLM-4-Flash / Ollama 本地 | 免费，日常快扫 |
| T2 | DeepSeek-V3 / 通义 Qwen-Plus / Claude Sonnet | 性价比，深度分析 |
| T3 | DeepSeek-R1 / Qwen-Max / Claude Sonnet | 最强推理，关键决策 |

系统按维度推荐梯度自动路由，不可用时自动降级到下一个提供商。

## 目录结构

```
prompts/
├── server.py                  # FastAPI Web 服务（SSE 实时进度推送）
├── static/
│   └── index.html             # 单页前端（Tailwind CSS 暗色终端风格）
├── stock_analyzer/            # 核心分析引擎
│   ├── __init__.py
│   ├── __main__.py            # CLI 入口 (python -m stock_analyzer)
│   ├── cli.py                 # 命令行交互
│   ├── config.py              # 场景/维度/市场配置加载
│   ├── data.py                # 数据获取（baostock K线 + akshare 财务）
│   ├── engine.py              # 分析引擎（prompt 构建 + LLM 调用 + 结果保存）
│   └── llm.py                 # 多提供商 LLM 客户端（自动路由 + 降级）
├── modules/                   # 9 大维度的 prompt 模板
│   ├── 01-company-basics.md
│   ├── 02-industry.md
│   ├── 03-business-model.md
│   ├── 04-financial-quality.md
│   ├── 05-valuation.md
│   ├── 06-moat.md
│   ├── 07-supply-chain.md
│   ├── 08-technical.md
│   └── 09-decision.md
├── scenarios.yaml             # 预设场景定义
├── llm-routing.yaml           # LLM 梯度路由配置
├── market-adapters.yaml       # A股/港股/美股市场适配
├── .env.example               # 环境变量模板
├── setup-ollama.sh            # Ollama 本地部署脚本
└── test-data/                 # 测试数据样本
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn akshare baostock numpy
```

### 2. 配置 LLM

**方式 A：本地大模型（推荐，免费）**

```bash
# 安装 Ollama 并拉取模型
bash setup-ollama.sh

# 或手动：
# brew install ollama
# ollama pull qwen2.5:14b
```

**方式 B：云端 API**

```bash
cp .env.example .env
# 编辑 .env，填入 API Key：
# DEEPSEEK_API_KEY=sk-xxx
# DASHSCOPE_API_KEY=sk-xxx
# ANTHROPIC_API_KEY=sk-xxx
```

### 3. 启动服务

```bash
python server.py
# 浏览器打开 http://localhost:8899
```

### 4. 命令行模式

```bash
python -m stock_analyzer 300054 --scenario quick_scan
```

## 数据来源

| 数据 | 来源 | 协议 |
|------|------|------|
| 日 K 线（前复权） | baostock | TCP（不受 HTTP 代理影响） |
| 公司信息 / 财务报表 | akshare (东方财富) | HTTP |
| 技术指标 | 本地计算（numpy） | - |

## 截图

![分析结果](https://raw.githubusercontent.com/ghlat520/prompts/main/stock-analyzer-final.png)
