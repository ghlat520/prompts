#!/usr/bin/env python3
"""股票分析 Web 服务 - FastAPI + SSE 实时进度"""

# 最先执行：清除代理，确保国内数据源（东方财富等）直连
import os
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

import asyncio
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 加载 .env 文件
def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)

_load_dotenv()

# 确保 stock_analyzer 可导入
sys.path.insert(0, str(Path(__file__).parent))

from stock_analyzer.config import (
    load_scenarios, load_market_adapters,
    get_market_for_code, get_scenario_dimensions,
    DIMENSION_MODULE_MAP, DIMENSION_TIER,
)
from stock_analyzer.data import fetch_stock_data, calculate_indicators, build_fundamental_summary
from stock_analyzer.llm import LLMClient

# ── 全局状态 ──
tasks: dict[str, dict] = {}
progress_queues: dict[str, asyncio.Queue] = {}
executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="股票智能分析", lifespan=lifespan)


# ── 请求模型 ──
class AnalyzeRequest(BaseModel):
    stock_code: str
    scenario: str = "quick_scan"
    max_tier: str = "T3"


# ── API 路由 ──
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/scenarios")
async def get_scenarios():
    scenarios = load_scenarios()
    return scenarios.get("scenarios", {})


@app.get("/api/dimensions")
async def get_dimensions():
    return {
        "dimensions": DIMENSION_MODULE_MAP,
        "tiers": DIMENSION_TIER,
    }


@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest):
    task_id = str(uuid.uuid4())[:8]
    queue = asyncio.Queue()
    progress_queues[task_id] = queue
    tasks[task_id] = {
        "task_id": task_id,
        "stock_code": req.stock_code,
        "scenario": req.scenario,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        _run_analysis,
        task_id, req.stock_code, req.scenario, req.max_tier, loop
    )

    return {"task_id": task_id, "status": "pending"}


@app.get("/api/progress/{task_id}")
async def stream_progress(task_id: str):
    if task_id not in progress_queues:
        raise HTTPException(404, "Task not found")

    async def event_gen():
        queue = progress_queues[task_id]
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg.get("type") in ("completed", "error"):
                break
        # 清理
        progress_queues.pop(task_id, None)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/tasks")
async def list_tasks():
    return list(tasks.values())


# ── 分析执行（在线程池中运行） ──
def _run_analysis(task_id: str, stock_code: str, scenario: str,
                  max_tier: str, loop: asyncio.AbstractEventLoop):
    """同步执行分析，通过队列推送进度"""

    def send(msg: dict):
        asyncio.run_coroutine_threadsafe(
            progress_queues[task_id].put(msg), loop
        )

    try:
        tasks[task_id]["status"] = "running"

        # 1. 确定维度
        dimensions = get_scenario_dimensions(scenario)
        has_decision = "DECISION" in dimensions
        if has_decision:
            dimensions = [d for d in dimensions if d != "DECISION"]

        send({"type": "info", "message": f"场景: {scenario} → 维度: {', '.join(dimensions)}"})

        # 2. 市场判断
        market = get_market_for_code(stock_code)
        adapters = load_market_adapters()
        market_vars = adapters.get("markets", {}).get(market, {})
        send({"type": "info", "message": f"市场: {market_vars.get('market', market)}"})

        # 3. 获取数据
        send({"type": "progress", "phase": "data", "message": "正在获取股票数据..."})
        raw_data = fetch_stock_data(stock_code, market)
        if raw_data.get("_errors"):
            send({"type": "warning", "message": f"部分数据获取失败: {raw_data['_errors']}"})

        stock_name = raw_data.get("stock_name", stock_code)
        summary = build_fundamental_summary(raw_data)
        indicators = calculate_indicators(raw_data)

        send({
            "type": "data_ready",
            "stock_name": stock_name,
            "price": summary.get("current_price", "N/A"),
            "pe": summary.get("pe_ttm", "N/A"),
            "market_cap": summary.get("total_mv_yi", "N/A"),
            "industry": summary.get("industry", "未知"),
            "trend": indicators.get("trend", "N/A") if isinstance(indicators, dict) else "N/A",
        })

        # 4. LLM 客户端
        llm = LLMClient(max_tier=max_tier)
        providers = llm.available_providers()
        if not providers:
            send({"type": "error", "message": "未检测到LLM API Key，请设置 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 等环境变量"})
            tasks[task_id]["status"] = "error"
            return

        send({"type": "info", "message": f"LLM提供商: {', '.join(providers)} | 最高梯度: {max_tier}"})

        # 5. 逐维度分析
        results = {}
        total_dims = len(dimensions) + (1 if has_decision else 0)

        for i, dim in enumerate(dimensions):
            tier = DIMENSION_TIER.get(dim, "T2")
            send({
                "type": "dimension_start",
                "dimension": dim,
                "index": i + 1,
                "total": total_dims,
                "tier": tier,
                "message": f"[{i+1}/{total_dims}] 分析 {dim}...",
            })

            t0 = time.time()
            try:
                from stock_analyzer.engine import _build_prompt
                system_prompt, user_prompt = _build_prompt(
                    dim, stock_code, stock_name, market, market_vars,
                    summary, indicators, raw_data
                )
                resp = llm.call(system_prompt, user_prompt, tier=tier)
                elapsed = int((time.time() - t0) * 1000)
                results[dim] = {
                    "content": resp.content,
                    "model": resp.model,
                    "provider": resp.provider,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_yuan": resp.cost_yuan,
                    "elapsed_ms": elapsed,
                }
                send({
                    "type": "dimension_done",
                    "dimension": dim,
                    "index": i + 1,
                    "total": total_dims,
                    "content": resp.content,
                    "model": resp.model,
                    "provider": resp.provider,
                    "tokens": resp.input_tokens + resp.output_tokens,
                    "cost": resp.cost_yuan,
                    "elapsed_ms": elapsed,
                })
            except Exception as e:
                elapsed = int((time.time() - t0) * 1000)
                results[dim] = {"content": f"[ERROR] {e}", "error": str(e)}
                send({
                    "type": "dimension_error",
                    "dimension": dim,
                    "error": str(e),
                    "elapsed_ms": elapsed,
                })

        # 6. DECISION 汇总
        if has_decision:
            send({
                "type": "dimension_start",
                "dimension": "DECISION",
                "index": total_dims,
                "total": total_dims,
                "tier": "T3",
                "message": f"[{total_dims}/{total_dims}] 综合决策汇总...",
            })
            try:
                from stock_analyzer.engine import _build_decision_prompt, _aggregate_results
                all_text = _aggregate_results(results, stock_code, stock_name)
                sys_p, usr_p = _build_decision_prompt(stock_code, stock_name, market_vars, summary, all_text)
                t0 = time.time()
                resp = llm.call(sys_p, usr_p, tier="T3")
                elapsed = int((time.time() - t0) * 1000)
                results["DECISION"] = {
                    "content": resp.content,
                    "model": resp.model,
                    "provider": resp.provider,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_yuan": resp.cost_yuan,
                    "elapsed_ms": elapsed,
                }
                send({
                    "type": "dimension_done",
                    "dimension": "DECISION",
                    "index": total_dims,
                    "total": total_dims,
                    "content": resp.content,
                    "model": resp.model,
                    "provider": resp.provider,
                    "tokens": resp.input_tokens + resp.output_tokens,
                    "cost": resp.cost_yuan,
                    "elapsed_ms": elapsed,
                })
            except Exception as e:
                results["DECISION"] = {"content": f"[ERROR] {e}", "error": str(e)}
                send({"type": "dimension_error", "dimension": "DECISION", "error": str(e)})

        # 7. 保存结果
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = str(Path(__file__).parent / "output" / f"{stock_code}_{date_str}")
        from stock_analyzer.engine import _save_results
        _save_results(results, stock_code, stock_name, summary, output_dir)

        # 8. 汇总
        total_in = sum(r.get("input_tokens", 0) for r in results.values())
        total_out = sum(r.get("output_tokens", 0) for r in results.values())
        total_cost = sum(r.get("cost_yuan", 0) for r in results.values())

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["stock_name"] = stock_name

        send({
            "type": "completed",
            "stock_code": stock_code,
            "stock_name": stock_name,
            "dimensions": list(results.keys()),
            "total_tokens": total_in + total_out,
            "total_cost": round(total_cost, 4),
            "output_dir": output_dir,
        })

    except Exception as e:
        import traceback
        tasks[task_id]["status"] = "error"
        send({"type": "error", "message": f"分析失败: {e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8899))
    print(f"启动股票分析 Web 服务: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
