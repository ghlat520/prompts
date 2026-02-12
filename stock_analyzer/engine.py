"""分析引擎 - 编排数据获取、prompt填充、LLM调用、报告生成"""

import json
import re
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    load_prompt_module, load_market_adapters, load_scenarios,
    get_market_for_code, get_scenario_dimensions,
    DIMENSION_MODULE_MAP, DIMENSION_TIER,
)
from .data import fetch_stock_data, calculate_indicators, build_fundamental_summary
from .llm import LLMClient, LLMResponse


def analyze_stock(
    stock_code: str,
    dimensions: list[str] = None,
    scenario: str = None,
    max_tier: str = "T3",
    output_dir: str = None,
    parallel: bool = True,
) -> dict:
    """
    分析单只股票

    Args:
        stock_code: 股票代码 (如 "300054", "00700", "AAPL")
        dimensions: 分析维度列表 (如 ["COMPANY_BASICS", "VALUATION"])
        scenario: 预设场景 (如 "quick_scan", "value_discovery", "deep_research")
        max_tier: 最高允许LLM梯度 ("T1"/"T2"/"T3")
        output_dir: 输出目录，None则输出到 prompts/output/{code}_{date}/
        parallel: 是否并行调用LLM

    Returns:
        dict: {dimension: {content, model, tokens, cost, elapsed}}
    """
    # 1. 确定维度
    if scenario:
        dimensions = get_scenario_dimensions(scenario)
        print(f"[场景] {scenario} → 维度: {dimensions}")
    elif not dimensions:
        dimensions = ["COMPANY_BASICS", "TECHNICAL"]  # 默认快速扫描
        print(f"[默认] 快速扫描 → 维度: {dimensions}")

    # DECISION必须最后，且需要其他维度结果
    has_decision = "DECISION" in dimensions
    if has_decision:
        dimensions = [d for d in dimensions if d != "DECISION"]

    # 2. 判断市场
    market = get_market_for_code(stock_code)
    adapters = load_market_adapters()
    market_vars = adapters.get("markets", {}).get(market, {})
    print(f"[市场] {market} ({market_vars.get('market', '')})")

    # 3. 获取数据
    print(f"\n[数据] 正在获取 {stock_code} 数据...")
    raw_data = fetch_stock_data(stock_code, market)
    if raw_data.get("_errors"):
        print(f"  [WARN] 数据获取部分失败: {raw_data['_errors']}")

    stock_name = raw_data.get("stock_name", stock_code)
    summary = build_fundamental_summary(raw_data)
    indicators = calculate_indicators(raw_data)

    print(f"  股票: {stock_name} | 价格: {summary.get('current_price', 'N/A')} | "
          f"PE: {summary.get('pe_ttm', 'N/A')} | 市值: {summary.get('total_mv_yi', 'N/A')}亿")
    if isinstance(indicators, dict) and "error" not in indicators:
        print(f"  MA5={indicators['ma5']} MA10={indicators['ma10']} "
              f"MA20={indicators['ma20']} | 趋势: {indicators['trend']}")

    # 4. LLM客户端
    llm = LLMClient(max_tier=max_tier)
    providers = llm.available_providers()
    if not providers:
        print(f"\n[ERROR] 未检测到任何LLM API Key！请设置环境变量:")
        print(f"  export DEEPSEEK_API_KEY=sk-xxx     # T2/T3")
        print(f"  export DASHSCOPE_API_KEY=sk-xxx     # 阿里 T2/T3")
        print(f"  export BAIDU_API_KEY=id:secret       # 百度 T1")
        print(f"  export ZHIPU_API_KEY=xxx             # 智谱 T1")
        print(f"  export OPENAI_API_KEY=xxx + OPENAI_BASE_URL=xxx  # 兼容API")
        return {"error": "no_api_key"}
    print(f"[LLM] 可用提供商: {providers} | 最高梯度: {max_tier}")

    # 5. 构建prompt并调用
    results = {}

    def _run_dimension(dim: str) -> tuple[str, dict]:
        tier = DIMENSION_TIER.get(dim, "T2")
        system_prompt, user_prompt = _build_prompt(
            dim, stock_code, stock_name, market, market_vars,
            summary, indicators, raw_data
        )
        print(f"  [{dim}] 调用中 (推荐{tier}, 实际{llm._min_tier(tier, max_tier)})...")
        t0 = time.time()
        try:
            resp = llm.call(system_prompt, user_prompt, tier=tier)
            elapsed = int((time.time() - t0) * 1000)
            print(f"  [{dim}] 完成 | {resp.model}({resp.provider}) | "
                  f"{resp.input_tokens}+{resp.output_tokens}tok | "
                  f"¥{resp.cost_yuan} | {elapsed}ms")
            return dim, {
                "content": resp.content,
                "model": resp.model,
                "provider": resp.provider,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_yuan": resp.cost_yuan,
                "elapsed_ms": elapsed,
            }
        except Exception as e:
            print(f"  [{dim}] 失败: {e}")
            return dim, {"content": f"[ERROR] {e}", "error": str(e)}

    print(f"\n[分析] 开始{len(dimensions)}个维度分析...")
    if parallel and len(dimensions) > 1:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_run_dimension, d): d for d in dimensions}
            for f in as_completed(futures):
                dim, result = f.result()
                results[dim] = result
    else:
        for dim in dimensions:
            dim_name, result = _run_dimension(dim)
            results[dim_name] = result

    # 6. DECISION汇总
    if has_decision:
        print(f"\n[DECISION] 汇总所有维度结果...")
        all_results_text = _aggregate_results(results, stock_code, stock_name)
        system_prompt, user_prompt = _build_decision_prompt(
            stock_code, stock_name, market_vars, summary, all_results_text
        )
        try:
            resp = llm.call(system_prompt, user_prompt, tier="T3")
            results["DECISION"] = {
                "content": resp.content,
                "model": resp.model,
                "provider": resp.provider,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_yuan": resp.cost_yuan,
                "elapsed_ms": resp.elapsed_ms,
            }
            print(f"  [DECISION] 完成 | {resp.model}({resp.provider}) | "
                  f"{resp.input_tokens}+{resp.output_tokens}tok | ¥{resp.cost_yuan}")
        except Exception as e:
            print(f"  [DECISION] 失败: {e}")
            results["DECISION"] = {"content": f"[ERROR] {e}", "error": str(e)}

    # 7. 输出
    if output_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = str(Path(__file__).parent.parent / "output" / f"{stock_code}_{date_str}")

    _save_results(results, stock_code, stock_name, summary, output_dir)

    # 8. 汇总统计
    total_in = sum(r.get("input_tokens", 0) for r in results.values())
    total_out = sum(r.get("output_tokens", 0) for r in results.values())
    total_cost = sum(r.get("cost_yuan", 0) for r in results.values())
    print(f"\n[完成] {stock_name}({stock_code}) 分析完毕")
    print(f"  维度: {list(results.keys())}")
    print(f"  Token: 输入{total_in} + 输出{total_out} = {total_in + total_out}")
    print(f"  成本: ¥{round(total_cost, 4)}")
    print(f"  输出: {output_dir}/")

    return results


def analyze_stocks(
    stock_codes: list[str],
    scenario: str = "quick_scan",
    max_tier: str = "T2",
    output_dir: str = None,
) -> dict:
    """批量分析多只股票"""
    all_results = {}
    for i, code in enumerate(stock_codes):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(stock_codes)}] 分析 {code}")
        print(f"{'='*60}")
        result = analyze_stock(code, scenario=scenario, max_tier=max_tier, output_dir=output_dir)
        all_results[code] = result
    return all_results


def _build_prompt(dim, stock_code, stock_name, market, market_vars,
                  summary, indicators, raw_data) -> tuple[str, str]:
    """构建维度prompt（系统+用户）"""
    module_file = DIMENSION_MODULE_MAP.get(dim, "")
    if not module_file:
        raise ValueError(f"未知维度: {dim}")

    module_text = load_prompt_module(module_file)

    # 提取系统提示词和用户提示词
    system_prompt, user_prompt = _parse_module(module_text)

    # 基础变量替换
    variables = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "market": market_vars.get("market", "A股"),
        "market_specific_factors": market_vars.get("market_specific_factors", ""),
        "trading_rules": market_vars.get("trading_rules", ""),
        "industry": summary.get("industry", "未知"),
        "date": summary.get("date", datetime.now().strftime("%Y-%m-%d")),
    }

    # 维度特定数据注入
    if dim == "TECHNICAL":
        variables["technical_indicators"] = json.dumps(indicators, ensure_ascii=False, indent=2)
        kline = raw_data.get("kline", [])[-10:]
        variables["kline_data"] = "\n".join(
            f"{k['date']} open={k['open']} close={k['close']} high={k['high']} low={k['low']} vol={k['volume']} turnover={k.get('turnover',0)}%"
            for k in kline
        )
    elif dim == "VALUATION":
        variables["fundamental_data"] = json.dumps(summary, ensure_ascii=False, indent=2)
        variables["comparable_companies"] = "请根据行业知识列出可比公司及其PE/PB"
    elif dim in ("COMPANY_BASICS", "BUSINESS_MODEL", "FINANCIAL_QUALITY", "MOAT", "SUPPLY_CHAIN", "INDUSTRY"):
        variables["fundamental_data"] = json.dumps(summary, ensure_ascii=False, indent=2)
        # 注入财务原始数据摘要
        ps = raw_data.get("profit_sheet", [])[:4]
        cf = raw_data.get("cashflow", [])[:4]
        bs = raw_data.get("balance_sheet", [])[:2]
        fin_text = []
        for r in ps:
            period = r.get("REPORT_DATE_NAME", "")
            rev = r.get("TOTAL_OPERATE_INCOME", 0) or 0
            net = r.get("NETPROFIT", 0) or 0
            fin_text.append(f"{period}: 营收{rev/1e8:.2f}亿, 净利{net/1e8:.2f}亿")
        for r in cf:
            period = r.get("REPORT_DATE_NAME", "")
            ocf = r.get("NETCASH_OPERATE", 0) or 0
            fin_text.append(f"{period} 经营现金流: {ocf/1e8:.2f}亿")
        variables["financial_details"] = "\n".join(fin_text)

    # 替换变量
    for k, v in variables.items():
        system_prompt = system_prompt.replace(f"{{{k}}}", str(v))
        user_prompt = user_prompt.replace(f"{{{k}}}", str(v))

    # 将未替换的财务数据附加到user_prompt
    if dim != "TECHNICAL" and "fundamental_data" not in user_prompt:
        user_prompt += f"\n\n**基本面数据：**\n```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```"
        if variables.get("financial_details"):
            user_prompt += f"\n\n**财务明细：**\n{variables['financial_details']}"

    return system_prompt, user_prompt


def _build_decision_prompt(stock_code, stock_name, market_vars, summary, all_results_text):
    """构建DECISION维度prompt"""
    module_text = load_prompt_module("09-decision.md")
    system_prompt, user_prompt = _parse_module(module_text)

    variables = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "market": market_vars.get("market", "A股"),
        "all_module_results": all_results_text,
        "date": summary.get("date", ""),
        "current_price": summary.get("current_price", ""),
    }

    for k, v in variables.items():
        system_prompt = system_prompt.replace(f"{{{k}}}", str(v))
        user_prompt = user_prompt.replace(f"{{{k}}}", str(v))

    return system_prompt, user_prompt


def _parse_module(module_text: str) -> tuple[str, str]:
    """从模块Markdown中提取系统提示词和用户提示词"""
    # 找系统提示词部分
    sys_match = re.search(r'## 系统提示词\s*\n(.*?)(?=\n## 用户提示词|\n---)', module_text, re.DOTALL)
    usr_match = re.search(r'## 用户提示词\s*\n(.*?)$', module_text, re.DOTALL)

    system_prompt = sys_match.group(1).strip() if sys_match else ""
    user_prompt = usr_match.group(1).strip() if usr_match else module_text

    return system_prompt, user_prompt


def _aggregate_results(results: dict, stock_code: str, stock_name: str) -> str:
    """汇总所有维度结果为文本"""
    lines = []
    for dim in ["COMPANY_BASICS", "INDUSTRY", "BUSINESS_MODEL", "FINANCIAL_QUALITY",
                 "VALUATION", "MOAT", "SUPPLY_CHAIN", "TECHNICAL"]:
        if dim not in results:
            continue
        content = results[dim].get("content", "")
        # 截取前2000字作为摘要
        if len(content) > 2000:
            content = content[:2000] + "\n...(已截断)"
        lines.append(f"【维度-{dim}】\n{content}\n")
    return "\n".join(lines)


def _save_results(results: dict, stock_code: str, stock_name: str,
                  summary: dict, output_dir: str):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 各维度单独文件
    for dim, data in results.items():
        content = data.get("content", "")
        filename = f"{dim.lower()}.md"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    # 合并报告
    merged = []
    merged.append(f"# {stock_name}({stock_code}) 综合分析报告\n")
    merged.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    merged.append("---\n")

    dim_order = ["COMPANY_BASICS", "INDUSTRY", "BUSINESS_MODEL", "FINANCIAL_QUALITY",
                 "VALUATION", "MOAT", "SUPPLY_CHAIN", "TECHNICAL", "DECISION"]
    for dim in dim_order:
        if dim in results and "error" not in results[dim]:
            merged.append(results[dim]["content"])
            merged.append("\n\n---\n\n")

    with open(os.path.join(output_dir, "full_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(merged))

    # 统计信息
    stats = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "generated_at": datetime.now().isoformat(),
        "dimensions": {},
        "summary": summary,
    }
    for dim, data in results.items():
        stats["dimensions"][dim] = {
            "model": data.get("model", ""),
            "provider": data.get("provider", ""),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "cost_yuan": data.get("cost_yuan", 0),
            "elapsed_ms": data.get("elapsed_ms", 0),
        }
    total_cost = sum(d.get("cost_yuan", 0) for d in results.values())
    stats["total_cost_yuan"] = round(total_cost, 4)

    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"  [保存] {output_dir}/full_report.md")
