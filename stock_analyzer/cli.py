#!/usr/bin/env python3
"""
股票智能分析CLI

用法:
    # 单股分析 - 快速扫描(默认)
    python -m stock_analyzer 300054

    # 指定场景
    python -m stock_analyzer 300054 --scenario value_discovery

    # 指定维度
    python -m stock_analyzer 300054 --dims COMPANY_BASICS,VALUATION,MOAT

    # 指定最高梯度(控制成本)
    python -m stock_analyzer 300054 --scenario deep_research --tier T2

    # 批量分析
    python -m stock_analyzer 300054,600519,688019 --scenario quick_scan

    # 列出可用场景
    python -m stock_analyzer --list-scenarios

场景:
    quick_scan      快速扫描 (基础+技术面)
    value_discovery 价值挖掘 (基础+行业+业务+财务+估值+护城河)
    deep_research   深度研报 (全部9维度)
    supply_chain_invest 产业链投资 (基础+行业+产业链)
    timing          买卖时机 (估值+技术面+综合决策)
"""

import argparse
import sys
import os

# 确保包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_analyzer.engine import analyze_stock, analyze_stocks
from stock_analyzer.config import load_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="股票智能分析 - 模块化提示词 + LLM多梯度路由",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("codes", nargs="?", help="股票代码，多只用逗号分隔 (如 300054,600519)")
    parser.add_argument("--scenario", "-s", help="分析场景 (quick_scan/value_discovery/deep_research/supply_chain_invest/timing)")
    parser.add_argument("--dims", "-d", help="分析维度，逗号分隔 (如 COMPANY_BASICS,VALUATION)")
    parser.add_argument("--tier", "-t", default="T3", help="最高LLM梯度 (T1/T2/T3，默认T3)")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--sequential", action="store_true", help="串行调用(调试用)")
    parser.add_argument("--list-scenarios", action="store_true", help="列出可用场景")
    parser.add_argument("--list-dims", action="store_true", help="列出可用维度")

    args = parser.parse_args()

    if args.list_scenarios:
        _print_scenarios()
        return

    if args.list_dims:
        _print_dimensions()
        return

    if not args.codes:
        parser.print_help()
        return

    codes = [c.strip() for c in args.codes.split(",")]
    dimensions = [d.strip() for d in args.dims.split(",")] if args.dims else None

    if len(codes) == 1:
        analyze_stock(
            stock_code=codes[0],
            dimensions=dimensions,
            scenario=args.scenario,
            max_tier=args.tier,
            output_dir=args.output,
            parallel=not args.sequential,
        )
    else:
        analyze_stocks(
            stock_codes=codes,
            scenario=args.scenario or "quick_scan",
            max_tier=args.tier,
            output_dir=args.output,
        )


def _print_scenarios():
    scenarios = load_scenarios()
    print("可用分析场景:\n")
    for key, cfg in scenarios.get("scenarios", {}).items():
        dims = ", ".join(cfg.get("dimensions", []))
        print(f"  {key:25s} {cfg['name']}")
        print(f"  {'':25s} 维度: {dims}")
        print(f"  {'':25s} 推荐梯度: {cfg.get('recommended_tier', 'T2')} | 成本: {cfg.get('estimated_cost', '?')}")
        print()


def _print_dimensions():
    from stock_analyzer.config import DIMENSION_MODULE_MAP, DIMENSION_TIER
    print("可用分析维度:\n")
    for dim, module in DIMENSION_MODULE_MAP.items():
        tier = DIMENSION_TIER.get(dim, "T2")
        print(f"  {dim:25s} [{tier}] {module}")


if __name__ == "__main__":
    main()
