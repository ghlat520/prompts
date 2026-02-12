"""配置加载 - 读取YAML配置文件"""

import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # prompts/

def _load_yaml(filename: str) -> dict:
    path = BASE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_scenarios() -> dict:
    return _load_yaml("scenarios.yaml")

def load_market_adapters() -> dict:
    return _load_yaml("market-adapters.yaml")

def load_llm_routing() -> dict:
    return _load_yaml("llm-routing.yaml")

def load_prompt_module(module_name: str) -> str:
    """加载prompt模块文件内容"""
    path = BASE_DIR / "modules" / module_name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_market_for_code(stock_code: str) -> str:
    """根据股票代码判断市场"""
    code = stock_code.strip().upper()
    if code.endswith(".HK") or code.startswith("0") and len(code) == 5:
        return "H"
    if code.endswith(".US") or code.isalpha():
        return "US"
    # A股: 6位纯数字
    return "A"

def get_scenario_dimensions(scenario_name: str) -> list[str]:
    """获取场景对应的维度列表"""
    scenarios = load_scenarios()
    scenario = scenarios.get("scenarios", {}).get(scenario_name)
    if not scenario:
        raise ValueError(f"未知场景: {scenario_name}，可选: {list(scenarios.get('scenarios', {}).keys())}")
    return scenario["dimensions"]

# 维度代号 → 模块文件名映射
DIMENSION_MODULE_MAP = {
    "COMPANY_BASICS": "01-company-basics.md",
    "INDUSTRY": "02-industry.md",
    "BUSINESS_MODEL": "03-business-model.md",
    "FINANCIAL_QUALITY": "04-financial-quality.md",
    "VALUATION": "05-valuation.md",
    "MOAT": "06-moat.md",
    "SUPPLY_CHAIN": "07-supply-chain.md",
    "TECHNICAL": "08-technical.md",
    "DECISION": "09-decision.md",
}

# 维度 → 推荐梯度
DIMENSION_TIER = {
    "COMPANY_BASICS": "T1",
    "INDUSTRY": "T2",
    "BUSINESS_MODEL": "T2",
    "FINANCIAL_QUALITY": "T2",
    "VALUATION": "T3",
    "MOAT": "T2",
    "SUPPLY_CHAIN": "T2",
    "TECHNICAL": "T1",
    "DECISION": "T3",
}
