"""股票智能分析工具 - 模块化提示词体系 + LLM多梯度路由"""

from .engine import analyze_stock, analyze_stocks

__all__ = ["analyze_stock", "analyze_stocks"]
__version__ = "0.1.0"
