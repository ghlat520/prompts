"""数据获取 - baostock(K线) + akshare(财务) + 技术指标计算"""

import os
import akshare as ak
import baostock as bs
import numpy as np
from datetime import datetime, timedelta


def fetch_stock_data(stock_code: str, market: str = "A") -> dict:
    """获取股票全套数据，返回结构化字典"""
    data = {
        "stock_code": stock_code,
        "market": market,
        "fetch_time": datetime.now().isoformat(),
        "_errors": [],
    }

    if market == "A":
        _fetch_a_share(stock_code, data)
    elif market == "H":
        _fetch_hk_share(stock_code, data)
    elif market == "US":
        _fetch_us_share(stock_code, data)

    return data


def _fetch_a_share(code: str, data: dict):
    """A股数据获取"""
    errors = data["_errors"]
    symbol = f"SZ{code}" if code.startswith("3") or code.startswith("0") else f"SH{code}"

    # 1. 公司信息
    try:
        df = ak.stock_individual_info_em(symbol=code)
        info = {}
        for _, r in df.iterrows():
            info[r.iloc[0]] = r.iloc[1]
        data["company_info"] = info
        data["stock_name"] = info.get("股票简称", code)
    except Exception as e:
        errors.append(f"company_info: {e}")
        data["stock_name"] = code

    # 2. 日K线（最近400日，baostock：独立TCP协议，不受HTTP代理影响）
    try:
        bs_prefix = "sz" if code.startswith("3") or code.startswith("0") else "sh"
        bs_code = f"{bs_prefix}.{code}"
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")

        lg = bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start, end_date=end,
            frequency="d", adjustflag="2",  # 前复权
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        data["kline"] = [{
            "date": r[0],
            "open": float(r[1]) if r[1] else 0,
            "close": float(r[4]) if r[4] else 0,
            "high": float(r[2]) if r[2] else 0,
            "low": float(r[3]) if r[3] else 0,
            "volume": float(r[5]) if r[5] else 0,
            "amount": float(r[6]) if r[6] else 0,
            "turnover": float(r[7]) if r[7] else 0,
        } for r in rows if r[4]]  # 跳过 close 为空的行
    except Exception as e:
        errors.append(f"kline: {e}")

    # 3. 利润表
    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
        data["profit_sheet"] = df.head(8).to_dict(orient="records")
    except Exception as e:
        errors.append(f"profit_sheet: {e}")

    # 4. 现金流量表
    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)
        data["cashflow"] = df.head(8).to_dict(orient="records")
    except Exception as e:
        errors.append(f"cashflow: {e}")

    # 5. 资产负债表
    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
        data["balance_sheet"] = df.head(8).to_dict(orient="records")
    except Exception as e:
        errors.append(f"balance_sheet: {e}")

    # 6. 财务摘要
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        data["financials"] = df.head(8).to_dict(orient="records")
    except Exception as e:
        errors.append(f"financials: {e}")

    # 7. 分红
    try:
        df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
        data["dividends"] = df.head(10).to_dict(orient="records")
    except Exception as e:
        errors.append(f"dividends: {e}")


def _fetch_hk_share(code: str, data: dict):
    """港股数据获取"""
    errors = data["_errors"]
    try:
        df = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")
        data["kline_raw"] = df
        data["kline"] = [{
            "date": str(r.get("日期", "")),
            "open": float(r.get("开盘", 0)),
            "close": float(r.get("收盘", 0)),
            "high": float(r.get("最高", 0)),
            "low": float(r.get("最低", 0)),
            "volume": float(r.get("成交量", 0)),
            "amount": float(r.get("成交额", 0)),
            "turnover": float(r.get("换手率", 0)),
        } for _, r in df.tail(250).iterrows()]
    except Exception as e:
        errors.append(f"kline: {e}")


def _fetch_us_share(code: str, data: dict):
    """美股数据获取（通过yfinance或akshare）"""
    errors = data["_errors"]
    try:
        df = ak.stock_us_hist(symbol=code, period="daily", adjust="qfq")
        data["kline_raw"] = df
        data["kline"] = [{
            "date": str(r.get("日期", "")),
            "open": float(r.get("开盘", 0)),
            "close": float(r.get("收盘", 0)),
            "high": float(r.get("最高", 0)),
            "low": float(r.get("最低", 0)),
            "volume": float(r.get("成交量", 0)),
        } for _, r in df.tail(250).iterrows()]
    except Exception as e:
        errors.append(f"kline: {e}")


def calculate_indicators(data: dict) -> dict:
    """计算技术指标，返回结构化指标字典"""
    kline = data.get("kline", [])
    if len(kline) < 30:
        return {"error": "K线数据不足30条，无法计算技术指标"}

    closes = np.array([k["close"] for k in kline], dtype=float)
    volumes = np.array([k["volume"] for k in kline], dtype=float)
    highs = np.array([k["high"] for k in kline], dtype=float)
    lows = np.array([k["low"] for k in kline], dtype=float)

    latest = closes[-1]

    # MA
    ma5 = float(np.mean(closes[-5:]))
    ma10 = float(np.mean(closes[-10:]))
    ma20 = float(np.mean(closes[-20:]))
    ma60 = float(np.mean(closes[-60:])) if len(closes) >= 60 else None

    # MA偏离率
    ma5_bias = (latest - ma5) / ma5 * 100

    # 均量线
    vol_ma5 = float(np.mean(volumes[-5:]))
    vol_ma10 = float(np.mean(volumes[-10:]))

    # MACD (12, 26, 9)
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    macd_hist = 2 * (dif - dea)

    # RSI (6, 12, 24)
    rsi6 = _rsi(closes, 6)
    rsi12 = _rsi(closes, 12)
    rsi24 = _rsi(closes, 24)

    # KDJ (9, 3, 3)
    k, d, j = _kdj(highs, lows, closes)

    # 布林带 (20, 2)
    bb_mid = ma20
    bb_std = float(np.std(closes[-20:]))
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # 趋势判断
    if ma5 > ma10 and ma10 > ma20:
        trend = "多头排列"
    elif ma5 < ma10 and ma10 < ma20:
        trend = "空头排列"
    else:
        trend = "震荡盘整"

    # 近期高低点
    high_20d = float(np.max(highs[-20:]))
    low_20d = float(np.min(lows[-20:]))

    indicators = {
        "date": kline[-1]["date"],
        "close": latest,
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2) if ma60 else None,
        "ma5_bias_pct": round(ma5_bias, 2),
        "trend": trend,
        "volume": float(volumes[-1]),
        "vol_ma5": round(vol_ma5, 0),
        "vol_ma10": round(vol_ma10, 0),
        "vol_ratio": round(volumes[-1] / vol_ma5, 2) if vol_ma5 > 0 else None,
        "macd_dif": round(float(dif[-1]), 4),
        "macd_dea": round(float(dea[-1]), 4),
        "macd_hist": round(float(macd_hist[-1]), 4),
        "macd_signal": "金叉" if dif[-1] > dea[-1] and dif[-2] <= dea[-2] else ("死叉" if dif[-1] < dea[-1] and dif[-2] >= dea[-2] else ("DIF>DEA" if dif[-1] > dea[-1] else "DIF<DEA")),
        "rsi6": round(rsi6, 2),
        "rsi12": round(rsi12, 2),
        "rsi24": round(rsi24, 2),
        "kdj_k": round(k, 2),
        "kdj_d": round(d, 2),
        "kdj_j": round(j, 2),
        "boll_upper": round(bb_upper, 2),
        "boll_mid": round(bb_mid, 2),
        "boll_lower": round(bb_lower, 2),
        "high_20d": round(high_20d, 2),
        "low_20d": round(low_20d, 2),
        "turnover": kline[-1].get("turnover", 0),
    }

    return indicators


def _ema(data, period):
    """指数移动平均"""
    arr = np.array(data, dtype=float)
    ema = np.zeros_like(arr)
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def _rsi(closes, period):
    """RSI"""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def _kdj(highs, lows, closes, n=9, m1=3, m2=3):
    """KDJ指标"""
    length = len(closes)
    if length < n:
        return 50.0, 50.0, 50.0

    rsv_list = []
    for i in range(n - 1, length):
        h = np.max(highs[i - n + 1:i + 1])
        l = np.min(lows[i - n + 1:i + 1])
        if h == l:
            rsv_list.append(50.0)
        else:
            rsv_list.append((closes[i] - l) / (h - l) * 100)

    k, d = 50.0, 50.0
    for rsv in rsv_list:
        k = (2 * k + rsv) / 3
        d = (2 * d + k) / 3
    j = 3 * k - 2 * d
    return k, d, j


def build_fundamental_summary(data: dict) -> dict:
    """从原始数据构建基本面摘要"""
    summary = {
        "stock_code": data.get("stock_code", ""),
        "stock_name": data.get("stock_name", ""),
        "market": data.get("market", "A"),
    }

    # 公司信息
    info = data.get("company_info", {})
    kline = data.get("kline", [])

    if kline:
        latest = kline[-1]
        summary["date"] = latest["date"]
        summary["current_price"] = latest["close"]
    if info:
        total_shares = float(info.get("总股本", 0))
        summary["total_shares_yi"] = round(total_shares / 1e8, 2)
        summary["total_mv_yi"] = round(float(info.get("总市值", 0)) / 1e8, 2)
        summary["industry"] = info.get("行业", "未知")

    # 财务数据
    ps = data.get("profit_sheet", [])
    cf = data.get("cashflow", [])
    bs = data.get("balance_sheet", [])

    def _find_period(records, keyword):
        for r in records:
            if keyword in str(r.get("REPORT_DATE_NAME", "")):
                return r
        return {}

    annual = _find_period(ps, "年报")
    q3 = _find_period(ps, "三季报")

    if annual:
        rev = annual.get("TOTAL_OPERATE_INCOME", 0) or 0
        net = annual.get("NETPROFIT", 0) or 0
        summary["annual_revenue_yi"] = round(rev / 1e8, 2)
        summary["annual_net_profit_yi"] = round(net / 1e8, 2)
        summary["annual_period"] = annual.get("REPORT_DATE_NAME", "")
        # PE
        if net > 0 and kline:
            price = kline[-1]["close"]
            shares = float(info.get("总股本", 0))
            summary["pe_ttm"] = round(price * shares / net, 2)

    if q3:
        rev = q3.get("TOTAL_OPERATE_INCOME", 0) or 0
        net = q3.get("NETPROFIT", 0) or 0
        summary["q3_revenue_yi"] = round(rev / 1e8, 2)
        summary["q3_net_profit_yi"] = round(net / 1e8, 2)
        summary["q3_period"] = q3.get("REPORT_DATE_NAME", "")
        yoy = q3.get("NETPROFIT_YOY")
        if yoy:
            summary["q3_net_profit_yoy"] = round(float(yoy), 2)

    # 现金流
    cf_annual = _find_period(cf, "年报")
    if cf_annual:
        ocf = cf_annual.get("NETCASH_OPERATE", 0) or 0
        summary["annual_ocf_yi"] = round(ocf / 1e8, 2)
        net = annual.get("NETPROFIT", 0) or 0
        if net > 0:
            summary["ocf_to_profit"] = round(ocf / net, 2)

    # 资产负债
    bs_latest = bs[0] if bs else {}
    if bs_latest:
        summary["total_assets_yi"] = round((bs_latest.get("TOTAL_ASSETS", 0) or 0) / 1e8, 2)
        summary["equity_yi"] = round((bs_latest.get("TOTAL_PARENT_EQUITY", 0) or 0) / 1e8, 2)
        summary["accounts_receivable_yi"] = round((bs_latest.get("ACCOUNTS_RECE", 0) or 0) / 1e8, 2)
        summary["inventory_yi"] = round((bs_latest.get("INVENTORY", 0) or 0) / 1e8, 2)
        summary["short_loan_yi"] = round((bs_latest.get("SHORT_LOAN", 0) or 0) / 1e8, 2)
        summary["long_loan_yi"] = round((bs_latest.get("LONG_LOAN", 0) or 0) / 1e8, 2)

        # PB
        equity = bs_latest.get("TOTAL_PARENT_EQUITY", 0) or 0
        if equity > 0 and kline and info:
            mv = kline[-1]["close"] * float(info.get("总股本", 0))
            summary["pb"] = round(mv / equity, 2)

    # 分红
    divs = data.get("dividends", [])
    if divs:
        recent_div = divs[0]
        summary["recent_dividend"] = f"{recent_div.get('派息', 0)}元/10股 ({recent_div.get('公告日期', '')})"

    return summary
