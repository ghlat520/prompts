#!/usr/bin/env python3
"""抓取鼎龙股份(300054)全套数据，输出JSON供prompt测试"""
import akshare as ak
import json
import datetime
import traceback

data = {
    "stock_code": "300054",
    "stock_name": "鼎龙股份",
    "market": "A",
    "industry": "半导体材料/电子化学品",
    "fetch_time": datetime.datetime.now().isoformat(),
}

errors = []

# 1. 实时行情
try:
    df = ak.stock_zh_a_spot_em()
    row = df[df["代码"] == "300054"].iloc[0]
    data["realtime"] = {
        "price": float(row["最新价"]),
        "change_pct": float(row["涨跌幅"]),
        "volume": float(row["成交量"]),
        "amount": float(row["成交额"]),
        "turnover": float(row.get("换手率", 0)),
        "pe": float(row.get("市盈率-动态", 0)),
        "pb": float(row.get("市净率", 0)),
        "total_mv": float(row.get("总市值", 0)),
        "circ_mv": float(row.get("流通市值", 0)),
        "high_52w": float(row.get("52周最高", 0)) if "52周最高" in row.index else None,
        "low_52w": float(row.get("52周最低", 0)) if "52周最低" in row.index else None,
    }
    print(f"[OK] realtime: price={data['realtime']['price']}, pe={data['realtime']['pe']}, mv={data['realtime']['total_mv']}")
except Exception as e:
    errors.append(f"realtime: {e}")
    traceback.print_exc()

# 2. 日K线（最近120日）
try:
    df = ak.stock_zh_a_hist(symbol="300054", period="daily", start_date="20250801", adjust="qfq")
    records = df.tail(120).to_dict(orient="records")
    # 转换列名
    data["kline_daily"] = [{
        "date": str(r.get("日期", "")),
        "open": r.get("开盘", 0),
        "close": r.get("收盘", 0),
        "high": r.get("最高", 0),
        "low": r.get("最低", 0),
        "volume": r.get("成交量", 0),
        "amount": r.get("成交额", 0),
        "turnover": r.get("换手率", 0),
    } for r in records]
    print(f"[OK] kline_daily: {len(data['kline_daily'])} records, latest={data['kline_daily'][-1]['date']}")
except Exception as e:
    errors.append(f"kline: {e}")
    traceback.print_exc()

# 3. 财务指标（最近几期）
try:
    df = ak.stock_financial_abstract_ths(symbol="300054", indicator="按报告期")
    data["financials"] = df.head(8).to_dict(orient="records")
    print(f"[OK] financials: {len(data['financials'])} periods")
except Exception as e:
    errors.append(f"financials: {e}")
    traceback.print_exc()

# 4. 利润表
try:
    df = ak.stock_profit_sheet_by_report_em(symbol="SZ300054")
    data["profit_sheet"] = df.head(8).to_dict(orient="records")
    print(f"[OK] profit_sheet: {len(data['profit_sheet'])} periods")
except Exception as e:
    errors.append(f"profit_sheet: {e}")
    traceback.print_exc()

# 5. 现金流量表
try:
    df = ak.stock_cash_flow_sheet_by_report_em(symbol="SZ300054")
    data["cashflow"] = df.head(8).to_dict(orient="records")
    print(f"[OK] cashflow: {len(data['cashflow'])} periods")
except Exception as e:
    errors.append(f"cashflow: {e}")
    traceback.print_exc()

# 6. 资产负债表
try:
    df = ak.stock_balance_sheet_by_report_em(symbol="SZ300054")
    data["balance_sheet"] = df.head(8).to_dict(orient="records")
    print(f"[OK] balance_sheet: {len(data['balance_sheet'])} periods")
except Exception as e:
    errors.append(f"balance_sheet: {e}")
    traceback.print_exc()

# 7. 个股信息（公司概况）
try:
    df = ak.stock_individual_info_em(symbol="300054")
    info = {}
    for _, r in df.iterrows():
        info[r.iloc[0]] = r.iloc[1]
    data["company_info"] = info
    print(f"[OK] company_info: {list(info.keys())}")
except Exception as e:
    errors.append(f"company_info: {e}")
    traceback.print_exc()

# 8. 主营构成
try:
    df = ak.stock_zygc_ym(symbol="300054")
    data["revenue_breakdown"] = df.head(20).to_dict(orient="records")
    print(f"[OK] revenue_breakdown: {len(data['revenue_breakdown'])} items")
except Exception as e:
    errors.append(f"revenue_breakdown: {e}")
    traceback.print_exc()

# 9. 分红历史
try:
    df = ak.stock_history_dividend_detail(symbol="300054", indicator="分红")
    data["dividends"] = df.head(10).to_dict(orient="records")
    print(f"[OK] dividends: {len(data['dividends'])} records")
except Exception as e:
    errors.append(f"dividends: {e}")
    traceback.print_exc()

# 10. 十大股东
try:
    df = ak.stock_main_stock_holder(stock="300054")
    data["top_holders"] = df.head(10).to_dict(orient="records")
    print(f"[OK] top_holders: {len(data['top_holders'])} records")
except Exception as e:
    errors.append(f"top_holders: {e}")
    traceback.print_exc()

data["_errors"] = errors

# 保存
out_path = "/Applications/soft/CodeSpace/prompts/test-data/300054_data.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, default=str)

print(f"\n=== Done === saved to {out_path}")
print(f"Errors: {len(errors)}")
for e in errors:
    print(f"  - {e}")
