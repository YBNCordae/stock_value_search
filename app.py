import datetime as dt
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


st.set_page_config(page_title="A股股价对比", layout="wide")
st.title("A股：最新价 vs 历史区间均值 / 最高 / 最低")

def to_ts_code(code: str) -> str:
    code = code.strip().upper()
    if code.endswith((".SH", ".SZ", ".BJ")):
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
    return code  # 兜底：原样返回

def yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

@st.cache_data(ttl=900)  # 15分钟缓存，省调用 & 更快
def tushare_daily(ts_code: str, start_date: str, end_date: str, token: str) -> pd.DataFrame:
    url = "http://api.tushare.pro"
    payload = {
        "api_name": "daily",
        "token": token,
        "params": {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
        "fields": "trade_date,close",
    }
    r = requests.post(url, json=payload, timeout=12)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"TuShare error {j.get('code')}: {j.get('msg')}")
    data = j["data"]
    df = pd.DataFrame(data["items"], columns=data["fields"])
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("trade_date")
    return df

# -------- UI 输入 --------
colL, colR = st.columns([2, 3])
with colL:
    code = st.text_input("股票代码（例：600519 / 000001 / 600519.SH）", value="600519")
    mode = st.radio("区间选择", ["最近N个交易日", "自定义起止日期"], horizontal=True)

with colR:
    if mode == "最近N个交易日":
        window = st.number_input(
            "N（交易日）",
            min_value=5,
            max_value=2000,
            value=60,
            step=1
        )
        window = int(window)

        # 取足够长的日线，再 tail(N)
        end = dt.date.today()
        start = end - dt.timedelta(days=900)
    else:
        end = st.date_input("结束日期", value=dt.date.today())
        start = st.date_input("开始日期", value=end - dt.timedelta(days=180))
        window = None

ts_code = to_ts_code(code)

# -------- 读取 token（Secrets）--------
if "TUSHARE_TOKEN" not in st.secrets:
    st.error("未检测到 TUSHARE_TOKEN。请在 Streamlit Cloud 的 Secrets 里配置它（见下方第4步）。")
    st.stop()
token = st.secrets["TUSHARE_TOKEN"]

# -------- 拉取 + 计算 --------
try:
    df = tushare_daily(ts_code, yyyymmdd(start), yyyymmdd(end), token)
    if df.empty:
        st.warning("没有取到数据：请检查代码是否正确，或该区间是否有交易数据。")
        st.stop()

    if mode == "最近N个交易日":
        if len(df) < window:
            st.warning(f"数据不足：仅取到 {len(df)} 条，少于 N={window}。")
            st.stop()
        df_use = df.tail(window).copy()
    else:
        df_use = df.copy()

    closes = df_use["close"].to_numpy()
    today_close = float(closes[-1])  # 注意：这是“最新一个交易日收盘价”
    mean = float(closes.mean())
    high = float(closes.max())
    low = float(closes.min())
    dev_vs_mean = (today_close - mean) / mean if mean != 0 else 0.0
    pos_in_range = (today_close - low) / (high - low) if high != low else 0.0
    asof = df_use["trade_date"].iloc[-1].date().isoformat()

except Exception as e:
    st.error(f"数据获取或计算失败：{e}")
    st.stop()

# -------- 展示：数字卡片 --------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新交易日收盘", f"{today_close:.2f}", help=f"asof: {asof}")
c2.metric("区间均值", f"{mean:.2f}", f"{dev_vs_mean*100:+.2f}% vs 均值")
c3.metric("区间最高", f"{high:.2f}")
c4.metric("区间最低", f"{low:.2f}")
c5.metric("区间位置(0~1)", f"{pos_in_range:.2f}")

st.caption("说明：这里的“今日价”默认采用**最新交易日收盘价**（盘中实时价需要额外接实时行情源）。")

# ===== 更直观的“区间位置”展示 =====
st.subheader("区间位置可视化")

# 1) Low / Today / High 三个数字卡片
a, b, c = st.columns(3)
a.metric("区间最低 (Low)", f"{low:.2f}")
b.metric(f"最新价 (Today, asof {asof})", f"{today_close:.2f}")
c.metric("区间最高 (High)", f"{high:.2f}")

# 2) 位置进度条（0~100%）
pos_pct = int(round(pos_in_range * 100))
st.progress(pos_in_range)  # 0~1
st.caption(f"位置：{pos_pct}%（0% = 贴近区间最低，100% = 贴近区间最高）")

# 3) 可选：把“距离两端”也写清楚（更有感觉）
st.write(
    f"距Low：{(today_close - low):.2f} ｜ 距High：{(high - today_close):.2f} ｜ 区间宽度：{(high - low):.2f}"
)


# -------- 展示：Plotly 更美观的交互图 --------
x = df_use["trade_date"]
y = df_use["close"]

fig = go.Figure()

# 价格折线
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode="lines",
    name="Close",
    hovertemplate="Date=%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"
))

# 最新点（更醒目）
fig.add_trace(go.Scatter(
    x=[x.iloc[-1]], y=[today_close],
    mode="markers",
    name="Latest",
    marker=dict(size=10),
    hovertemplate="Latest<br>Date=%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"
))

# 三条水平线：mean/high/low（用 shape 画，干净不抢 legend）
xmin, xmax = x.iloc[0], x.iloc[-1]
for val, label in [(mean, "Mean"), (high, "High"), (low, "Low")]:
    fig.add_shape(
        type="line",
        x0=xmin, x1=xmax,
        y0=val, y1=val,
        xref="x", yref="y",
        line=dict(width=1, dash="dot"),
    )
    # 右侧标注文字
    fig.add_annotation(
        x=xmax, y=val,
        xref="x", yref="y",
        text=f"{label}: {val:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=12)
    )

# 可选：把 low~high 区间做成淡淡的“范围带”，更直观（喜欢就留着）
fig.add_shape(
    type="rect",
    x0=xmin, x1=xmax,
    y0=low, y1=high,
    xref="x", yref="y",
    fillcolor="rgba(0,0,0,0.05)",
    line=dict(width=0),
    layer="below"
)

fig.update_layout(
    title=f"{ts_code} | {len(df_use)} bars | asof {asof}",
    xaxis_title="Date",
    yaxis_title="Close",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)


with st.expander("查看原始数据（含单股价格）"):
    df_show = df_use.copy()
    df_show = df_show.rename(columns={"trade_date": "日期", "close": "单股价格"})
    st.dataframe(df_show, use_container_width=True)

    csv = df_show.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载 CSV",
        data=csv,
        file_name=f"{ts_code}_{asof}_last{len(df_use)}.csv",
        mime="text/csv"
    )

