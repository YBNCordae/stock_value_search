import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from io import BytesIO



# =============================
# 页面配置
# =============================
st.set_page_config(page_title="A股股价区间对比", layout="wide")
st.title("A股：今日收盘价 vs 区间统计（均值 / 最高 / 最低）")


# =============================
# 工具函数
# =============================

def to_ts_code(code: str) -> str:
    """把 6 位代码自动补全为 TuShare ts_code。"""
    code = (code or "").strip().upper()
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


def cn_date(d: dt.date) -> str:
    """把日期显示为中文：YYYY年M月D日（不强制补零）。"""
    return f"{d.year}年{d.month}月{d.day}日"


def _extreme_date_summary(dates: pd.Series) -> tuple[str, str]:
    """返回极值日期的简短展示与说明（处理多次出现同一极值的情况）。"""
    if dates is None or len(dates) == 0:
        return "-", "未找到日期"

    d_unique = pd.to_datetime(dates).dt.date.unique()
    d_unique = sorted(d_unique)
    if len(d_unique) == 1:
        d = cn_date(d_unique[0])
        return d, f"出现日期：{d}"

    first = cn_date(d_unique[0])
    last = cn_date(d_unique[-1])
    return f"{first} 等{len(d_unique)}天", f"共出现 {len(d_unique)} 天；最早：{first}；最晚：{last}"


@st.cache_data(ttl=900)  # 15 分钟缓存，减少调用 & 加速

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
        raise RuntimeError(f"TuShare接口错误 {j.get('code')}: {j.get('msg')}")

    data = j["data"]
    df = pd.DataFrame(data["items"], columns=data["fields"])
    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("trade_date")
    return df


# =============================
# UI：输入
# =============================

tz = ZoneInfo("Asia/Taipei")
today_local = dt.datetime.now(tz=tz).date()

colL, colR, colP = st.columns([2.2, 2.2, 2.6])

with colL:
    code = st.text_input("股票代码（例：600519 / 000001 / 600519.SH）", value="600519")
    mode = st.radio("区间选择", ["最近N个交易日", "自定义起止日期"], horizontal=True)

with colR:
    if mode == "最近N个交易日":
        window = int(st.number_input("N（交易日）", min_value=5, max_value=2000, value=60, step=1))
        end = today_local
        # 取足够长的范围再截取 tail(N)，避免长假导致不够 N 条
        start = end - dt.timedelta(days=900)
    else:
        end = st.date_input("结束日期", value=today_local)
        start = st.date_input("开始日期", value=end - dt.timedelta(days=180))
        window = None

with colP:
    st.markdown("**我的持仓（可选）**")
    buy_price = st.number_input("我的买入价（元）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    shares = st.number_input("持股数量（股）", min_value=0, value=0, step=100)

ts_code = to_ts_code(code)


# =============================
# Token：优先 Secrets，其次环境变量
# =============================

token = None
try:
    token = st.secrets.get("TUSHARE_TOKEN")  # Streamlit Cloud
except Exception:
    token = None

if not token:
    token = os.environ.get("TUSHARE_TOKEN")  # CloudBase / Docker / 本地

if not token:
    st.error("未检测到 TUSHARE_TOKEN。请在环境变量中配置 TUSHARE_TOKEN（CloudBase）或在 Secrets 中配置（Streamlit Cloud）。")
    st.stop()


# =============================
# 拉取 + 计算
# =============================

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
    today_close = float(closes[-1])  # 区间内最新一个交易日收盘价
    mean = float(closes.mean())
    high = float(closes.max())
    low = float(closes.min())
    asof_date = df_use["trade_date"].iloc[-1].date()
    asof = cn_date(asof_date)
    asof_key = asof_date.strftime("%Y%m%d")  # 用于文件名等（避免中文文件名兼容性问题）

    # 高低点出现的日期（可能不止一天）
    high_dates = df_use.loc[df_use["close"] == high, "trade_date"]
    low_dates = df_use.loc[df_use["close"] == low, "trade_date"]
    high_date_short, high_date_help = _extreme_date_summary(high_dates)
    low_date_short, low_date_help = _extreme_date_summary(low_dates)

    # 你要的：用高/低/今日收盘做计算
    dev_vs_mean = (today_close - mean) / mean if mean != 0 else 0.0
    rise_from_low = (today_close - low) / low if low != 0 else 0.0
    drawdown_from_high = (today_close - high) / high if high != 0 else 0.0  # 通常为负
    amplitude = (high - low) / low if low != 0 else 0.0  # 区间振幅（相对最低）

    pos_in_range = (today_close - low) / (high - low) if high != low else 0.0
    pos_pct = pos_in_range * 100.0

except Exception as e:
    st.error(f"数据获取或计算失败：{e}")
    st.stop()


# =============================
# 展示：核心指标（去掉重复展示）
# =============================

st.subheader("区间核心指标")

c1, c2, c3, c4, c5, c6 = st.columns(6)

# 今日收盘（若填写买入价，则展示相对买入的涨跌幅）
delta_today = None
if buy_price and buy_price > 0:
    delta_today = f"{(today_close - buy_price) / buy_price * 100:+.2f}%（相对买入）"

c1.metric("今日收盘价", f"{today_close:.2f}", delta_today, help=f"截至：{asof}")

# 均值（展示相对均值偏离）
c2.metric("区间均值", f"{mean:.2f}", f"{dev_vs_mean*100:+.2f}%（相对均值）")

# 最高 / 最低（在 help 里显示出现时间）
c3.metric("区间最高价", f"{high:.2f}", help=f"最高价出现：{high_date_short}\n\n{high_date_help}")
c4.metric("区间最低价", f"{low:.2f}", help=f"最低价出现：{low_date_short}\n\n{low_date_help}")

# 今日相对低点涨幅 / 相对高点回撤
c5.metric("今日相对最低涨幅", f"{rise_from_low*100:+.2f}%")
c6.metric("今日相对最高回撤", f"{drawdown_from_high*100:+.2f}%")

# 区间位置 + 振幅
p1, p2, p3 = st.columns([2, 2, 3])
with p1:
    st.metric("区间位置", f"{pos_pct:.1f}%", help="0% = 靠近区间最低价；100% = 靠近区间最高价")
with p2:
    st.metric("区间振幅", f"{amplitude*100:.2f}%", help="(最高 - 最低) / 最低")
with p3:
    st.progress(max(0.0, min(1.0, pos_in_range)))
    st.caption(f"位置进度：{pos_pct:.1f}%（按区间最低~最高线性归一化）")

st.write(f"区间最高价出现：{high_date_short} ｜ 区间最低价出现：{low_date_short}")

st.caption("说明：本页“今日”默认使用**区间内最新交易日收盘价**（不是盘中实时价）。")


# =============================
# 可选：我的持仓表现
# =============================

if buy_price and buy_price > 0:
    st.subheader("我的持仓表现（按今日收盘价计算）")

    diff = today_close - buy_price
    ret = diff / buy_price

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("买入价", f"{buy_price:.2f}")
    b2.metric("每股盈亏", f"{diff:+.2f}")
    b3.metric("收益率", f"{ret*100:+.2f}%")

    if shares and shares > 0:
        b4.metric("浮动盈亏（元）", f"{diff * shares:+.2f}")
    else:
        b4.metric("浮动盈亏（元）", "-", help="填入持股数量后自动计算")

    st.write(
        f"距离区间最高还差：{(high - today_close):.2f} 元（{((high - today_close) / today_close * 100) if today_close else 0.0:+.2f}%）"
        f" ｜ 距离区间最低高出：{(today_close - low):.2f} 元（{rise_from_low*100:+.2f}%）"
    )


# =============================
# 图表：收盘价 + 高低点标注（中文）
# =============================

st.subheader("区间走势（收盘价）")

x = df_use["trade_date"]
y = df_use["close"]

# 高低点用于画点：若多次出现，取第一次出现的日期
x_high = pd.to_datetime(high_dates.iloc[0]) if len(high_dates) > 0 else x.iloc[int(y.values.argmax())]
x_low = pd.to_datetime(low_dates.iloc[0]) if len(low_dates) > 0 else x.iloc[int(y.values.argmin())]

fig = go.Figure()

# 收盘价折线
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="收盘价",
        hovertemplate="日期=%{x|%Y年%m月%d日}<br>收盘=%{y:.2f}<extra></extra>",
    )
)

# 最新收盘点
fig.add_trace(
    go.Scatter(
        x=[x.iloc[-1]],
        y=[today_close],
        mode="markers",
        name="最新收盘",
        marker=dict(size=10),
        hovertemplate="最新收盘<br>日期=%{x|%Y年%m月%d日}<br>收盘=%{y:.2f}<extra></extra>",
    )
)

# 最高点 / 最低点
fig.add_trace(
    go.Scatter(
        x=[x_high],
        y=[high],
        mode="markers+text",
        name="区间最高",
        text=[f"最高 {high:.2f}"],
        textposition="top center",
        marker=dict(size=10),
        hovertemplate="区间最高<br>日期=%{x|%Y年%m月%d日}<br>收盘=%{y:.2f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=[x_low],
        y=[low],
        mode="markers+text",
        name="区间最低",
        text=[f"最低 {low:.2f}"],
        textposition="bottom center",
        marker=dict(size=10),
        hovertemplate="区间最低<br>日期=%{x|%Y年%m月%d日}<br>收盘=%{y:.2f}<extra></extra>",
    )
)

# 均值/最高/最低水平线（shape）
xmin, xmax = x.iloc[0], x.iloc[-1]
for val, label in [(mean, "均值"), (high, "最高"), (low, "最低")]:
    fig.add_shape(
        type="line",
        x0=xmin,
        x1=xmax,
        y0=val,
        y1=val,
        xref="x",
        yref="y",
        line=dict(width=1, dash="dot"),
    )
    fig.add_annotation(
        x=xmax,
        y=val,
        xref="x",
        yref="y",
        text=f"{label}：{val:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=12),
    )

# 区间范围带
fig.add_shape(
    type="rect",
    x0=xmin,
    x1=xmax,
    y0=low,
    y1=high,
    xref="x",
    yref="y",
    fillcolor="rgba(0,0,0,0.05)",
    line=dict(width=0),
    layer="below",
)

fig.update_layout(
    title=f"{ts_code}｜区间 {cn_date(df_use['trade_date'].iloc[0].date())} ~ {asof}｜共 {len(df_use)} 个交易日",
    xaxis_title="日期",
    xaxis=dict(tickformat="%Y年%m月%d日"),
    yaxis_title="收盘价",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)


# =============================
# 原始数据
# =============================

with st.expander("查看原始数据（日期 / 收盘价）"):
    df_show = df_use.rename(columns={"trade_date": "日期", "close": "收盘价"}).copy()
    df_show["日期"] = pd.to_datetime(df_show["日期"]).dt.strftime("%Y年%m月%d日")
    st.dataframe(df_show, use_container_width=True)

    csv = df_show.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载 CSV",
        data=csv,
        file_name=f"{ts_code}_{asof_key}_last{len(df_use)}.csv",
        mime="text/csv",
    )
    # =============================
    # 导出 Excel（当前查询信息）
    # =============================

    summary_rows = [
        ["股票代码", ts_code],
        ["区间选择", mode],
        ["区间开始", cn_date(df_use["trade_date"].iloc[0].date())],
        ["区间结束（最新交易日）", asof],
        ["交易日数量", len(df_use)],
        ["今日收盘价", today_close],
        ["区间均值", mean],
        ["区间最高价", high],
        ["最高价出现", high_date_short],
        ["区间最低价", low],
        ["最低价出现", low_date_short],
        ["今日相对最低涨幅(%)", rise_from_low * 100],
        ["今日相对最高回撤(%)", drawdown_from_high * 100],
        ["区间振幅(%)", amplitude * 100],
        ["区间位置(%)", pos_pct],
    ]

    if buy_price and buy_price > 0:
        diff = today_close - buy_price
        ret = diff / buy_price
        summary_rows += [
            ["我的买入价", buy_price],
            ["每股盈亏", diff],
            ["收益率(%)", ret * 100],
        ]
        if shares and shares > 0:
            summary_rows.append(["浮动盈亏(元)", diff * shares])

    df_summary = pd.DataFrame(summary_rows, columns=["指标", "数值"])

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="查询概览")

        # 区间数据：沿用你上面 df_show（日期已是中文字符串）
        df_show.to_excel(writer, index=False, sheet_name="区间数据")

    st.download_button(
        "导出 Excel（查询概览+区间数据）",
        data=excel_buffer.getvalue(),
        file_name=f"{ts_code}_{asof_key}_查询信息.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
