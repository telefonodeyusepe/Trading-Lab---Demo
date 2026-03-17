from __future__ import annotations

import base64

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import CONDITION_COLUMNS, run_backtest
from data_loader import ASSET_CONFIGS, get_asset_config, get_assets_by_group


st.set_page_config(page_title="Dylan's Trading Lab", layout="wide")

APP_BG = "#0b0e11"
PANEL_BG = "#1e2026"
BORDER = "#2b2f36"
GOLD = "#f0b90b"
GOLD_BRIGHT = "#f0b90b"
SILVER = "#c0c8d8"
SILVER_SOFT = "#848e9c"
WHITE = "#eaecef"
POSITIVE = "#0ecb81"
NEGATIVE = "#f6465d"
LINE_COLORS = {
    "BTC-USD": "#f7931a",
    "ETH-USD": "#627eea",
    "XRP-USD": "#346aa9",
    "SPY": "#2962ff",
    "NVDA": "#76b900",
    "AAPL": "#e8eaf0",
}
REGIME_COLORS = {
    "Bull Run": POSITIVE,
    "Bear": NEGATIVE,
    "Crash": NEGATIVE,
    "Neutral": SILVER_SOFT,
}
ASSET_CARD_STYLES = {
    "BTC-USD": {"color": GOLD, "accent": PANEL_BG},
    "ETH-USD": {"color": GOLD, "accent": PANEL_BG},
    "XRP-USD": {"color": GOLD, "accent": PANEL_BG},
    "SPY": {"color": GOLD, "accent": PANEL_BG},
    "NVDA": {"color": GOLD, "accent": PANEL_BG},
    "AAPL": {"color": GOLD, "accent": PANEL_BG},
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_result(symbol: str):
    return run_backtest(symbol)


def _safe_load_result(symbol: str):
    try:
        return load_result(symbol), None
    except Exception as exc:
        return None, str(exc)


def _safe_load_many(symbols: list[str]) -> tuple[dict[str, object], dict[str, str]]:
    results: dict[str, object] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        result, error = _safe_load_result(symbol)
        if result is not None:
            results[symbol] = result
        elif error is not None:
            errors[symbol] = error
    return results, errors


def _apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');
        html, body, [class*="css"] {{
            font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
            font-weight: 300;
        }}
        [data-testid="stAppViewContainer"] {{
            background: {APP_BG};
            color: {WHITE};
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            display: none;
        }}
        #MainMenu, footer, header {{
            visibility: hidden;
        }}
        .block-container {{
            max-width: 1460px;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 1.4rem !important;
            padding-right: 1.4rem !important;
        }}
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {{
            gap: 0.7rem;
        }}
        .lux-page {{
            padding: 1.2rem 0 2.8rem;
        }}
        .lux-logo {{
            text-align: center;
            margin: 3.8rem 0 0.5rem;
            color: {WHITE};
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: 0;
        }}
        .lux-subtitle {{
            text-align: center;
            color: {SILVER_SOFT};
            font-size: 0.7rem;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            margin-bottom: 1.25rem;
        }}
        .lux-description {{
            max-width: 700px;
            margin: 0 auto 3.2rem;
            padding: 0.75rem 0 0.6rem;
            color: {SILVER_SOFT};
            font-size: 0.84rem;
            line-height: 1.8;
            text-align: center;
        }}
        .lux-footer {{
            text-align: center;
            color: {SILVER_SOFT};
            font-size: 0.64rem;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            margin: 4rem 0 2rem;
        }}
        .lux-section-title {{
            color: {WHITE};
            font-size: 1.05rem;
            letter-spacing: 0;
            text-transform: none;
            font-weight: 600;
            margin: 0.6rem 0 0.2rem;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid {BORDER};
        }}
        .lux-section-subtitle {{
            color: {SILVER_SOFT};
            font-size: 0.68rem;
            letter-spacing: 0.04rem;
            text-transform: uppercase;
            margin-bottom: 1.4rem;
        }}
        .lux-back button {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            color: {SILVER} !important;
            letter-spacing: 0.08rem !important;
            font-size: 0.68rem !important;
            text-transform: uppercase !important;
            padding: 0 !important;
            justify-content: flex-start !important;
        }}
        .lux-back button:hover {{
            color: {WHITE} !important;
        }}
        .lux-card-link {{
            text-decoration: none;
            display: block;
        }}
        .lux-card-link, .lux-card-link:hover, .lux-card-link:focus, .lux-card-link:visited {{
            text-decoration: none !important;
        }}
        .lux-card-link * {{
            text-decoration: none !important;
        }}
        .lux-card {{
            background: {PANEL_BG};
            border: 1px solid {BORDER};
            border-radius: 6px;
            min-height: 328px;
            padding: 2.35rem 1.5rem 2.25rem;
            box-shadow: none;
            outline: none;
            transition: transform 0.3s ease, background 0.3s ease;
        }}
        .lux-card:hover {{
            transform: translateY(-2px);
            background: {PANEL_BG};
            border-color: {GOLD};
        }}
        .lux-card-icon-wrap {{
            height: 128px;
            margin-bottom: 1.55rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .lux-card-title {{
            color: {WHITE};
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.18rem;
            text-transform: uppercase;
            text-align: center;
            margin-bottom: 0.95rem;
        }}
        .lux-card-subtitle {{
            color: {SILVER_SOFT};
            font-size: 0.68rem;
            line-height: 1.9;
            letter-spacing: 0.03rem;
            text-align: center;
        }}
        .lux-card-label {{
            color: {SILVER};
            font-size: 0.62rem;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            text-align: center;
            margin-top: 1rem;
        }}
        .lux-asset-name {{
            color: {WHITE};
            font-size: 1.55rem;
            font-weight: 500;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }}
        .lux-asset-ticker {{
            color: {SILVER};
            font-size: 0.72rem;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            margin-top: 0.35rem;
        }}
        .lux-badges {{
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin: 1rem 0 1.2rem;
        }}
        .lux-badge {{
            border: 1px solid {BORDER};
            color: {WHITE};
            border-radius: 999px;
            padding: 0.38rem 0.7rem;
            font-size: 0.63rem;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            background: {PANEL_BG};
        }}
        .lux-metric {{
            background: {PANEL_BG};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 0.95rem 1rem 1rem;
            min-height: 108px;
            box-shadow: none;
            outline: none;
        }}
        .lux-metric-label {{
            color: {SILVER_SOFT};
            font-size: 0.62rem;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }}
        .lux-metric-value {{
            color: {WHITE};
            font-size: 1.36rem;
            font-weight: 400;
            letter-spacing: 0.04rem;
            line-height: 1.2;
        }}
        .lux-table-wrap {{
            background: {PANEL_BG};
            border: 1px solid {BORDER};
            border-radius: 0;
            padding: 0;
        }}
        .lux-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .lux-table th {{
            color: {SILVER_SOFT};
            font-size: 0.58rem;
            font-weight: 400;
            letter-spacing: 0.1rem;
            text-transform: uppercase;
            padding: 0.72rem 0.8rem;
            border-bottom: 1px solid {BORDER};
            text-align: left;
            background: {BORDER};
        }}
        .lux-table td {{
            color: {WHITE};
            font-size: 0.75rem;
            padding: 0.78rem 0.8rem;
            border-bottom: 1px solid {BORDER};
            vertical-align: top;
            background: {PANEL_BG};
        }}
        .lux-table tr:nth-child(even) td {{
            background: #16181d;
        }}
        .lux-check {{
            color: {POSITIVE};
            font-size: 0.9rem;
        }}
        .lux-dash {{
            color: {NEGATIVE};
            font-size: 0.9rem;
        }}
        .lux-positive {{
            color: {POSITIVE};
        }}
        .lux-negative {{
            color: {NEGATIVE};
        }}
        .lux-neutral {{
            color: {WHITE};
        }}
        .lux-warning {{
            border: 1px solid {BORDER};
            border-radius: 0;
            background: {PANEL_BG};
            color: {SILVER};
            padding: 0.9rem 1rem;
            font-size: 0.75rem;
            letter-spacing: 0.05rem;
        }}
        .lux-divider {{
            text-align: center;
            margin: 1.6rem 0 1.1rem;
        }}
        .lux-divider-label {{
            color: {WHITE};
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0;
            text-transform: none;
            margin-bottom: 0.5rem;
        }}
        .lux-divider-line {{
            border-top: 1px solid {BORDER};
            width: 100%;
        }}
        .stMarkdown p {{
            color: {SILVER};
        }}
        a, a:hover, a:focus, a:visited {{
            text-decoration: none !important;
        }}
        .stAlert {{
            background: {PANEL_BG};
            border: 1px solid {BORDER};
            color: {SILVER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}%"


def _return_tone(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _metric_value_class(tone: str | None = None) -> str:
    if tone == "positive":
        return "lux-metric-value lux-positive"
    if tone == "negative":
        return "lux-metric-value lux-negative"
    return "lux-metric-value lux-neutral"


def _table_value_html(text: str, tone: str = "neutral", emphasize: bool = False) -> str:
    color = WHITE
    if tone == "positive":
        color = POSITIVE
    elif tone == "negative":
        color = NEGATIVE
    font_size = "1.04rem" if emphasize else "0.82rem"
    font_weight = "600" if emphasize or tone != "neutral" else "400"
    return f'<span style="color:{color}; font-size:{font_size}; font-weight:{font_weight};">{text}</span>'


def _signal_badge_style(signal: str) -> str:
    signal_upper = (signal or "").upper()
    if signal_upper == "LONG":
        color = POSITIVE
    elif signal_upper == "HOLD":
        color = GOLD
    else:
        color = SILVER_SOFT
    return f"border-color:{color}; color:{color};"


def _regime_badge_style(regime: str) -> str:
    regime_text = (regime or "").lower()
    if "bull" in regime_text:
        color = POSITIVE
    elif "bear" in regime_text or "crash" in regime_text:
        color = NEGATIVE
    else:
        color = SILVER_SOFT
    return f"border-color:{color}; color:{color};"


def _normalize_datetime_index(index: pd.Index) -> pd.Index:
    dt_index = pd.to_datetime(index)
    if isinstance(dt_index, pd.DatetimeIndex) and dt_index.tz is not None:
        return dt_index.tz_convert(None)
    return dt_index


def _get_query_param(key: str, default: str | None = None) -> str | None:
    value = st.query_params.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _set_route(page: str, asset: str | None = None) -> None:
    st.query_params.clear()
    st.query_params["page"] = page
    if asset is not None:
        st.query_params["asset"] = asset
    st.rerun()


def _show_welcome_toast() -> None:
    if "welcome_toast_shown" not in st.session_state:
        st.session_state["welcome_toast_shown"] = True
        st.toast("Wassup yall, made by Dylan. Paper trading since March 2024.")


def _svg_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def _landing_icon(icon_kind: str) -> str:
    if icon_kind == "crypto":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="30" fill="none" stroke="{GOLD}" stroke-width="1.7"/>
          <text x="60" y="69" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="34" font-weight="300" fill="{GOLD}">₿</text>
        </svg>
        """
    elif icon_kind == "stocks":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
          <path d="M30 80h60" fill="none" stroke="{GOLD}" stroke-width="1.7" stroke-linecap="round"/>
          <path d="M37 80V52M55 80V40M73 80V32M88 80V48" fill="none" stroke="{GOLD}" stroke-width="1.7" stroke-linecap="round"/>
          <path d="M35 56c12-14 18-7 27-18 7-7 12-4 26-16" fill="none" stroke="{GOLD}" stroke-width="1.7" stroke-linecap="round"/>
        </svg>
        """
    else:
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="30" fill="none" stroke="{GOLD}" stroke-width="1.7"/>
          <path d="M60 60L60 30A30 30 0 0 1 88 74Z" fill="none" stroke="{GOLD}" stroke-width="1.7"/>
          <path d="M60 60L88 74A30 30 0 0 1 38 83Z" fill="none" stroke="{GOLD}" stroke-width="1.7"/>
          <path d="M60 60L38 83A30 30 0 0 1 60 30Z" fill="none" stroke="{GOLD}" stroke-width="1.7"/>
        </svg>
        """
    return _svg_data_uri(svg.strip())


def _asset_icon(symbol: str) -> str:
    style = ASSET_CARD_STYLES[symbol]
    label = symbol.replace("-USD", "")
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="86" height="86" viewBox="0 0 86 86">
      <circle cx="43" cy="43" r="21" fill="none" stroke="{style['color']}" stroke-width="1.5"/>
      <text x="43" y="48" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="300" letter-spacing="1.8" fill="{style['color']}">{label}</text>
    </svg>
    """
    return _svg_data_uri(svg.strip())


def _route_card(
    title: str,
    subtitle: str,
    page: str,
    icon_kind: str,
    strategy_label: str = "",
    strategy_color: str = GOLD,
) -> str:
    icon_uri = _landing_icon(icon_kind)
    strategy_html = (
        f'<div class="lux-card-label" style="color:{strategy_color};">{strategy_label}</div>'
        if strategy_label
        else ""
    )
    return f"""
    <a class="lux-card-link" href="?page={page}" target="_self">
      <div class="lux-card">
        <div class="lux-card-icon-wrap"><img src="{icon_uri}" style="width:120px;height:120px;" /></div>
        <div class="lux-card-title">{title}</div>
        <div class="lux-card-subtitle">{subtitle}</div>
        {strategy_html}
      </div>
    </a>
    """


def _asset_card(symbol: str) -> str:
    config = get_asset_config(symbol)
    subtitle = (
        "Active Trading · March 2024 – Present"
        if config.strategy_mode == "active"
        else "Buy & Hold · March 2024 – Present"
    )

    return f"""
    <a class="lux-card-link" href="?page=asset&asset={symbol}" target="_self">
      <div class="lux-card">
        <div class="lux-card-icon-wrap"><img src="{_asset_icon(symbol)}" style="width:86px;height:86px;" /></div>
        <div class="lux-card-label" style="margin-top:0;margin-bottom:0.85rem;">{config.symbol}</div>
        <div class="lux-card-title">{config.display_name}</div>
        <div class="lux-card-subtitle" style="color:{POSITIVE if config.strategy_mode == 'active' else SILVER_SOFT};">{subtitle}</div>
      </div>
    </a>
    """


def _base_chart_layout(fig: go.Figure, title: str, yaxis_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": WHITE, "family": "Inter"}, "x": 0.01},
        xaxis_title="",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 32, "r": 18, "t": 54, "b": 28},
        font={"color": SILVER, "size": 11, "family": "Inter"},
        legend={"font": {"size": 10, "color": SILVER}, "orientation": "h", "y": -0.16},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont={"size": 10, "color": SILVER_SOFT},
        title_font={"size": 10, "color": SILVER_SOFT},
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont={"size": 10, "color": SILVER_SOFT},
        title_font={"size": 10, "color": SILVER_SOFT},
        zeroline=False,
    )
    return fig


def _price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    asset_color = LINE_COLORS.get(symbol, SILVER)
    if {"open", "high", "low", "close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candles",
                increasing={"line": {"color": POSITIVE, "width": 1}, "fillcolor": POSITIVE},
                decreasing={"line": {"color": NEGATIVE, "width": 1}, "fillcolor": NEGATIVE},
                whiskerwidth=0.3,
                opacity=0.42,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name=symbol,
            line={"color": asset_color, "width": 1.8},
        )
    )
    for regime_name, color in REGIME_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"].where(df["regime_name"] == regime_name),
                mode="lines",
                name=regime_name,
                line={"color": color, "width": 1.5, "dash": "dot"},
            )
        )
    return _base_chart_layout(fig, f"{symbol} Price With Regime Overlay", "Price")


def _stock_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    asset_color = LINE_COLORS.get(symbol, SILVER)
    if {"open", "high", "low", "close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candles",
                increasing={"line": {"color": POSITIVE, "width": 1}, "fillcolor": POSITIVE},
                decreasing={"line": {"color": NEGATIVE, "width": 1}, "fillcolor": NEGATIVE},
                whiskerwidth=0.3,
                opacity=0.42,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name=symbol,
            line={"color": asset_color, "width": 1.7},
        )
    )
    return _base_chart_layout(fig, f"{symbol} Price Since March 2024", "Price")


def _stock_equity_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["strategy_equity"],
            mode="lines",
            name=symbol,
            line={"color": LINE_COLORS.get(symbol, SILVER), "width": 1.8},
        )
    )
    return _base_chart_layout(fig, f"{symbol} Buy & Hold Equity Curve", "Portfolio Value")


def _equity_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["strategy_equity"],
            mode="lines",
            name=f"{symbol} Strategy",
            line={"color": LINE_COLORS.get(symbol, SILVER), "width": 1.8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["buy_hold_equity"],
            mode="lines",
            name="Buy & Hold",
            line={"color": SILVER_SOFT, "width": 1.2},
        )
    )
    return _base_chart_layout(fig, "Equity Curve vs Buy & Hold", "Portfolio Value")


def _equity_frame(results_map: dict[str, object]) -> pd.DataFrame:
    if not results_map:
        return pd.DataFrame()

    frames = []
    for symbol, result in results_map.items():
        frame = result.data[["strategy_equity", "buy_hold_equity"]].copy()
        frame.index = _normalize_datetime_index(frame.index)
        frame = frame.rename(
            columns={
                "strategy_equity": f"{symbol}_strategy_equity",
                "buy_hold_equity": f"{symbol}_buy_hold_equity",
            }
        )
        frames.append(frame)

    equity_df = pd.concat(frames, axis=1).sort_index().ffill().dropna()
    strategy_cols = [f"{symbol}_strategy_equity" for symbol in results_map]
    buy_hold_cols = [f"{symbol}_buy_hold_equity" for symbol in results_map]
    equity_df["combined_strategy_equity"] = equity_df[strategy_cols].sum(axis=1)
    equity_df["combined_buy_hold_equity"] = equity_df[buy_hold_cols].sum(axis=1)
    return equity_df


def _portfolio_summary(results_map: dict[str, object], benchmark_mode: str = "buy_hold") -> dict:
    if not results_map:
        return {
            "starting_capital": 0.0,
            "ending_equity": 0.0,
            "total_return_pct": None,
            "benchmark_return_pct": None,
            "alpha_pct": None,
            "best_asset": "N/A",
            "worst_asset": "N/A",
            "combined_win_rate_pct": None,
            "max_drawdown_pct": None,
            "trade_count": 0,
            "combined_trades": pd.DataFrame(),
            "equity_df": pd.DataFrame(),
        }

    starting_capital = sum(result.metrics.get("starting_capital", 0.0) for result in results_map.values())
    ending_equity = sum(result.metrics.get("ending_equity", 0.0) for result in results_map.values())
    total_return = ending_equity / starting_capital - 1 if starting_capital else 0.0

    if benchmark_mode == "spy":
        spy_result = results_map.get("SPY")
        benchmark_ending = (
            starting_capital * (1 + spy_result.metrics.get("buy_hold_return_pct", 0.0) / 100)
            if spy_result is not None
            else None
        )
    else:
        benchmark_ending = sum(
            result.data["buy_hold_equity"].iloc[-1]
            if "buy_hold_equity" in result.data.columns
            else result.metrics.get("ending_equity", 0.0)
            for result in results_map.values()
        )

    benchmark_return = (
        benchmark_ending / starting_capital - 1
        if benchmark_ending is not None and starting_capital
        else None
    )
    equity_df = _equity_frame(results_map)
    drawdown = (
        (equity_df["combined_strategy_equity"] / equity_df["combined_strategy_equity"].cummax() - 1).min()
        if not equity_df.empty
        else None
    )

    trades = []
    for symbol, result in results_map.items():
        if not result.trades.empty:
            trade_log = result.trades.copy()
            trade_log["asset"] = symbol
            trades.append(trade_log)
    combined_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    win_rate = combined_trades["win"].mean() if not combined_trades.empty else 0.0
    best_asset = max(results_map, key=lambda item: results_map[item].metrics.get("total_return_pct", float("-inf")))
    worst_asset = min(results_map, key=lambda item: results_map[item].metrics.get("total_return_pct", float("inf")))

    return {
        "starting_capital": starting_capital,
        "ending_equity": ending_equity,
        "total_return_pct": total_return * 100,
        "benchmark_return_pct": benchmark_return * 100 if benchmark_return is not None else None,
        "alpha_pct": (total_return - benchmark_return) * 100 if benchmark_return is not None else None,
        "best_asset": best_asset,
        "worst_asset": worst_asset,
        "combined_win_rate_pct": win_rate * 100 if not pd.isna(win_rate) else None,
        "max_drawdown_pct": drawdown * 100 if drawdown is not None else None,
        "trade_count": int(len(combined_trades)),
        "combined_trades": combined_trades,
        "equity_df": equity_df,
    }


def _group_equity_chart(results_map: dict[str, object], title: str, include_total: bool = True) -> go.Figure:
    equity_df = _equity_frame(results_map)
    fig = go.Figure()
    if not results_map or equity_df.empty:
        return _base_chart_layout(fig, title, "Portfolio Value")

    for symbol in results_map:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df[f"{symbol}_strategy_equity"],
                mode="lines",
                name=symbol,
                line={"width": 1.4, "color": LINE_COLORS[symbol]},
            )
        )
    if include_total:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df["combined_strategy_equity"],
                mode="lines",
                name="Combined Total",
                line={"width": 1.9, "color": GOLD_BRIGHT},
            )
        )
    return _base_chart_layout(fig, title, "Portfolio Value")


def _contribution_pie_chart(results_map: dict[str, object]) -> go.Figure:
    fig = go.Figure()
    if not results_map:
        return _base_chart_layout(fig, "Each Asset's Contribution to Total Return", "")
    labels = []
    values = []
    for symbol, result in results_map.items():
        pnl = result.metrics.get("ending_equity", 0.0) - result.metrics.get("starting_capital", 0.0)
        labels.append(symbol)
        values.append(abs(pnl) if abs(pnl) > 0 else 0.01)
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={
                "colors": [LINE_COLORS[label] for label in labels],
                "line": {"color": BORDER, "width": 1},
            },
            textfont={"color": SILVER, "size": 10},
        )
    )
    fig.update_layout(
        title={"text": "Each Asset's Contribution to Total Return", "font": {"size": 14, "color": WHITE}},
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        font={"color": SILVER, "size": 11, "family": "Inter"},
    )
    return fig


def _return_bar_chart(results_map: dict[str, object]) -> go.Figure:
    fig = go.Figure()
    if not results_map:
        return _base_chart_layout(fig, "Asset Return vs Buy & Hold", "Return (%)")
    symbols = list(results_map.keys())
    strategy_returns = [results_map[symbol].metrics.get("total_return_pct", 0.0) for symbol in symbols]
    buy_hold_returns = [results_map[symbol].metrics.get("buy_hold_return_pct", 0.0) for symbol in symbols]
    asset_colors = [LINE_COLORS[symbol] for symbol in symbols]
    fig.add_trace(go.Bar(x=symbols, y=strategy_returns, name="Strategy Return", marker_color=asset_colors))
    fig.add_trace(go.Bar(x=symbols, y=buy_hold_returns, name="Buy & Hold Return", marker_color=asset_colors, opacity=0.45))
    fig.update_layout(barmode="group")
    return _base_chart_layout(fig, "Asset Return vs Buy & Hold", "Return (%)")


def _render_metric_cards(cards: list[dict]) -> None:
    cols = st.columns(len(cards), gap="medium")
    for col, card in zip(cols, cards):
        with col:
            value_class = _metric_value_class(card.get("tone"))
            st.markdown(
                f"""
                <div class="lux-metric">
                  <div class="lux-metric-label">{card['label']}</div>
                  <div class="{value_class}">{card['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    header_html = "".join(f"<th>{label}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{row.get(key, '')}</td>" for key, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    table_html = f"""
    <div class="lux-table-wrap">
      <table class="lux-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _summary_rows(results, symbol: str, strategy_mode: str) -> list[dict]:
    starting_capital = results.metrics.get("starting_capital")
    ending_equity = results.metrics.get("ending_equity")
    total_return_pct = results.metrics.get("total_return_pct")
    alpha_pct = results.metrics.get("alpha_pct")
    max_drawdown_pct = results.metrics.get("max_drawdown_pct")

    return [
        {"label": "Starting Capital", "value": _table_value_html(_fmt_money(starting_capital), "neutral")},
        {
            "label": "Ending Equity",
            "value": _table_value_html(
                _fmt_money(ending_equity),
                _return_tone((ending_equity or 0) - (starting_capital or 0)),
                emphasize=True,
            ),
        },
        {
            "label": "Total Return",
            "value": _table_value_html(_fmt_pct(total_return_pct), _return_tone(total_return_pct)),
        },
        {
            "label": "Alpha vs Benchmark",
            "value": _table_value_html(
                _fmt_pct(alpha_pct) if symbol != "SPY" or strategy_mode == "active" else "N/A",
                _return_tone(alpha_pct) if symbol != "SPY" or strategy_mode == "active" else "neutral",
            ),
        },
        {
            "label": "Win Rate",
            "value": _table_value_html(
                _fmt_pct(results.metrics.get("win_rate_pct")) if strategy_mode == "active" else "Buy & Hold",
                "neutral",
            ),
        },
        {
            "label": "Max Drawdown",
            "value": _table_value_html(_fmt_pct(max_drawdown_pct), "negative"),
        },
        {
            "label": "Total Trades",
            "value": results.metrics.get("trade_count", len(results.trades)) if strategy_mode == "active" else "N/A",
        },
    ]


def _render_condition_checklist(results, latest: pd.Series) -> None:
    rows = []
    for column in CONDITION_COLUMNS:
        passed = bool(latest[column])
        rows.append(
            {
                "condition": results.condition_labels[column],
                "status": f'<span class="lux-check">✓</span>' if passed else '<span class="lux-dash">✗</span>',
            }
        )
    if results.config.require_extreme_momentum:
        rows.append(
            {
                "condition": "Extreme momentum (top 10% historically)",
                "status": '<span class="lux-check">✓</span>' if bool(latest["cond_extreme_momentum"]) else '<span class="lux-dash">✗</span>',
            }
        )
    _render_table(rows, [("condition", "Condition"), ("status", "Status")])


def _render_trade_log(df: pd.DataFrame) -> None:
    if df.empty:
        st.markdown('<div class="lux-warning">No trades were triggered for the sampled window.</div>', unsafe_allow_html=True)
        return
    trade_log = df.copy()
    if "entry_time" in trade_log.columns:
        trade_log["entry_time"] = pd.to_datetime(trade_log["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
    if "exit_time" in trade_log.columns:
        trade_log["exit_time"] = pd.to_datetime(trade_log["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
    if "trade_return_pct" in trade_log.columns:
        trade_log["trade_return_pct"] = trade_log["trade_return_pct"].apply(
            lambda value: (
                f'<span class="{"lux-positive" if value > 0 else "lux-negative" if value < 0 else "lux-neutral"}">{value:,.2f}%</span>'
                if pd.notna(value)
                else "N/A"
            )
        )
    rows = trade_log.to_dict("records")
    columns = [(col, col.replace("_", " ").title()) for col in trade_log.columns]
    _render_table(rows, columns)


def _render_divider(title: str) -> None:
    st.markdown(
        f"""
        <div class="lux-divider">
          <div class="lux-divider-label">{title}</div>
          <div class="lux-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="lux-section-title">{title}</div>
        <div class="lux-section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_landing_page() -> None:
    st.markdown('<div class="lux-page">', unsafe_allow_html=True)
    st.markdown('<div class="lux-logo">CRYPTO &amp; STOCKS MODEL — DEMO — DYLAN PIERCE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lux-subtitle">PRIVATE PORTFOLIO INTELLIGENCE · EST. 2024</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="lux-description">
          This is a paper trading simulation using Hidden Markov Models (HMM) to detect market regimes and generate trade signals across crypto assets. The model is trained on 2 years of hourly price data and uses Gaussian distributions to classify market conditions into distinct states — ranging from strong bull runs to crashes and choppy noise. For crypto (BTC, ETH, XRP), an active trading strategy is applied: the model only enters trades during confirmed bull regimes when a minimum number of technical conditions are met, and exits immediately when conditions deteriorate. For stocks (SPY, NVDA, AAPL), a simple buy and hold strategy is used instead, as research shows active trading rarely outperforms passive holding for large-cap equities over the long term. All results shown are simulated from March 2024 to present using $1,500 per asset ($9,000 total). No real money is involved.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns(3, gap="large")
    with left:
        st.markdown(
            _route_card("CRYPTO", "Active Trading · March 2024 – Present", "crypto", "crypto", "Active Trading Strategy", SILVER_SOFT),
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            _route_card("STOCKS", "Buy & Hold · March 2024 – Present", "stocks", "stocks", "Buy & Hold Strategy", SILVER_SOFT),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            _route_card(
                "OVERALL PORTFOLIO",
                "Crypto + Stocks Combined · March 2024 – Present",
                "portfolio",
                "portfolio",
            ),
            unsafe_allow_html=True,
        )
    st.markdown('<div class="lux-footer">FOR PRIVATE USE ONLY · DYLAN © 2026</div></div>', unsafe_allow_html=True)


def _render_hub(group: str) -> None:
    title = "Crypto" if group == "crypto" else "Stocks"
    subtitle = (
        "Active trading dashboards for digital assets"
        if group == "crypto"
        else "Buy and hold dashboards for equity holdings"
    )
    top_left, top_right = st.columns([1, 8])
    with top_left:
        st.markdown('<div class="lux-back">', unsafe_allow_html=True)
        if st.button("← RETURN", use_container_width=True, key=f"{group}_back"):
            _set_route("home")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        _render_page_header(title, subtitle)
    cols = st.columns(3, gap="large")
    for col, config in zip(cols, get_assets_by_group(group)):
        with col:
            st.markdown(_asset_card(config.symbol), unsafe_allow_html=True)


def _render_asset_hero(config, results, latest: pd.Series) -> None:
    regime = latest["regime_name"] if config.strategy_mode == "active" else "Buy & Hold"
    signal = latest["signal"] if config.strategy_mode == "active" else "Buy & Hold"
    st.markdown(
        f"""
        <div class="lux-asset-name">{config.display_name}</div>
        <div class="lux-asset-ticker">{config.symbol}</div>
        <div class="lux-badges">
          <div class="lux-badge" style="{_regime_badge_style(regime)}">{regime}</div>
          <div class="lux-badge" style="{_signal_badge_style(signal)}">{signal}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_asset_dashboard(symbol: str) -> None:
    config = get_asset_config(symbol)
    top_left, top_right = st.columns([1, 8])
    with top_left:
        st.markdown('<div class="lux-back">', unsafe_allow_html=True)
        if st.button("← RETURN", use_container_width=True, key=f"{symbol}_back"):
            _set_route(config.asset_group)
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        _render_page_header(config.asset_group, "Private asset dashboard")

    with st.spinner(
        f"Training {config.symbol} model and building the dashboard..."
        if config.strategy_mode == "active"
        else f"Loading {config.symbol} buy and hold dashboard..."
    ):
        results, load_error = _safe_load_result(symbol)

    if load_error is not None or results is None:
        st.markdown(
            f'<div class="lux-warning">Could not load {config.symbol}. {load_error or "Unknown error."}</div>',
            unsafe_allow_html=True,
        )
        return

    df = results.data.copy()
    df.index = _normalize_datetime_index(df.index)
    latest = df.iloc[-1]
    current_price = float(df["close"].iloc[-1]) if "close" in df.columns else None
    today_change_pct = (
        ((df["close"].iloc[-1] / df["close"].iloc[-2]) - 1) * 100
        if "close" in df.columns and len(df) > 1
        else None
    )

    _render_asset_hero(config, results, latest)

    if config.strategy_mode == "buy_hold":
        stock_metric_cards = [
            {
                "label": "Current Price",
                "value": _fmt_money(results.metrics.get("current_price", current_price)),
                "tone": "neutral",
            },
            {
                "label": "Change Today",
                "value": _fmt_pct(results.metrics.get("today_change_pct", today_change_pct)),
                "tone": _return_tone(results.metrics.get("today_change_pct", today_change_pct)),
            },
            {
                "label": "Total Return",
                "value": _fmt_pct(results.metrics.get("total_return_pct")),
                "tone": _return_tone(results.metrics.get("total_return_pct")),
            },
            {
                "label": "Ending Equity",
                "value": _fmt_money(results.metrics.get("ending_equity")),
                "tone": _return_tone((results.metrics.get("ending_equity") or 0) - (results.metrics.get("starting_capital") or 0)),
            },
            {
                "label": "Vs S&P 500",
                "value": _fmt_pct(results.metrics.get("alpha_pct")) if symbol != "SPY" else "N/A",
                "tone": _return_tone(results.metrics.get("alpha_pct")) if symbol != "SPY" else "neutral",
            },
        ]
        _render_metric_cards(stock_metric_cards)
        left, right = st.columns(2, gap="large")
        with left:
            st.plotly_chart(_stock_price_chart(df, symbol), use_container_width=True)
        with right:
            st.plotly_chart(_stock_equity_chart(df, symbol), use_container_width=True)
        _render_divider("SUMMARY")
        _render_table(_summary_rows(results, symbol, config.strategy_mode), [("label", "Metric"), ("value", "Value")])
        return

    _render_metric_cards(
        [
            {"label": "Current Price", "value": _fmt_money(current_price), "tone": "neutral"},
            {"label": "Change Today", "value": _fmt_pct(today_change_pct), "tone": _return_tone(today_change_pct)},
            {
                "label": "Confidence",
                "value": _fmt_pct(latest.get("regime_confidence", 0) * 100 if "regime_confidence" in latest else None),
                "tone": "neutral",
            },
            {
                "label": "Total Return",
                "value": _fmt_pct(results.metrics.get("total_return_pct")),
                "tone": _return_tone(results.metrics.get("total_return_pct")),
            },
            {
                "label": "Ending Equity",
                "value": _fmt_money(results.metrics.get("ending_equity")),
                "tone": _return_tone((results.metrics.get("ending_equity") or 0) - (results.metrics.get("starting_capital") or 0)),
            },
        ]
    )
    _render_divider("Condition Checklist")
    _render_condition_checklist(results, latest)
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(_price_chart(df, symbol), use_container_width=True)
    with right:
        st.plotly_chart(_equity_chart(df, symbol), use_container_width=True)
    _render_divider("SUMMARY")
    _render_table(_summary_rows(results, symbol, config.strategy_mode), [("label", "Metric"), ("value", "Value")])
    _render_divider("Trade Log")
    _render_trade_log(results.trades)


def _render_portfolio_page() -> None:
    top_left, top_right = st.columns([1, 8])
    with top_left:
        st.markdown('<div class="lux-back">', unsafe_allow_html=True)
        if st.button("← RETURN", use_container_width=True, key="portfolio_back"):
            _set_route("home")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        _render_page_header("Overall Portfolio", "$1,500 per asset · $9,000 total")

    with st.spinner("Training all models and building the overall portfolio view..."):
        all_results, load_errors = _safe_load_many(list(ASSET_CONFIGS))

    if load_errors:
        failed_assets = ", ".join(sorted(load_errors))
        st.markdown(
            f'<div class="lux-warning">Some assets could not be loaded and were skipped: {failed_assets}.</div>',
            unsafe_allow_html=True,
        )

    crypto_results = {symbol: result for symbol, result in all_results.items() if result.config.asset_group == "crypto"}
    stock_results = {symbol: result for symbol, result in all_results.items() if result.config.asset_group == "stocks"}
    crypto_summary = _portfolio_summary(crypto_results, benchmark_mode="buy_hold")
    stocks_summary = _portfolio_summary(stock_results, benchmark_mode="spy")
    combined_summary = _portfolio_summary(all_results, benchmark_mode="buy_hold")

    _render_divider("Crypto Section")
    _render_metric_cards(
        [
            {"label": "Starting Capital", "value": _fmt_money(crypto_summary["starting_capital"]), "tone": "neutral"},
            {
                "label": "Ending Equity",
                "value": _fmt_money(crypto_summary["ending_equity"]),
                "tone": _return_tone((crypto_summary["ending_equity"] or 0) - (crypto_summary["starting_capital"] or 0)),
            },
            {
                "label": "Total Return",
                "value": _fmt_pct(crypto_summary["total_return_pct"]),
                "tone": _return_tone(crypto_summary["total_return_pct"]),
            },
            {
                "label": "Alpha vs Buy & Hold",
                "value": _fmt_pct(crypto_summary["alpha_pct"]),
                "tone": _return_tone(crypto_summary["alpha_pct"]),
            },
        ]
    )
    st.plotly_chart(_group_equity_chart(crypto_results, "BTC, ETH, XRP Equity Curves", include_total=True), use_container_width=True)

    _render_divider("Stocks Section")
    _render_metric_cards(
        [
            {"label": "Starting Capital", "value": _fmt_money(stocks_summary["starting_capital"]), "tone": "neutral"},
            {
                "label": "Ending Equity",
                "value": _fmt_money(stocks_summary["ending_equity"]),
                "tone": _return_tone((stocks_summary["ending_equity"] or 0) - (stocks_summary["starting_capital"] or 0)),
            },
            {
                "label": "Total Return",
                "value": _fmt_pct(stocks_summary["total_return_pct"]),
                "tone": _return_tone(stocks_summary["total_return_pct"]),
            },
            {
                "label": "Alpha vs S&P 500",
                "value": _fmt_pct(stocks_summary["alpha_pct"]),
                "tone": _return_tone(stocks_summary["alpha_pct"]),
            },
        ]
    )
    st.plotly_chart(_group_equity_chart(stock_results, "SPY, NVDA, AAPL Equity Curves", include_total=True), use_container_width=True)

    _render_divider("Combined Section")
    _render_metric_cards(
        [
            {"label": "Starting Capital", "value": _fmt_money(combined_summary["starting_capital"]), "tone": "neutral"},
            {
                "label": "Ending Equity",
                "value": _fmt_money(combined_summary["ending_equity"]),
                "tone": _return_tone((combined_summary["ending_equity"] or 0) - (combined_summary["starting_capital"] or 0)),
            },
            {
                "label": "Total Return",
                "value": _fmt_pct(combined_summary["total_return_pct"]),
                "tone": _return_tone(combined_summary["total_return_pct"]),
            },
            {"label": "Best Asset", "value": combined_summary["best_asset"], "tone": "neutral"},
            {"label": "Worst Asset", "value": combined_summary["worst_asset"], "tone": "neutral"},
        ]
    )
    st.plotly_chart(_group_equity_chart(all_results, "All Assets Plus Total Portfolio Line", include_total=True), use_container_width=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(_contribution_pie_chart(all_results), use_container_width=True)
    with right:
        st.plotly_chart(_return_bar_chart(all_results), use_container_width=True)


def main() -> None:
    _apply_theme()
    _show_welcome_toast()
    page = _get_query_param("page", "home") or "home"
    asset = _get_query_param("asset")
    if page == "crypto":
        _render_hub("crypto")
    elif page == "stocks":
        _render_hub("stocks")
    elif page == "portfolio":
        _render_portfolio_page()
    elif page == "asset" and asset in ASSET_CONFIGS:
        _render_asset_dashboard(asset)
    else:
        _render_landing_page()


if __name__ == "__main__":
    main()
