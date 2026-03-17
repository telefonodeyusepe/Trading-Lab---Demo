from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from data_loader import AssetConfig, fit_regimes, get_asset_config


CONDITION_COLUMNS = [
    "cond_rsi",
    "cond_momentum",
    "cond_volatility",
    "cond_volume",
    "cond_adx",
    "cond_price_ma",
    "cond_macd",
    "cond_regime_confidence",
]


@dataclass
class BacktestResult:
    data: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    condition_labels: dict
    state_summary: pd.DataFrame | None
    config: AssetConfig
    bull_states: list[int]
    negative_states: list[int]
    crash_state: int | None
    momentum_threshold: float | None


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def _can_enter(
    config: AssetConfig,
    row: pd.Series,
    passed_count: int,
    timestamp: pd.Timestamp,
    cooldown_until: pd.Timestamp | None,
) -> bool:
    if cooldown_until is not None and timestamp < cooldown_until:
        return False
    if passed_count < config.confirmations_required:
        return False
    if not row["is_bullish_regime"]:
        return False
    if config.require_extreme_momentum and not row["cond_extreme_momentum"]:
        return False
    return True


def _should_exit(
    config: AssetConfig,
    row: pd.Series,
    passed_count: int,
    hold_bars: int,
) -> tuple[bool, str]:
    if config.exit_policy == "btc_regime_flip":
        if row["is_crash_regime"]:
            return True, "Crash regime"
        if hold_bars >= config.min_hold_candles and row["is_negative_return_state"]:
            return True, "Negative regime after minimum hold"
        return False, "Hold"

    if config.exit_policy == "negative_return_state":
        if row["is_negative_return_state"]:
            return True, "Negative-return regime"
        return False, "Hold"

    if config.exit_policy == "any_bearish_signal":
        bearish_signal = (
            row["is_negative_return_state"]
            or not row["is_bullish_regime"]
            or passed_count < config.confirmations_required
            or not row["cond_extreme_momentum"]
        )
        if bearish_signal:
            return True, "Bearish signal detected"
        return False, "Hold"

    return False, "Hold"


def run_buy_and_hold(symbol, start_capital=1500):
    df = yf.download(symbol, start="2024-03-01", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No {symbol} data returned from yfinance.")
    df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    start_price = float(df["Close"].iloc[0])
    end_price = float(df["Close"].iloc[-1])
    total_return = (end_price - start_price) / start_price
    ending_equity = start_capital * (1 + total_return)
    equity_curve = start_capital * (df["Close"] / start_price)

    return {
        "equity_curve": equity_curve,
        "total_return_pct": float(total_return * 100),
        "ending_equity": float(ending_equity),
        "start_price": start_price,
        "end_price": end_price,
        "current_price": end_price,
        "price_data": df,
    }


def run_backtest(symbol: str) -> BacktestResult:
    config = get_asset_config(symbol)
    if config.strategy_mode == "buy_hold":
        buy_hold = run_buy_and_hold(symbol, config.starting_capital)
        df = buy_hold["price_data"].rename(columns={"Close": "close"}).copy()
        df["returns"] = df["close"].pct_change()
        df["strategy_return"] = df["returns"].fillna(0)
        df["strategy_equity"] = buy_hold["equity_curve"]
        df["buy_hold_equity"] = buy_hold["equity_curve"]
        df["signal"] = "Buy & Hold"
        df["passed_conditions"] = 0

        today_change = (
            df["close"].iloc[-1] / df["close"].iloc[-2] - 1 if len(df) > 1 else 0.0
        )

        benchmark_return_pct = None
        alpha_pct = None
        benchmark_name = None
        if symbol != "SPY":
            benchmark = run_buy_and_hold("SPY", config.starting_capital)
            benchmark_return_pct = benchmark["total_return_pct"]
            alpha_pct = buy_hold["total_return_pct"] - benchmark_return_pct
            benchmark_name = "S&P 500 (SPY)"

        metrics = {
            "starting_capital": config.starting_capital,
            "start_price": buy_hold["start_price"],
            "end_price": buy_hold["end_price"],
            "ending_equity": buy_hold["ending_equity"],
            "total_return_pct": buy_hold["total_return_pct"],
            "buy_hold_return_pct": buy_hold["total_return_pct"],
            "benchmark_return_pct": benchmark_return_pct,
            "alpha_pct": alpha_pct,
            "benchmark_name": benchmark_name,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": float(_max_drawdown(df["strategy_equity"]) * 100),
            "trade_count": 0,
            "current_price": buy_hold["current_price"],
            "today_change_pct": float(today_change * 100),
        }

        return BacktestResult(
            data=df,
            trades=pd.DataFrame(),
            metrics=metrics,
            condition_labels={},
            state_summary=None,
            config=config,
            bull_states=[],
            negative_states=[],
            crash_state=None,
            momentum_threshold=None,
        )

    artifacts = fit_regimes(symbol)
    required_columns = CONDITION_COLUMNS + [
        "returns",
        "close",
        "state",
        "regime_name",
        "regime_confidence",
        "cond_extreme_momentum",
    ]
    df = artifacts.data.copy().dropna(subset=required_columns)

    position = 0
    entry_price = np.nan
    entry_time = None
    entry_equity = np.nan
    hold_bars = 0
    cooldown_until = None
    equity = config.starting_capital
    trades: list[dict] = []
    strategy_returns: list[float] = []
    signal_labels: list[str] = []
    passed_counts: list[int] = []

    for timestamp, row in df.iterrows():
        passed_count = int(row[CONDITION_COLUMNS].sum())
        passed_counts.append(passed_count)

        period_return = config.leverage * row["returns"] if position == 1 else 0.0
        strategy_returns.append(period_return)
        equity *= 1 + period_return

        if position == 1:
            hold_bars += 1
            should_exit, exit_reason = _should_exit(config, row, passed_count, hold_bars)
            if should_exit:
                trade_return = (equity / entry_equity) - 1
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": row["close"],
                        "bars_held": hold_bars,
                        "exit_reason": exit_reason,
                        "regime_at_exit": row["regime_name"],
                        "trade_return_pct": trade_return * 100,
                        "equity_after_trade": equity,
                    }
                )
                position = 0
                entry_price = np.nan
                entry_time = None
                entry_equity = np.nan
                hold_bars = 0
                cooldown_until = timestamp + pd.Timedelta(hours=config.cooldown_hours)
                signal_labels.append("Cash")
            else:
                signal_labels.append("Hold")
            continue

        if _can_enter(config, row, passed_count, timestamp, cooldown_until):
            position = 1
            entry_price = row["close"]
            entry_time = timestamp
            entry_equity = equity
            hold_bars = 0
            signal_labels.append("Long")
        else:
            signal_labels.append("Cash")

    if position == 1 and entry_time is not None:
        final_row = df.iloc[-1]
        trade_return = (equity / entry_equity) - 1
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": final_row.name,
                "entry_price": entry_price,
                "exit_price": final_row["close"],
                "bars_held": hold_bars,
                "exit_reason": "Open position at end of sample",
                "regime_at_exit": final_row["regime_name"],
                "trade_return_pct": trade_return * 100,
                "equity_after_trade": equity,
            }
        )

    df["strategy_return"] = strategy_returns
    df["strategy_equity"] = config.starting_capital * (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = config.starting_capital * (1 + df["returns"].fillna(0)).cumprod()
    df["signal"] = signal_labels
    df["passed_conditions"] = passed_counts

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["win"] = trades_df["trade_return_pct"] > 0

    total_return = df["strategy_equity"].iloc[-1] / config.starting_capital - 1
    buy_hold_return = df["buy_hold_equity"].iloc[-1] / config.starting_capital - 1
    win_rate = trades_df["win"].mean() if not trades_df.empty else 0.0

    metrics = {
        "starting_capital": config.starting_capital,
        "ending_equity": float(df["strategy_equity"].iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "buy_hold_return_pct": float(buy_hold_return * 100),
        "alpha_pct": float((total_return - buy_hold_return) * 100),
        "win_rate_pct": float(win_rate * 100),
        "max_drawdown_pct": float(_max_drawdown(df["strategy_equity"]) * 100),
        "trade_count": int(len(trades_df)),
    }

    condition_labels = {
        "cond_rsi": "RSI < 90",
        "cond_momentum": "Positive momentum",
        "cond_volatility": "Low volatility",
        "cond_volume": "Above average volume",
        "cond_adx": "ADX > 20",
        "cond_price_ma": "Price above MA",
        "cond_macd": "MACD signal positive",
        "cond_regime_confidence": "Regime confidence > 70%",
    }

    return BacktestResult(
        data=df,
        trades=trades_df,
        metrics=metrics,
        condition_labels=condition_labels,
        state_summary=artifacts.state_summary,
        config=config,
        bull_states=artifacts.bull_states,
        negative_states=artifacts.negative_states,
        crash_state=artifacts.crash_state,
        momentum_threshold=artifacts.momentum_threshold,
    )
