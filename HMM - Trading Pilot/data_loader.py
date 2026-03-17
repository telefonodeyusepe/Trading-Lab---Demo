from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["returns", "range", "volume_change"]
REGIME_NAMES = {
    "bull": "Bull Run",
    "bear": "Bear",
    "crash": "Crash",
    "neutral": "Neutral",
}


@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    display_name: str
    asset_group: str
    strategy_mode: str
    interval: str
    period_days: int
    hmm_states: int
    leverage: float
    confirmations_required: int
    cooldown_hours: int
    starting_capital: float
    bull_state_count: int
    min_hold_candles: int
    exit_policy: str
    momentum_lag: int
    volatility_window: int
    volatility_compare_window: int
    average_volume_window: int
    moving_average_window: int
    require_extreme_momentum: bool = False
    momentum_quantile: float | None = None


BUY_HOLD_START_DATE = "2024-03-01"


ASSET_CONFIGS: dict[str, AssetConfig] = {
    "BTC-USD": AssetConfig(
        symbol="BTC-USD",
        display_name="Bitcoin",
        asset_group="crypto",
        strategy_mode="active",
        interval="1h",
        period_days=730,
        hmm_states=7,
        leverage=2.5,
        confirmations_required=7,
        cooldown_hours=48,
        starting_capital=1_500.0,
        bull_state_count=1,
        min_hold_candles=3,
        exit_policy="btc_regime_flip",
        momentum_lag=3,
        volatility_window=24,
        volatility_compare_window=24 * 7,
        average_volume_window=24,
        moving_average_window=24,
    ),
    "ETH-USD": AssetConfig(
        symbol="ETH-USD",
        display_name="Ethereum",
        asset_group="crypto",
        strategy_mode="active",
        interval="1h",
        period_days=730,
        hmm_states=7,
        leverage=1.5,
        confirmations_required=5,
        cooldown_hours=24,
        starting_capital=1_500.0,
        bull_state_count=2,
        min_hold_candles=0,
        exit_policy="negative_return_state",
        momentum_lag=3,
        volatility_window=24,
        volatility_compare_window=24 * 7,
        average_volume_window=24,
        moving_average_window=24,
    ),
    "XRP-USD": AssetConfig(
        symbol="XRP-USD",
        display_name="XRP",
        asset_group="crypto",
        strategy_mode="active",
        interval="1h",
        period_days=730,
        hmm_states=9,
        leverage=1.0,
        confirmations_required=4,
        cooldown_hours=12,
        starting_capital=1_500.0,
        bull_state_count=1,
        min_hold_candles=0,
        exit_policy="any_bearish_signal",
        momentum_lag=3,
        volatility_window=24,
        volatility_compare_window=24 * 7,
        average_volume_window=24,
        moving_average_window=24,
        require_extreme_momentum=True,
        momentum_quantile=0.90,
    ),
    "SPY": AssetConfig(
        symbol="SPY",
        display_name="SPDR S&P 500 ETF",
        asset_group="stocks",
        strategy_mode="buy_hold",
        interval="1d",
        period_days=730,
        hmm_states=5,
        leverage=1.0,
        confirmations_required=6,
        cooldown_hours=72,
        starting_capital=1_500.0,
        bull_state_count=1,
        min_hold_candles=0,
        exit_policy="negative_return_state",
        momentum_lag=5,
        volatility_window=20,
        volatility_compare_window=60,
        average_volume_window=20,
        moving_average_window=20,
    ),
    "NVDA": AssetConfig(
        symbol="NVDA",
        display_name="NVIDIA",
        asset_group="stocks",
        strategy_mode="buy_hold",
        interval="1d",
        period_days=730,
        hmm_states=6,
        leverage=1.0,
        confirmations_required=5,
        cooldown_hours=48,
        starting_capital=1_500.0,
        bull_state_count=1,
        min_hold_candles=0,
        exit_policy="negative_return_state",
        momentum_lag=5,
        volatility_window=20,
        volatility_compare_window=60,
        average_volume_window=20,
        moving_average_window=20,
    ),
    "AAPL": AssetConfig(
        symbol="AAPL",
        display_name="Apple",
        asset_group="stocks",
        strategy_mode="buy_hold",
        interval="1d",
        period_days=730,
        hmm_states=5,
        leverage=1.0,
        confirmations_required=6,
        cooldown_hours=72,
        starting_capital=1_500.0,
        bull_state_count=1,
        min_hold_candles=0,
        exit_policy="negative_return_state",
        momentum_lag=5,
        volatility_window=20,
        volatility_compare_window=60,
        average_volume_window=20,
        moving_average_window=20,
    ),
}


@dataclass
class RegimeArtifacts:
    data: pd.DataFrame
    model: GaussianHMM
    scaler: StandardScaler
    config: AssetConfig
    bull_states: list[int]
    negative_states: list[int]
    crash_state: int
    momentum_threshold: float | None
    state_summary: pd.DataFrame


__all__ = [
    "ASSET_CONFIGS",
    "BUY_HOLD_START_DATE",
    "FEATURE_COLUMNS",
    "REGIME_NAMES",
    "AssetConfig",
    "RegimeArtifacts",
    "fit_regimes",
    "get_asset_config",
    "get_assets_by_group",
    "load_market_data",
    "load_buy_hold_data",
]


def get_asset_config(symbol: str) -> AssetConfig:
    if symbol not in ASSET_CONFIGS:
        supported = ", ".join(ASSET_CONFIGS)
        raise ValueError(f"Unsupported asset '{symbol}'. Expected one of: {supported}.")
    return ASSET_CONFIGS[symbol]


def get_assets_by_group(group: str) -> list[AssetConfig]:
    return [config for config in ASSET_CONFIGS.values() if config.asset_group == group]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr_components = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    )

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace(
        [np.inf, -np.inf], np.nan
    )
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _engineer_indicators(df: pd.DataFrame, config: AssetConfig) -> pd.DataFrame:
    close = df["close"]
    volume = df["volume"]

    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    df["returns"] = close.pct_change()
    df["range"] = (df["high"] - df["low"]) / close.replace(0, np.nan)
    df["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)

    df["rsi"] = _rsi(close)
    df["momentum"] = close.pct_change(config.momentum_lag)
    df["volatility"] = df["returns"].rolling(config.volatility_window).std()
    df["avg_volume"] = volume.rolling(config.average_volume_window).mean()
    df["adx"] = _adx(df["high"], df["low"], close)
    df["ma"] = close.rolling(config.moving_average_window).mean()
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal

    df["cond_rsi"] = df["rsi"] < 90
    df["cond_momentum"] = df["momentum"] > 0
    df["cond_volatility"] = (
        df["volatility"] < df["volatility"].rolling(config.volatility_compare_window).median()
    )
    df["cond_volume"] = volume > df["avg_volume"]
    df["cond_adx"] = df["adx"] > 20
    df["cond_price_ma"] = close > df["ma"]
    df["cond_macd"] = df["macd_line"] > df["macd_signal"]
    return df


def _normalize_datetime_index(index: pd.Index) -> pd.Index:
    dt_index = pd.to_datetime(index)
    if isinstance(dt_index, pd.DatetimeIndex) and dt_index.tz is not None:
        return dt_index.tz_convert(None)
    return dt_index


def _download_with_retry(symbol: str, download_kwargs: dict, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(**download_kwargs)
            if raw.empty:
                raise ValueError(f"No {symbol} data returned from yfinance.")
            return raw
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2)

    message = f"Unable to load {symbol} market data after {retries} attempts."
    if last_error is not None:
        raise ValueError(f"{message} Last error: {last_error}") from last_error
    raise ValueError(message)


def load_market_data(symbol: str) -> pd.DataFrame:
    config = get_asset_config(symbol)
    download_kwargs = {
        "tickers": symbol,
        "interval": config.interval,
        "auto_adjust": False,
        "progress": False,
    }
    if config.strategy_mode == "buy_hold":
        download_kwargs["start"] = BUY_HOLD_START_DATE
    else:
        download_kwargs["period"] = f"{config.period_days}d"

    raw = _download_with_retry(symbol, download_kwargs)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" in raw.columns:
        raw["close"] = raw["adj_close"].fillna(raw["close"])

    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index = _normalize_datetime_index(df.index)
    df = df.sort_index()
    df["symbol"] = symbol
    return _engineer_indicators(df, config)


def load_buy_hold_data(symbol: str) -> pd.DataFrame:
    config = get_asset_config(symbol)
    raw = _download_with_retry(
        symbol,
        {
            "tickers": symbol,
            "start": BUY_HOLD_START_DATE,
            "interval": config.interval,
            "auto_adjust": False,
            "progress": False,
        },
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" in raw.columns:
        raw["close"] = raw["adj_close"].fillna(raw["close"])

    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index = _normalize_datetime_index(df.index)
    df = df.sort_index()
    df["symbol"] = symbol
    df["returns"] = df["close"].pct_change()
    return df


def fit_regimes(symbol: str) -> RegimeArtifacts:
    config = get_asset_config(symbol)
    df = load_market_data(symbol)

    model_frame = df.dropna(subset=FEATURE_COLUMNS).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(model_frame[FEATURE_COLUMNS])

    model = GaussianHMM(
        n_components=config.hmm_states,
        covariance_type="full",
        n_iter=400,
        random_state=42,
    )
    model.fit(X)

    states = model.predict(X)
    state_probs = model.predict_proba(X)

    model_frame["state"] = states
    model_frame["regime_confidence"] = state_probs.max(axis=1)

    state_summary = (
        model_frame.groupby("state")
        .agg(
            mean_return=("returns", "mean"),
            mean_range=("range", "mean"),
            mean_volume_change=("volume_change", "mean"),
            observations=("state", "size"),
        )
        .sort_values("mean_return", ascending=False)
    )

    positive_states = [
        int(state) for state, mean_return in state_summary["mean_return"].items() if mean_return > 0
    ]
    bull_states = positive_states[: config.bull_state_count] if positive_states else [int(state_summary.index[0])]

    negative_states = [
        int(state) for state, mean_return in state_summary["mean_return"].items() if mean_return < 0
    ]
    crash_state = int(state_summary["mean_return"].idxmin())

    state_summary = state_summary.sort_index()
    state_summary["regime_bucket"] = REGIME_NAMES["neutral"]
    state_summary.loc[state_summary.index.isin(bull_states), "regime_bucket"] = REGIME_NAMES["bull"]
    state_summary.loc[state_summary.index.isin(negative_states), "regime_bucket"] = REGIME_NAMES["bear"]
    state_summary.loc[crash_state, "regime_bucket"] = REGIME_NAMES["crash"]

    model_frame["regime_name"] = REGIME_NAMES["neutral"]
    model_frame.loc[model_frame["state"].isin(bull_states), "regime_name"] = REGIME_NAMES["bull"]
    model_frame.loc[model_frame["state"].isin(negative_states), "regime_name"] = REGIME_NAMES["bear"]
    model_frame.loc[model_frame["state"] == crash_state, "regime_name"] = REGIME_NAMES["crash"]
    model_frame["cond_regime_confidence"] = model_frame["regime_confidence"] > 0.70

    momentum_threshold = None
    if config.require_extreme_momentum and config.momentum_quantile is not None:
        threshold = model_frame["momentum"].dropna().quantile(config.momentum_quantile)
        momentum_threshold = float(threshold) if pd.notna(threshold) else None
    model_frame["cond_extreme_momentum"] = (
        model_frame["momentum"] >= momentum_threshold if momentum_threshold is not None else True
    )

    df = df.join(
        model_frame[
            [
                "state",
                "regime_name",
                "regime_confidence",
                "cond_regime_confidence",
                "cond_extreme_momentum",
            ]
        ],
        how="left",
    )
    df["is_bullish_regime"] = df["state"].isin(bull_states)
    df["is_negative_return_state"] = df["state"].isin(negative_states)
    df["is_bearish_regime"] = df["is_negative_return_state"] & (df["state"] != crash_state)
    df["is_crash_regime"] = df["state"] == crash_state

    return RegimeArtifacts(
        data=df,
        model=model,
        scaler=scaler,
        config=config,
        bull_states=bull_states,
        negative_states=negative_states,
        crash_state=crash_state,
        momentum_threshold=momentum_threshold,
        state_summary=state_summary,
    )
