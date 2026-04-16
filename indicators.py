# ============================================================
# indicators.py — Indicadores técnicos y utilidades de datos
# ForexPulse AI Pro
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Símbolos Forex compatibles con yfinance ──────────────────
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "USDCHF=X",
}

TIMEFRAMES = {
    "5m":  {"interval": "5m",  "period": "60d"},
    "15m": {"interval": "15m", "period": "60d"},
}


# ── Cálculo de indicadores técnicos ─────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    pct_d = pct_k.rolling(d_period).mean()
    return pct_k, pct_d


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr = compute_atr(high, low, close, period)
    plus_di = 100 * compute_ema(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * compute_ema(minus_dm, period) / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return compute_ema(dx, period)


def add_all_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Añade todos los indicadores técnicos al DataFrame.
    params: dict con períodos optimizados (opcional).
    """
    p = {
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "stoch_k": 14,
        "stoch_d": 3,
        "adx_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    }
    if params:
        p.update(params)

    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    df["EMA_fast"] = compute_ema(close, p["ema_fast"])
    df["EMA_slow"] = compute_ema(close, p["ema_slow"])
    df["RSI"]      = compute_rsi(close, p["rsi_period"])

    macd, macd_sig, macd_hist = compute_macd(
        close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    df["MACD"]      = macd
    df["MACD_sig"]  = macd_sig
    df["MACD_hist"] = macd_hist

    bb_up, bb_mid, bb_lo = compute_bollinger(close, p["bb_period"], p["bb_std"])
    df["BB_upper"] = bb_up
    df["BB_mid"]   = bb_mid
    df["BB_lower"] = bb_lo

    df["ATR"]      = compute_atr(high, low, close, p["atr_period"])
    df["Stoch_K"], df["Stoch_D"] = compute_stochastic(
        high, low, close, p["stoch_k"], p["stoch_d"])
    df["ADX"]      = compute_adx(high, low, close, p["adx_period"])

    # ── Característica adicional: retorno logarítmico ──
    df["LogReturn"] = np.log(close / close.shift(1))

    return df.dropna()


# ── Señales de confirmación ──────────────────────────────────

def count_confirmations(row: pd.Series, direction: str, params: dict = None) -> int:
    """
    Cuenta cuántos indicadores confirman la dirección.
    direction: 'BUY' | 'SELL'
    Retorna número de confirmaciones (0-7).
    """
    p = {
        "rsi_oversold": 40,
        "rsi_overbought": 60,
        "adx_threshold": 20,
    }
    if params:
        p.update(params)

    score = 0
    buy = direction == "BUY"

    # 1. EMA crossover
    if buy and row["EMA_fast"] > row["EMA_slow"]:
        score += 1
    elif not buy and row["EMA_fast"] < row["EMA_slow"]:
        score += 1

    # 2. RSI no en zona extrema contraria
    if buy and row["RSI"] < p["rsi_overbought"]:
        score += 1
    elif not buy and row["RSI"] > p["rsi_oversold"]:
        score += 1

    # 3. MACD histogram dirección
    if buy and row["MACD_hist"] > 0:
        score += 1
    elif not buy and row["MACD_hist"] < 0:
        score += 1

    # 4. Precio vs Bollinger mid
    if buy and row["Close"] > row["BB_mid"]:
        score += 1
    elif not buy and row["Close"] < row["BB_mid"]:
        score += 1

    # 5. Stochastic
    if buy and row["Stoch_K"] > row["Stoch_D"] and row["Stoch_K"] < 80:
        score += 1
    elif not buy and row["Stoch_K"] < row["Stoch_D"] and row["Stoch_K"] > 20:
        score += 1

    # 6. ADX trend strength
    if row["ADX"] > p["adx_threshold"]:
        score += 1

    # 7. MACD line vs signal
    if buy and row["MACD"] > row["MACD_sig"]:
        score += 1
    elif not buy and row["MACD"] < row["MACD_sig"]:
        score += 1

    return score
