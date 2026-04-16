# ============================================================
# backtester.py — Motor de backtesting profesional walk-forward
# ForexPulse AI Pro
# ============================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

from indicators import add_all_indicators, count_confirmations


@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   Optional[pd.Timestamp]
    direction:   str      # BUY | SELL
    entry_price: float
    exit_price:  float    = 0.0
    sl_price:    float    = 0.0
    tp_price:    float    = 0.0
    pnl_pips:    float    = 0.0
    outcome:     str      = "OPEN"   # WIN | LOSS | OPEN
    lstm_prob:   float    = 0.5
    confirms:    int      = 0


class Backtester:
    """
    Backtesting profesional con walk-forward y métricas completas.
    """

    def __init__(self, initial_capital: float = 10_000.0,
                 risk_pct: float = 1.0):
        self.initial_capital = initial_capital
        self.risk_pct        = risk_pct  # % del capital arriesgado por trade
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    # ── Backtest completo ────────────────────────────────────
    def run(self, df: pd.DataFrame, params: dict,
            lstm_predictor=None) -> Dict:
        """
        Ejecuta el backtest con los parámetros dados.
        """
        self.trades = []
        df_ind = add_all_indicators(df, params)
        if len(df_ind) < 80:
            return self._empty_metrics()

        min_confirms  = int(round(params.get("min_confirms", 4)))
        lstm_thresh   = params.get("lstm_threshold", 0.72)
        sl_mult       = params.get("sl_atr_mult", 1.5)
        tp_mult       = params.get("tp_atr_mult", 2.5)
        adx_thresh    = params.get("adx_threshold", 22)

        capital   = self.initial_capital
        equity    = [capital]
        in_trade  = False
        current_trade: Optional[Trade] = None

        for i in range(60, len(df_ind) - 1):
            row       = df_ind.iloc[i]
            next_row  = df_ind.iloc[i + 1]
            atr       = max(row["ATR"], 1e-5)

            # Gestión de trade abierto
            if in_trade and current_trade:
                exit_price = None
                outcome    = None

                if current_trade.direction == "BUY":
                    if next_row["Low"] <= current_trade.sl_price:
                        exit_price = current_trade.sl_price
                        outcome    = "LOSS"
                    elif next_row["High"] >= current_trade.tp_price:
                        exit_price = current_trade.tp_price
                        outcome    = "WIN"
                else:  # SELL
                    if next_row["High"] >= current_trade.sl_price:
                        exit_price = current_trade.sl_price
                        outcome    = "LOSS"
                    elif next_row["Low"] <= current_trade.tp_price:
                        exit_price = current_trade.tp_price
                        outcome    = "WIN"

                if exit_price is not None:
                    if current_trade.direction == "BUY":
                        pnl_pips = (exit_price - current_trade.entry_price) / atr
                    else:
                        pnl_pips = (current_trade.entry_price - exit_price) / atr

                    # Tamaño de posición basado en riesgo
                    risk_amount  = capital * (self.risk_pct / 100)
                    position_val = risk_amount / (sl_mult * atr + 1e-10)

                    pnl_dollars = pnl_pips * sl_mult * risk_amount
                    capital    += pnl_dollars

                    current_trade.exit_time  = next_row.name if hasattr(next_row, "name") else None
                    current_trade.exit_price = exit_price
                    current_trade.pnl_pips   = pnl_pips
                    current_trade.outcome    = outcome
                    self.trades.append(current_trade)
                    in_trade      = False
                    current_trade = None

                equity.append(capital)
                continue

            # Buscar nueva señal
            if row["ADX"] < adx_thresh:
                equity.append(capital)
                continue

            # LSTM prob
            if lstm_predictor and hasattr(lstm_predictor, "is_trained") and lstm_predictor.is_trained:
                sub = df_ind.iloc[max(0, i - 80): i + 1]
                lstm_prob = lstm_predictor.predict_proba(sub)
            else:
                ret = row.get("LogReturn", 0)
                lstm_prob = float(np.clip(0.5 + ret * 200, 0.1, 0.9))

            new_trade = None

            if lstm_prob >= lstm_thresh:
                conf = count_confirmations(row, "BUY", params)
                if conf >= min_confirms:
                    new_trade = Trade(
                        entry_time  = row.name if hasattr(row, "name") else pd.Timestamp.now(),
                        exit_time   = None,
                        direction   = "BUY",
                        entry_price = row["Close"],
                        sl_price    = row["Close"] - sl_mult * atr,
                        tp_price    = row["Close"] + tp_mult * atr,
                        lstm_prob   = lstm_prob,
                        confirms    = conf,
                    )
            elif lstm_prob <= (1 - lstm_thresh):
                conf = count_confirmations(row, "SELL", params)
                if conf >= min_confirms:
                    new_trade = Trade(
                        entry_time  = row.name if hasattr(row, "name") else pd.Timestamp.now(),
                        exit_time   = None,
                        direction   = "SELL",
                        entry_price = row["Close"],
                        sl_price    = row["Close"] + sl_mult * atr,
                        tp_price    = row["Close"] - tp_mult * atr,
                        lstm_prob   = lstm_prob,
                        confirms    = conf,
                    )

            if new_trade:
                in_trade      = True
                current_trade = new_trade

            equity.append(capital)

        self.equity_curve = equity
        return self._compute_metrics()

    # ── Walk-Forward ─────────────────────────────────────────
    def walk_forward(self, df: pd.DataFrame, params: dict,
                     n_windows: int = 4, train_ratio: float = 0.7,
                     lstm_predictor=None) -> List[Dict]:
        """
        Walk-forward validation: divide el dataset en ventanas
        y evalúa en cada periodo out-of-sample.
        """
        window_size = len(df) // n_windows
        results = []

        for i in range(n_windows):
            start = i * window_size
            end   = min(start + window_size, len(df))
            if end - start < 100:
                continue

            train_end = start + int((end - start) * train_ratio)
            test_df   = df.iloc[train_end: end]

            if len(test_df) < 60:
                continue

            bt = Backtester(self.initial_capital, self.risk_pct)
            metrics = bt.run(test_df, params, lstm_predictor)
            metrics["window"] = i + 1
            metrics["test_start"] = str(test_df.index[0])[:10] if len(test_df) > 0 else ""
            metrics["test_end"]   = str(test_df.index[-1])[:10] if len(test_df) > 0 else ""
            results.append(metrics)

        return results

    # ── Métricas ──────────────────────────────────────────────
    def _compute_metrics(self) -> Dict:
        trades = [t for t in self.trades if t.outcome != "OPEN"]
        if not trades:
            return self._empty_metrics()

        wins   = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        n      = len(trades)

        win_rate      = len(wins) / n if n > 0 else 0
        gross_profit  = sum(t.pnl_pips for t in wins) if wins else 0
        gross_loss    = abs(sum(t.pnl_pips for t in losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        pnl_series    = [t.pnl_pips for t in trades]
        expectancy    = np.mean(pnl_series) if pnl_series else 0

        # Equity curve → Sharpe y Drawdown
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-10)
        sharpe  = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0

        # Max Drawdown
        peak     = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-10)
        max_dd   = float(np.min(drawdown))

        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return  = (final_capital - self.initial_capital) / self.initial_capital

        return {
            "n_trades":      n,
            "n_wins":        len(wins),
            "n_losses":      len(losses),
            "win_rate":      round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy":    round(expectancy, 4),
            "sharpe":        round(sharpe, 2),
            "max_drawdown":  round(max_dd * 100, 2),
            "total_return":  round(total_return * 100, 2),
            "final_capital": round(final_capital, 2),
            "trades":        trades,
            "equity_curve":  self.equity_curve,
        }

    def _empty_metrics(self) -> Dict:
        return {
            "n_trades": 0, "n_wins": 0, "n_losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "total_return": 0.0,
            "final_capital": self.initial_capital,
            "trades": [], "equity_curve": [self.initial_capital],
        }
