# ============================================================
# genetic_optimizer.py — Optimización Genética de parámetros
# ForexPulse AI Pro
# ============================================================

import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")

from indicators import add_all_indicators, count_confirmations


# ── Espacio de búsqueda de parámetros ───────────────────────

PARAM_SPACE = {
    # Períodos de indicadores
    "ema_fast":        (5, 20),
    "ema_slow":        (15, 50),
    "rsi_period":      (10, 21),
    "bb_period":       (15, 30),
    "bb_std":          (1.5, 2.5),
    "atr_period":      (10, 21),
    "stoch_k":         (10, 21),
    "stoch_d":         (2, 5),
    "adx_period":      (10, 21),
    "macd_fast":       (8, 15),
    "macd_slow":       (20, 30),
    "macd_signal":     (7, 12),
    # Umbrales de señal
    "lstm_threshold":  (0.60, 0.85),     # prob mínima LSTM
    "min_confirms":    (3, 6),           # confirmaciones técnicas mínimas
    "adx_threshold":   (18, 30),         # ADX mínimo para tendencia
    "rsi_oversold":    (30, 45),
    "rsi_overbought":  (55, 70),
    # Gestión de riesgo
    "sl_atr_mult":     (1.0, 3.0),       # multiplicador ATR para SL
    "tp_atr_mult":     (1.5, 4.0),       # multiplicador ATR para TP
}


@dataclass
class Individual:
    """Individuo del algoritmo genético."""
    genes: Dict[str, float]
    fitness: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0


# ── Motor de backtesting rápido (para fitness) ──────────────

def quick_backtest(df: pd.DataFrame, params: dict,
                   lstm_predictor=None) -> Tuple[float, float, int]:
    """
    Backtest simplificado para evaluación de fitness.
    Retorna (win_rate, profit_factor, n_trades).
    """
    if len(df) < 120:
        return 0.0, 0.0, 0

    df_ind = add_all_indicators(df, params)
    if len(df_ind) < 60:
        return 0.0, 0.0, 0

    wins, losses = [], []
    in_trade = False
    entry_price = 0.0
    direction = None
    sl_price, tp_price = 0.0, 0.0

    min_confirms = int(round(params.get("min_confirms", 4)))
    lstm_thresh  = params.get("lstm_threshold", 0.72)
    sl_mult      = params.get("sl_atr_mult", 1.5)
    tp_mult      = params.get("tp_atr_mult", 2.5)
    adx_thresh   = params.get("adx_threshold", 22)

    for i in range(60, len(df_ind) - 1):
        row  = df_ind.iloc[i]
        next_close = df_ind.iloc[i + 1]["Close"]
        atr = row["ATR"] if row["ATR"] > 0 else 1e-5

        # Gestión de trade abierto
        if in_trade:
            if direction == "BUY":
                if next_close <= sl_price:
                    losses.append(sl_price - entry_price)
                    in_trade = False
                elif next_close >= tp_price:
                    wins.append(tp_price - entry_price)
                    in_trade = False
            else:  # SELL
                if next_close >= sl_price:
                    losses.append(entry_price - sl_price)
                    in_trade = False
                elif next_close <= tp_price:
                    wins.append(entry_price - tp_price)
                    in_trade = False
            continue

        # Sin trade abierto: buscar señal
        if row["ADX"] < adx_thresh:
            continue

        # Estimar prob LSTM (si disponible, sino heurística)
        if lstm_predictor and hasattr(lstm_predictor, "is_trained") and lstm_predictor.is_trained:
            sub_df = df_ind.iloc[max(0, i - 80): i + 1]
            lstm_prob = lstm_predictor.predict_proba(sub_df)
        else:
            # Heurística de momentum rápida
            ret = row["LogReturn"] if "LogReturn" in row else 0
            lstm_prob = 0.5 + np.clip(ret * 200, -0.25, 0.25)

        if lstm_prob > lstm_thresh:
            conf = count_confirmations(row, "BUY", params)
            if conf >= min_confirms:
                direction    = "BUY"
                in_trade     = True
                entry_price  = row["Close"]
                sl_price     = entry_price - sl_mult * atr
                tp_price     = entry_price + tp_mult * atr

        elif lstm_prob < (1 - lstm_thresh):
            conf = count_confirmations(row, "SELL", params)
            if conf >= min_confirms:
                direction    = "SELL"
                in_trade     = True
                entry_price  = row["Close"]
                sl_price     = entry_price + sl_mult * atr
                tp_price     = entry_price - tp_mult * atr

    n_trades = len(wins) + len(losses)
    if n_trades == 0:
        return 0.0, 0.0, 0

    win_rate = len(wins) / n_trades
    gross_profit = sum(wins) if wins else 0.0
    gross_loss   = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss

    return win_rate, profit_factor, n_trades


# ── Algoritmo Genético ───────────────────────────────────────

class GeneticOptimizer:
    """
    Optimizador genético para maximizar win_rate × profit_factor.
    Usa selección por torneo, cruce de un punto y mutación gaussiana.
    """

    def __init__(self,
                 population_size: int = 30,
                 generations: int = 20,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.75,
                 elite_size: int = 3,
                 seed: int = 42):
        self.pop_size      = population_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size    = elite_size
        self.seed          = seed
        self.best_params   = None
        self.best_fitness  = 0.0
        self.history       = []   # fitness por generación

        random.seed(seed)
        np.random.seed(seed)

    # ── Inicialización ────────────────────────────────────────
    def _random_individual(self) -> Individual:
        genes = {}
        for key, (lo, hi) in PARAM_SPACE.items():
            if key in ("stoch_d", "min_confirms", "adx_period",
                       "atr_period", "rsi_period", "stoch_k",
                       "macd_fast", "macd_slow", "macd_signal",
                       "bb_period", "ema_fast", "ema_slow"):
                genes[key] = float(random.randint(int(lo), int(hi)))
            else:
                genes[key] = random.uniform(lo, hi)
        # Restricción: ema_fast < ema_slow, macd_fast < macd_slow
        if genes["ema_fast"] >= genes["ema_slow"]:
            genes["ema_slow"] = genes["ema_fast"] + random.randint(5, 15)
        if genes["macd_fast"] >= genes["macd_slow"]:
            genes["macd_slow"] = genes["macd_fast"] + random.randint(5, 10)
        return Individual(genes=genes)

    # ── Fitness ───────────────────────────────────────────────
    def _evaluate(self, ind: Individual, df: pd.DataFrame,
                  lstm_predictor=None) -> Individual:
        wr, pf, n = quick_backtest(df, ind.genes, lstm_predictor)
        # Penalizar si hay muy pocos trades (puede ser sobreajuste)
        trade_bonus = min(1.0, n / 30.0)
        fitness = wr * min(pf, 5.0) * trade_bonus
        ind.fitness       = fitness
        ind.win_rate      = wr
        ind.profit_factor = pf
        ind.n_trades      = n
        return ind

    # ── Selección por torneo ──────────────────────────────────
    def _tournament(self, population: List[Individual], k: int = 3) -> Individual:
        contestants = random.sample(population, min(k, len(population)))
        return max(contestants, key=lambda x: x.fitness)

    # ── Cruce ─────────────────────────────────────────────────
    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        keys = list(PARAM_SPACE.keys())
        if random.random() > self.crossover_rate:
            return p1, p2
        point = random.randint(1, len(keys) - 1)
        g1, g2 = {}, {}
        for i, k in enumerate(keys):
            if i < point:
                g1[k], g2[k] = p1.genes[k], p2.genes[k]
            else:
                g1[k], g2[k] = p2.genes[k], p1.genes[k]
        return Individual(genes=g1), Individual(genes=g2)

    # ── Mutación gaussiana ────────────────────────────────────
    def _mutate(self, ind: Individual) -> Individual:
        genes = ind.genes.copy()
        for key, (lo, hi) in PARAM_SPACE.items():
            if random.random() < self.mutation_rate:
                sigma = (hi - lo) * 0.1
                val = genes[key] + random.gauss(0, sigma)
                val = np.clip(val, lo, hi)
                # Mantener entero si corresponde
                if key in ("stoch_d", "min_confirms", "adx_period",
                           "atr_period", "rsi_period", "stoch_k",
                           "macd_fast", "macd_slow", "macd_signal",
                           "bb_period", "ema_fast", "ema_slow"):
                    val = float(int(round(val)))
                genes[key] = val
        # Restricciones
        if genes["ema_fast"] >= genes["ema_slow"]:
            genes["ema_slow"] = genes["ema_fast"] + 5
        if genes["macd_fast"] >= genes["macd_slow"]:
            genes["macd_slow"] = genes["macd_fast"] + 5
        return Individual(genes=genes)

    # ── Loop principal ────────────────────────────────────────
    def run(self, df: pd.DataFrame, lstm_predictor=None,
            progress_callback=None) -> Dict:
        """
        Ejecuta la optimización genética.
        progress_callback(gen, total, best_fitness) → para actualizar UI.
        """
        population = [self._random_individual() for _ in range(self.pop_size)]

        # Evaluación inicial
        for ind in population:
            self._evaluate(ind, df, lstm_predictor)

        self.history = []

        for gen in range(self.generations):
            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]
            self.history.append({
                "generation": gen + 1,
                "best_fitness": best.fitness,
                "best_wr": best.win_rate,
                "best_pf": best.profit_factor,
                "best_trades": best.n_trades,
            })

            if progress_callback:
                progress_callback(gen + 1, self.generations, best.fitness)

            # Elitismo
            new_pop = population[:self.elite_size]

            # Generar descendencia
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population)
                p2 = self._tournament(population)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                self._evaluate(c1, df, lstm_predictor)
                self._evaluate(c2, df, lstm_predictor)
                new_pop.extend([c1, c2])

            population = new_pop[:self.pop_size]

        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        self.best_params  = best.genes
        self.best_fitness = best.fitness

        return {
            "best_params":        best.genes,
            "best_fitness":       best.fitness,
            "best_win_rate":      best.win_rate,
            "best_profit_factor": best.profit_factor,
            "best_n_trades":      best.n_trades,
            "history":            self.history,
            "top5": [
                {
                    "fitness": p.fitness,
                    "win_rate": p.win_rate,
                    "profit_factor": p.profit_factor,
                    "n_trades": p.n_trades,
                }
                for p in population[:5]
            ],
        }

    # ── Parámetros por defecto si no se ha optimizado ────────
    @staticmethod
    def default_params() -> dict:
        return {k: (lo + hi) / 2 for k, (lo, hi) in PARAM_SPACE.items()}
