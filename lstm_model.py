# ============================================================
# lstm_model.py — Motor LSTM para predicción de dirección
# ForexPulse AI Pro
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Importación condicional de TensorFlow ────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                          BatchNormalization, Bidirectional)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA_fast", "EMA_slow", "RSI",
    "MACD", "MACD_sig", "MACD_hist",
    "BB_upper", "BB_mid", "BB_lower",
    "ATR", "Stoch_K", "Stoch_D", "ADX", "LogReturn",
]

SEQ_LEN = 60  # ventana de secuencia (número de velas)


class LSTMPredictor:
    """
    Modelo LSTM híbrido:
    - 2 capas LSTM bidireccionales + 1 LSTM simple
    - BatchNorm + Dropout para regularización
    - Salida: probabilidad de movimiento ALCISTA (>0.5 = BUY)
    """

    def __init__(self, seq_len: int = SEQ_LEN, n_features: int = len(FEATURES)):
        self.seq_len = seq_len
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        self.feature_cols = FEATURES

    # ── Construcción del modelo ──────────────────────────────
    def build(self):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow no disponible. Instala con: pip install tensorflow")

        tf.random.set_seed(42)
        model = Sequential([
            Bidirectional(
                LSTM(128, return_sequences=True),
                input_shape=(self.seq_len, self.n_features)
            ),
            BatchNormalization(),
            Dropout(0.25),

            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.20),

            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.15),

            Dense(16, activation="relu"),
            Dropout(0.10),
            Dense(1, activation="sigmoid"),   # probabilidad BUY
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    # ── Preparación de datos ─────────────────────────────────
    def _prepare_features(self, df: pd.DataFrame):
        """Selecciona y normaliza features disponibles."""
        cols = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols_used = cols
        X_raw = df[cols].values.astype(np.float32)
        return X_raw

    def _make_sequences(self, X_scaled: np.ndarray, y: np.ndarray = None):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - self.seq_len: i])
            if y is not None:
                ys.append(y[i])
        Xs = np.array(Xs, dtype=np.float32)
        if y is not None:
            return Xs, np.array(ys, dtype=np.float32)
        return Xs

    # ── Entrenamiento ────────────────────────────────────────
    def train(self, df: pd.DataFrame, epochs: int = 40, batch_size: int = 32,
              validation_split: float = 0.15, verbose: int = 0):
        """
        Entrena el modelo con los datos históricos.
        Target: 1 si el cierre siguiente > cierre actual (BUY), 0 si no (SELL).
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow no disponible"}

        X_raw = self._prepare_features(df)
        # Target: dirección del siguiente cierre
        close_idx = list(df.columns).index("Close") if "Close" in df.columns else 3
        y_raw = (df["Close"].shift(-1) > df["Close"]).astype(int).values[:-1]

        # Ajustar scaler en datos de entrenamiento
        X_scaled = self.scaler.fit_transform(X_raw)

        # Crear secuencias (alineadas con y)
        X_scaled_body = X_scaled[:-1]  # quitar último (no tiene target)
        Xs, ys = self._make_sequences(X_scaled_body, y_raw[self.seq_len:])

        if len(Xs) < 50:
            return {"error": "Datos insuficientes para entrenamiento"}

        # Split temporal (sin shuffle para evitar data leakage)
        n_val = int(len(Xs) * validation_split)
        X_train, X_val = Xs[:-n_val], Xs[-n_val:]
        y_train, y_val = ys[:-n_val], ys[-n_val:]

        if self.model is None:
            self.n_features = Xs.shape[2]
            self.build()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=4, min_lr=1e-5, verbose=0),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False,  # datos temporales: NO mezclar
        )

        self.is_trained = True
        # Calcular accuracy en validación
        val_pred = (self.model.predict(X_val, verbose=0).flatten() > 0.5).astype(int)
        val_acc = np.mean(val_pred == y_val)

        return {
            "val_accuracy": float(val_acc),
            "epochs_run": len(history.history["loss"]),
            "final_val_loss": float(history.history["val_loss"][-1]),
        }

    # ── Predicción ───────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        Retorna la probabilidad de movimiento ALCISTA (0-1).
        Usa las últimas `seq_len` filas.
        """
        if not self.is_trained or self.model is None:
            return 0.5  # sin modelo → neutral

        X_raw = self._prepare_features(df)
        if len(X_raw) < self.seq_len:
            return 0.5

        X_scaled = self.scaler.transform(X_raw)
        seq = X_scaled[-self.seq_len:]
        seq = seq.reshape(1, self.seq_len, -1)

        prob = float(self.model.predict(seq, verbose=0)[0][0])
        return prob

    # ── Forecast de múltiples pasos ──────────────────────────
    def forecast_next_n(self, df: pd.DataFrame, n: int = 5) -> list:
        """
        Genera n predicciones de probabilidad iterativas (autoregresivo simplificado).
        Solo para visualización indicativa.
        """
        probs = []
        temp_df = df.copy()
        for _ in range(n):
            p = self.predict_proba(temp_df)
            probs.append(p)
            # Simular siguiente vela (simplificado: mismo cierre ± ATR medio)
            last = temp_df.iloc[-1].copy()
            last["Close"] = last["Close"] * (1 + (0.0002 if p > 0.5 else -0.0002))
            temp_df = pd.concat([temp_df, last.to_frame().T], ignore_index=True)
        return probs


# ── Modelo fallback sin TensorFlow ──────────────────────────

class SimpleTrendPredictor:
    """
    Predictor de respaldo basado en reglas cuando TF no está disponible.
    Usa momentum + RSI + MACD para estimar probabilidad.
    """

    def __init__(self):
        self.is_trained = True

    def train(self, df, **kwargs):
        return {"val_accuracy": 0.0, "note": "Modelo de respaldo (sin TF)"}

    def predict_proba(self, df: pd.DataFrame) -> float:
        if len(df) < 10:
            return 0.5
        row = df.iloc[-1]
        score = 0.5

        if "EMA_fast" in df.columns and "EMA_slow" in df.columns:
            score += 0.1 if row["EMA_fast"] > row["EMA_slow"] else -0.1
        if "RSI" in df.columns:
            score += 0.05 if row["RSI"] < 50 else -0.05
        if "MACD_hist" in df.columns:
            score += 0.08 if row["MACD_hist"] > 0 else -0.08
        if "Stoch_K" in df.columns and "Stoch_D" in df.columns:
            score += 0.07 if row["Stoch_K"] > row["Stoch_D"] else -0.07

        return float(np.clip(score, 0.05, 0.95))

    def forecast_next_n(self, df, n=5):
        base = self.predict_proba(df)
        return [base] * n


def get_predictor():
    """Retorna el mejor predictor disponible."""
    if TF_AVAILABLE:
        return LSTMPredictor()
    return SimpleTrendPredictor()
