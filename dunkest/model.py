"""
Modelli gradient boosting (scikit-learn HistGradientBoostingRegressor).

  direct  PDK                      errore quadratico
  min     minuti                   errore quadratico
  rate    PDK al minuto            solo righe con >= 3 minuti, pesate per i minuti
  q10     10° percentile del PDK   loss quantilica (pavimento)
  q90     90° percentile del PDK   loss quantilica (tetto)

Previsione = (direct + min * rate) / 2.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES

GB_PARAMS = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=80,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                 random_state=0)
MIN_MINUTES_RATE = 3


def _gb(**extra):
    return HistGradientBoostingRegressor(**GB_PARAMS, **extra)


def fit_models(tr: pd.DataFrame, quantiles: bool = True) -> dict:
    X = tr[FEATURES]
    m = {"direct": _gb().fit(X, tr["PDK"]),
         "min": _gb().fit(X, tr["MIN"])}
    trr = tr[tr["MIN"] >= MIN_MINUTES_RATE]
    m["rate"] = _gb().fit(trr[FEATURES], trr["RATE"], sample_weight=trr["MIN"])
    if quantiles:
        for q in (0.1, 0.9):
            m[f"q{int(q * 100)}"] = _gb(loss="quantile", quantile=q).fit(X, tr["PDK"])
    return m


def predict_models(m: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    out = pd.DataFrame(index=df.index)
    out["MIN_PRED"] = np.clip(m["min"].predict(X), 0, 48)
    out["RATE_PRED"] = m["rate"].predict(X)
    out["PDK_DIRECT"] = m["direct"].predict(X)
    out["PDK_PRED"] = (out["PDK_DIRECT"] + out["MIN_PRED"] * out["RATE_PRED"]) / 2
    if "q10" in m:
        out["PDK_P10"] = np.minimum(m["q10"].predict(X), out["PDK_PRED"])
        out["PDK_P90"] = np.maximum(m["q90"].predict(X), out["PDK_PRED"])
    return out


def metrics(y, p, dates) -> str:
    """MAE, R² e correlazione di Spearman media tra i giocatori di uno stesso giorno."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    t = pd.DataFrame({"y": y, "p": p, "d": dates})
    sp = np.nanmean([spearmanr(x.y, x.p)[0] for _, x in t.groupby("d") if len(x) > 5])
    return f"MAE {np.abs(e).mean():5.2f}   R2 {r2:.3f}   Spearman/giorno {sp:.3f}"


def save_bundle(path, models: dict, trained_until: str, n_rows: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "trained_until": trained_until,
                     "n_rows": n_rows}, f)


def load_bundle(path) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"{path} non trovato: lancia prima `python main.py train`")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("features") != FEATURES:
        raise SystemExit("il modello salvato usa feature diverse da quelle del codice: "
                         "rilancia `python main.py train`")
    return bundle
