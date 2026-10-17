"""Validazione per stagione e allenamento dei modelli finali."""

import time

import numpy as np

from .features import build_features, load_absences, load_history, load_players, training_rows
from .model import fit_models, metrics, predict_models, save_bundle


def validate(data) -> None:
    """Allena sulle stagioni precedenti e testa sulle ultime due stagioni complete."""
    seasons = sorted(data["SEASON"].unique())
    tests = [s for s in seasons[1:] if (data["SEASON"] == s).sum() > 3000][-2:]
    print("\nvalidazione (alleno sulle stagioni precedenti, testo sulla stagione indicata)")
    print("Spearman/giorno = quanto bene il modello ordina i giocatori nelle partite di un giorno")
    for s in tests:
        tr, te = data[data["SEASON"] < s], data[data["SEASON"] == s]
        m = fit_models(tr, quantiles=True)
        te = te[te["GP_TOTAL"] > 0]   # confronto alla pari con la media mobile (serve storico)
        p = predict_models(m, te)
        y, d = te["PDK"].values, te["GAME_DATE"].values
        base = te["PDK_ewm10"].fillna(te["PDK_ewm30"]).values
        cover = ((te["PDK"] >= p["PDK_P10"]) & (te["PDK"] <= p["PDK_P90"])).mean()
        print(f"\n  {s}  (train {len(tr)}, test {len(te)})")
        print(f"    media mobile (baseline)  {metrics(y, base, d)}")
        print(f"    modello                  {metrics(y, p['PDK_PRED'].values, d)}")
        print(f"    minuti: MAE modello {np.abs(te['MIN'] - p['MIN_PRED']).mean():.2f} "
              f"vs media mobile {np.abs(te['MIN'] - te['MIN_ewm10']).mean():.2f}")
        print(f"    intervallo P10-P90 contiene il valore reale nel {cover:.0%} dei casi (atteso ~80%)")


def run_train(con, model_path, do_validate=True) -> None:
    t0 = time.time()
    print("carico i dati e costruisco le feature...")
    df = build_features(load_history(con), load_players(con), load_absences(con))
    data = training_rows(df)
    print(f"  {len(data)} righe di training, {time.time() - t0:.0f}s")

    if do_validate:
        validate(data)

    print("\nalleno i modelli finali su tutte le stagioni...")
    m = fit_models(data, quantiles=True)
    until = str(data["GAME_DATE"].max().date())
    save_bundle(model_path, m, until, len(data))
    print(f"salvato {model_path} (dati fino al {until}) in {time.time() - t0:.0f}s")
