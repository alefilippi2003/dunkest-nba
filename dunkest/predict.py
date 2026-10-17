"""Previsioni per le partite in calendario in un intervallo di date."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import season_of
from .features import (build_features, build_future_rows, load_absences, load_history,
                       load_players, load_schedule, strip_accents)
from .model import load_bundle, predict_models


def _apply_out(fut, absences, out_names):
    """Segna come assenti i giocatori indicati: escono dalle previsioni ed entrano nelle feature 'assenti'."""
    wanted = [strip_accents(x.strip()) for x in out_names.split(",") if x.strip()]
    names = fut.drop_duplicates("PLAYER_ID")[["PLAYER_ID", "PLAYER_NAME", "TEAM"]]
    norm = names["PLAYER_NAME"].map(strip_accents)
    out_ids = []
    for w in wanted:
        hit = names[norm.str.contains(w, regex=False)]
        if hit.empty:
            print(f"  ! '{w}' non trovato nei roster")
        for r in hit.itertuples(index=False):
            print(f"  assente: {r.PLAYER_NAME} ({r.TEAM})")
            out_ids.append(r.PLAYER_ID)
    fut_out = fut[fut["PLAYER_ID"].isin(out_ids)][["GAME_ID", "TEAM_ID", "PLAYER_ID"]]
    absences = pd.concat([absences, fut_out], ignore_index=True)
    return fut[~fut["PLAYER_ID"].isin(out_ids)], absences


def run_predict(con, model_path, output_dir, d0, d1, out_names="", top=25) -> pd.DataFrame:
    bundle = load_bundle(model_path)
    print(f"modelli allenati con dati fino al {bundle['trained_until']}")
    try:
        d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    except ValueError as e:
        raise SystemExit(f"date non valide ({e}): usa il formato YYYY-MM-DD")
    season = season_of(d0)

    hist = load_history(con)
    players = load_players(con)
    sched = load_schedule(con, d0, d1)
    sched = sched[~sched["GAME_ID"].astype(str).isin(set(hist["GAME_ID"]))]   # solo partite non giocate
    if sched.empty:
        raise SystemExit(f"nessuna partita da prevedere tra {d0.date()} e {d1.date()}")
    print(f"  {len(sched)} partite da prevedere tra {d0.date()} e {d1.date()}")

    fut = build_future_rows(hist, players, sched, season)
    absences = load_absences(con)
    if out_names:
        fut, absences = _apply_out(fut, absences, out_names)

    df = build_features(pd.concat([hist, fut], ignore_index=True), players, absences)
    fx = df[df["PLAYED"] == 0].copy()
    fx = pd.concat([fx, predict_models(bundle["models"], fx)], axis=1)

    last_played = hist.groupby("PLAYER_ID")["GAME_DATE"].max()
    fx["ULTIMA_PARTITA"] = fx["PLAYER_ID"].map(last_played).dt.date
    fx["AVVERSARIO"] = np.where(fx["HOME"] == 1, "vs ", "@ ") + fx["OPP"]

    per_game = fx[["GAME_DATE", "PLAYER_NAME", "TEAM", "POS", "AVVERSARIO", "MIN_PRED", "PDK_PRED",
                   "PDK_P10", "PDK_P90", "GP_TOTAL", "ULTIMA_PARTITA"]].copy()
    per_game["GAME_DATE"] = per_game["GAME_DATE"].dt.date
    per_game["GP_TOTAL"] = per_game["GP_TOTAL"].astype(int)

    summary = (fx.groupby(["PLAYER_ID", "PLAYER_NAME", "TEAM", "POS"])
               .agg(PARTITE=("PDK_PRED", "size"), AVVERSARI=("AVVERSARIO", ", ".join),
                    MIN_MEDI=("MIN_PRED", "mean"), PDK_MEDIO=("PDK_PRED", "mean"),
                    PDK_TOTALE=("PDK_PRED", "sum"), P10_MEDIO=("PDK_P10", "mean"),
                    P90_MEDIO=("PDK_P90", "mean"), STORICO_PARTITE=("GP_TOTAL", "max"),
                    ULTIMA_PARTITA=("ULTIMA_PARTITA", "first"))
               .reset_index().drop(columns="PLAYER_ID")
               .sort_values("PDK_TOTALE", ascending=False))
    # avvisi: pochi dati storici, o fermo da molto prima dell'ultima partita nel DB (la pausa estiva non conta)
    stale = pd.to_datetime(summary["ULTIMA_PARTITA"]) < hist["GAME_DATE"].max() - pd.Timedelta(days=30)
    summary["NOTE"] = np.where(summary["STORICO_PARTITE"] < 10, "pochi dati",
                               np.where(stale, "fermo da tempo: verifica", ""))
    summary["STORICO_PARTITE"] = summary["STORICO_PARTITE"].astype(int)
    for c in ("MIN_MEDI", "PDK_MEDIO", "PDK_TOTALE", "P10_MEDIO", "P90_MEDIO"):
        summary[c] = summary[c].round(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{d0.date()}_{d1.date()}"
    f_players = output_dir / f"predizioni_{tag}.csv"
    f_games = output_dir / f"predizioni_partite_{tag}.csv"
    summary.to_csv(f_players, index=False)
    (per_game.round(1).sort_values(["GAME_DATE", "PDK_PRED"], ascending=[True, False])
     .to_csv(f_games, index=False))
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\ntop {top} per PDK totale nel periodo:\n{summary.head(top).to_string(index=False)}")
    print(f"\nscritti {f_players} (per giocatore) e {f_games} (per partita)")
    return summary
