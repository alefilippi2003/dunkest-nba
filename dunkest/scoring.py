"""Punteggio Dunkest (PDK) calcolato dalle statistiche del boxscore."""

import numpy as np
import pandas as pd


def dunkest_score(df: pd.DataFrame) -> pd.Series:
    """
    Regolamento Dunkest (modalità classic), verificato su 24k PDK reali del 2023-24 (match 99.75%):

      +1      punto segnato
      +1      rimbalzo difensivo
      +1.25   rimbalzo offensivo
      +1.5    assist, recupero, stoppata
      -1.5    palla persa
      -0.5    stoppata subita
      -1      tiro dal campo sbagliato
      -1      tiro libero sbagliato
      +1      quintetto base
      +3/+4/+5  triple segnate: 3 / 4 / 5 o più
      +5/+10/+50  doppia / tripla / quadrupla doppia (su PTS, REB, AST, STL, BLK; non cumulativi)
      +5%     vittoria di squadra, calcolato sul valore assoluto del punteggio

    Colonne richieste: PTS, OREB, DREB, REB, AST, STL, BLK, TOV, BLKA, FGA, FGM, FTA, FTM,
    FG3M, STARTER (0/1, NaN = 0), WL ('W'/'L').
    Il capitano (x1.5) si applica in fase di formazione, non qui.
    """
    starter = df["STARTER"].fillna(0)
    missed = (df["FGA"] - df["FGM"]) + (df["FTA"] - df["FTM"])
    s = (df["PTS"] + df["DREB"] + 1.25 * df["OREB"]
         + 1.5 * (df["AST"] + df["STL"] + df["BLK"] - df["TOV"])
         - 0.5 * df["BLKA"] - missed + starter)
    n10 = (df[["PTS", "REB", "AST", "STL", "BLK"]] >= 10).sum(axis=1)
    s = s + np.select([n10 >= 4, n10 == 3, n10 == 2], [50, 10, 5], 0)
    s = s + np.where(df["FG3M"] >= 3, np.minimum(df["FG3M"], 5), 0)
    s = s + np.where(df["WL"] == "W", 0.05 * s.abs(), 0)
    return s.round(2)
