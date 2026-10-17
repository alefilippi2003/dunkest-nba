import pandas as pd
import pytest

from dunkest.scoring import dunkest_score

BASE = dict(PTS=0, OREB=0, DREB=0, REB=0, AST=0, STL=0, BLK=0, TOV=0, BLKA=0,
            FGA=0, FGM=0, FTA=0, FTM=0, FG3M=0, STARTER=0, WL="L")


def score(**kw):
    row = {**BASE, **kw}
    row["REB"] = row["OREB"] + row["DREB"]
    return float(dunkest_score(pd.DataFrame([row])).iloc[0])


def test_linea_normale():
    # 10 + 3 dif + 2*1.25 off + 1.5*(4 ast + 1 rec - 2 perse) - 0.5 stoppata subita - 5 errori + 1 titolare
    s = score(PTS=10, OREB=2, DREB=3, AST=4, STL=1, TOV=2, BLKA=1, FGA=8, FGM=4, FTA=2, FTM=1,
              FG3M=1, STARTER=1)
    assert s == pytest.approx(15.5)


def test_bonus_vittoria_5_percento():
    s = score(PTS=10, OREB=2, DREB=3, AST=4, STL=1, TOV=2, BLKA=1, FGA=8, FGM=4, FTA=2, FTM=1,
              FG3M=1, STARTER=1, WL="W")
    assert s == pytest.approx(15.5 * 1.05, abs=0.01)


def test_bonus_vittoria_su_punteggio_negativo_lo_alza():
    # -3 tiri sbagliati -1.5 palla persa = -4.5; con la vittoria: -4.5 + 5% di 4.5
    assert score(FGA=3, TOV=1, WL="W") == pytest.approx(-4.5 + 0.225, abs=0.01)


def test_doppia_doppia():
    assert score(PTS=10, DREB=10, FGA=5, FGM=5) == pytest.approx(10 + 10 + 5)


def test_tripla_doppia_non_cumulativa_e_bonus_triple():
    # 20 punti + 10 rimbalzi + 10 assist*1.5 = 45, +10 tripla doppia, +4 per 4 triple segnate
    s = score(PTS=20, DREB=10, AST=10, FGA=8, FGM=8, FTA=4, FTM=4, FG3M=4)
    assert s == pytest.approx(45 + 10 + 4)


def test_bonus_triple_massimo_5():
    assert score(PTS=21, FGA=7, FGM=7, FG3M=7) == pytest.approx(21 + 5)


def test_bonus_triple_sotto_soglia():
    assert score(PTS=6, FGA=2, FGM=2, FG3M=2) == pytest.approx(6)
