"""Controllo anti-leakage: le feature di una partita non devono dipendere dal suo risultato."""

import numpy as np
import pandas as pd

from dunkest.features import FEATURES, build_features

TEAMS = {"AAA": 1, "BBB": 2, "CCC": 3}


def synthetic():
    rng = np.random.default_rng(0)
    players = pd.DataFrame({
        "PLAYER_ID": range(1, 13),
        "POSITION": ["G", "F", "C", "G"] * 3,
        "FROM_YEAR": 2015, "DRAFT_NUMBER": 10,
        "TEAM_ID": [1] * 4 + [2] * 4 + [3] * 4,
    })
    players["POS"] = players["POSITION"]
    team_of = dict(zip(players.PLAYER_ID, players.TEAM_ID))
    abbr = {v: k for k, v in TEAMS.items()}
    pairs = [("AAA", "BBB"), ("BBB", "CCC"), ("CCC", "AAA")]
    rows, gid = [], 0
    for day in range(30):
        h, a = pairs[day % 3]
        gid += 1
        for pid, tid in team_of.items():
            if abbr[tid] not in (h, a):
                continue
            mn = float(rng.uniform(10, 36))
            rows.append({"SEASON": "2024-25", "GAME_ID": f"G{gid:03d}",
                         "GAME_DATE": pd.Timestamp("2024-11-01") + pd.Timedelta(days=2 * day),
                         "PLAYER_ID": pid, "PLAYER_NAME": f"P{pid}", "TEAM_ID": tid, "TEAM": abbr[tid],
                         "OPP": a if abbr[tid] == h else h, "HOME": int(abbr[tid] == h),
                         "STARTER": int(pid % 4 != 0), "MIN": mn, "PDK": mn * rng.uniform(0.4, 1.2),
                         "USG_PCT": 0.2, "FGA": 10.0, "FG3A": 3.0, "FTA": 2.0, "TEAM_PACE": 100.0})
    hist = pd.DataFrame(rows)
    absences = pd.DataFrame({"GAME_ID": ["G020"], "TEAM_ID": [1], "PLAYER_ID": [1]})
    hist = hist[~((hist.GAME_ID == "G020") & (hist.PLAYER_ID == 1))]
    return hist, players, absences


def test_feature_non_dipendono_dal_risultato_della_partita():
    hist, players, absences = synthetic()
    target_game = "G025"
    a = build_features(hist, players, absences)

    changed = hist.copy()
    mask = changed["GAME_ID"] == target_game
    changed.loc[mask, "PDK"] *= 5          # cambio drasticamente il risultato della partita
    changed.loc[mask, "MIN"] = 48
    b = build_features(changed, players, absences)

    key = ["PLAYER_ID", "GAME_ID"]
    fa = a[a.GAME_ID == target_game].set_index(key)[FEATURES].sort_index()
    fb = b[b.GAME_ID == target_game].set_index(key)[FEATURES].sort_index()
    pd.testing.assert_frame_equal(fa, fb)


def test_assenti_contano_per_i_compagni():
    hist, players, absences = synthetic()
    df = build_features(hist, players, absences)
    teammates = df[(df.GAME_ID == "G020") & (df.TEAM_ID == 1)]
    others = df[(df.GAME_ID == "G020") & (df.TEAM_ID != 1)]
    assert (teammates["OUT_MIN"] > 0).all()
    assert (others["OUT_MIN"] == 0).all()
    assert (others["OPP_OUT_PDK"] > 0).all()
