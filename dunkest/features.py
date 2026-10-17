"""
Caricamento dei dati dal DB e costruzione delle feature.

Regola fondamentale: ogni feature di una riga (giocatore, partita) usa solo informazioni
disponibili PRIMA di quella partita. Le medie usano `shift()` (escludono la partita corrente)
e i join temporali usano `merge_asof(..., allow_exact_matches=False)` (solo date precedenti).
"""

import unicodedata

import numpy as np
import pandas as pd

# assenze note prima della partita (infortuni, motivi personali, riposo, sospensioni):
# escluse le scelte tecniche ("Coach's Decision"), che si scoprono solo a partita iniziata
KNOWN_ABSENCE = "INJURY|Injury|NOT_WITH_TEAM|PERSONAL|REST|SUSPENSION|Suspension|DND"

FEATURES = [
    # forma del giocatore
    "PDK_ewm3", "PDK_ewm10", "PDK_ewm30", "MIN_ewm3", "MIN_ewm10", "MIN_ewm30",
    "STARTER_ewm3", "STARTER_ewm10", "STARTER_ewm30", "RATE_ewm5", "RATE_ewm20",
    "USG_PCT_ewm10", "FGA_ewm10", "FTA_ewm10", "FG3A_ewm10",
    "PDK_std10", "MIN_last", "PDK_last", "GP_TOTAL", "GP_SEASON",
    "PDK_season", "MIN_season",
    # contesto (MIN_trend = forma minuti recente meno quella lunga)
    "DAYS_REST", "B2B", "MIN_trend", "EXP", "TEAM_CHANGED", "HOME", "POS_G", "POS_C", "DRAFT_PICK",
    # avversario e ritmo
    "OPP_ALLOW_ewm", "OPP_ALLOW_G_ewm", "OPP_ALLOW_F_ewm", "OPP_ALLOW_C_ewm", "OPP_ALLOW_POS",
    "OPP_PACE_ewm", "TEAM_PACE_ewm",
    # assenti
    "OUT_MIN", "OUT_PDK", "OUT_MIN_NEW", "OUT_PDK_NEW", "OUT_MIN_POS", "OPP_OUT_PDK",
]


# ============================================================================ caricamento

def strip_accents(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower()


def load_history(con) -> pd.DataFrame:
    cols = ["SEASON", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM", "OPP",
            "HOME", "STARTER", "MIN", "PDK", "USG_PCT", "FGA", "FG3A", "FTA", "TEAM_PACE"]
    df = pd.read_sql(f"SELECT {', '.join(cols)} FROM player_games", con)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def load_players(con) -> pd.DataFrame:
    pl = pd.read_sql("SELECT * FROM players", con)
    pl = pl.rename(columns={"PERSON_ID": "PLAYER_ID"})
    pl["POS"] = pl["POSITION"].fillna("F").replace("", "F").str[0]
    return pl


def load_absences(con) -> pd.DataFrame:
    box = pd.read_sql("SELECT GAME_ID, TEAM_ID, PLAYER_ID, STATUS, COMMENT FROM box_players", con)
    known = box["COMMENT"].fillna("").str.contains(KNOWN_ABSENCE, regex=True)
    out = box[(box["STATUS"] == "INACTIVE") & known]
    return out[["GAME_ID", "TEAM_ID", "PLAYER_ID"]].drop_duplicates()


def load_schedule(con, d0, d1) -> pd.DataFrame:
    """Partite di regular season tra d0 e d1 (date ET, estremi inclusi) dalla tabella schedule."""
    try:
        sched = pd.read_sql("SELECT * FROM schedule", con)
    except Exception:
        sched = pd.DataFrame()
    if sched.empty or "GAME_DATE_EST" not in sched.columns:
        raise SystemExit("calendario mancante: lancia prima `python main.py update`")
    sched = sched[sched["GAME_ID"].astype(str).str.startswith("002")].copy()
    sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE_EST"].astype(str).str[:10])
    return sched[(sched["GAME_DATE"] >= d0) & (sched["GAME_DATE"] <= d1)]


def build_future_rows(hist, players, sched, season) -> pd.DataFrame:
    """Una riga senza statistiche per ogni giocatore del roster attuale in ogni partita futura."""
    pl = players.copy()
    active = pl["TEAM_ID"].fillna(0).astype(int) != 0
    if "ROSTER_STATUS" in pl.columns and pl["ROSTER_STATUS"].notna().any():
        active &= pl["ROSTER_STATUS"].fillna(0).astype(float) == 1
    else:
        active &= pd.to_numeric(pl.get("TO_YEAR"), errors="coerce") >= int(season[:4])
    pl = pl[active]
    rows = []
    for g in sched.itertuples(index=False):
        for team_id, team, opp, home in ((g.HOME_ID, g.HOME, g.AWAY, 1), (g.AWAY_ID, g.AWAY, g.HOME, 0)):
            roster = pl[pl["TEAM_ID"].astype(int) == int(team_id)]
            for p in roster.itertuples(index=False):
                rows.append({"SEASON": season, "GAME_ID": str(g.GAME_ID), "GAME_DATE": g.GAME_DATE,
                             "PLAYER_ID": p.PLAYER_ID,
                             "PLAYER_NAME": f"{p.PLAYER_FIRST_NAME} {p.PLAYER_LAST_NAME}",
                             "TEAM_ID": int(team_id), "TEAM": team, "OPP": opp, "HOME": home})
    fut = pd.DataFrame(rows)
    # nomi come compaiono nei log NBA (con accenti), se il giocatore ha già giocato
    names = hist.drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["PLAYER_NAME"]
    fut["PLAYER_NAME"] = fut["PLAYER_ID"].map(names).fillna(fut["PLAYER_NAME"])
    return fut


# ============================================================================ feature

def _asof(left, right, by, cols):
    """Per ogni riga di left, i valori di right all'ultima data STRETTAMENTE precedente."""
    left = left.reset_index().sort_values("GAME_DATE")
    right = right.sort_values("GAME_DATE")[[by, "GAME_DATE"] + cols]
    out = pd.merge_asof(left, right, on="GAME_DATE", by=by, allow_exact_matches=False)
    return out.set_index("index").sort_index()[cols]


def build_features(df, players, absences) -> pd.DataFrame:
    """
    df: righe giocatore-partita, quelle giocate (con PDK) ed eventualmente quelle future (PDK NaN).
    players: anagrafica (load_players). absences: assenti noti prima della partita (load_absences).
    """
    df = df.merge(players[["PLAYER_ID", "POS", "FROM_YEAR", "DRAFT_NUMBER"]], on="PLAYER_ID", how="left")
    # scelta al draft (non scelto = 61): utile soprattutto per i rookie senza storico
    df["DRAFT_PICK"] = pd.to_numeric(df["DRAFT_NUMBER"], errors="coerce").fillna(61)
    df["POS"] = df["POS"].fillna("F")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["PLAYED"] = df["PDK"].notna().astype(int)
    df["RATE"] = df["PDK"] / df["MIN"].clip(lower=1)

    # ---- forma del giocatore (solo partite precedenti; le medie proseguono tra le stagioni)
    g = df.groupby("PLAYER_ID", sort=False)

    def ewm_prev(col, hl):
        return g[col].transform(lambda s: s.shift().ewm(halflife=hl, min_periods=1).mean())

    for hl in (3, 10, 30):
        for c in ("PDK", "MIN", "STARTER"):
            df[f"{c}_ewm{hl}"] = ewm_prev(c, hl)
    for hl in (5, 20):
        df[f"RATE_ewm{hl}"] = ewm_prev("PDK", hl) / ewm_prev("MIN", hl).clip(lower=1)
    for c in ("USG_PCT", "FGA", "FTA", "FG3A"):
        df[f"{c}_ewm10"] = ewm_prev(c, 10)
    df["PDK_std10"] = g["PDK"].transform(lambda s: s.shift().rolling(10, min_periods=3).std())
    df["MIN_last"] = g["MIN"].transform(lambda s: s.shift().ffill())
    df["PDK_last"] = g["PDK"].transform(lambda s: s.shift().ffill())
    df["GP_TOTAL"] = g["PLAYED"].transform(lambda s: s.shift().fillna(0).cumsum())
    gs = df.groupby(["PLAYER_ID", "SEASON"], sort=False)
    df["GP_SEASON"] = gs["PLAYED"].transform(lambda s: s.shift().fillna(0).cumsum())
    df["PDK_season"] = gs["PDK"].transform(lambda s: s.shift().expanding().mean())
    df["MIN_season"] = gs["MIN"].transform(lambda s: s.shift().expanding().mean())
    df["DAYS_REST"] = g["GAME_DATE"].diff().dt.days.clip(upper=30)
    df["B2B"] = (df["DAYS_REST"] == 1).astype(int)
    df["MIN_trend"] = df["MIN_ewm3"] - df["MIN_ewm30"]
    df["EXP"] = df["SEASON"].str[:4].astype(int) - pd.to_numeric(df["FROM_YEAR"], errors="coerce")
    df["TEAM_CHANGED"] = (g["TEAM_ID"].shift() != df["TEAM_ID"]).astype(int)
    df["POS_G"] = (df["POS"] == "G").astype(int)
    df["POS_C"] = (df["POS"] == "C").astype(int)

    # ---- difesa avversaria: PDK concessi (totali e per ruolo) e ritmo, media esponenziale
    played = df[df["PLAYED"] == 1]
    tg = played.groupby(["GAME_ID", "GAME_DATE", "TEAM", "OPP"]).agg(
        PTS_PDK=("PDK", "sum"), PACE=("TEAM_PACE", "first")).reset_index()
    by_pos = played.pivot_table(index=["GAME_ID", "TEAM"], columns="POS", values="PDK",
                                aggfunc="sum", fill_value=0)
    by_pos = by_pos.reindex(columns=["G", "F", "C"], fill_value=0)
    by_pos.columns = [f"PDK_{c}" for c in by_pos.columns]
    tg = tg.merge(by_pos.reset_index(), on=["GAME_ID", "TEAM"])
    # concesso da una difesa = prodotto dall'attacco che la affrontava
    allow = tg.rename(columns={"OPP": "DEF", "PTS_PDK": "ALLOW", "PDK_G": "ALLOW_G",
                               "PDK_F": "ALLOW_F", "PDK_C": "ALLOW_C"}).sort_values(["DEF", "GAME_DATE"])
    for c in ("ALLOW", "ALLOW_G", "ALLOW_F", "ALLOW_C"):
        allow[f"OPP_{c}_ewm"] = allow.groupby("DEF")[c].transform(lambda s: s.ewm(halflife=15).mean())
    pace = tg[["TEAM", "GAME_DATE", "PACE"]].sort_values(["TEAM", "GAME_DATE"])
    pace["PACE_ewm"] = pace.groupby("TEAM")["PACE"].transform(lambda s: s.ewm(halflife=15).mean())

    opp_cols = ["OPP_ALLOW_ewm", "OPP_ALLOW_G_ewm", "OPP_ALLOW_F_ewm", "OPP_ALLOW_C_ewm"]
    df[opp_cols] = _asof(df[["OPP", "GAME_DATE"]].rename(columns={"OPP": "DEF"}), allow, "DEF", opp_cols)
    df["OPP_ALLOW_POS"] = np.select([df["POS"] == "G", df["POS"] == "F"],
                                    [df["OPP_ALLOW_G_ewm"], df["OPP_ALLOW_F_ewm"]], df["OPP_ALLOW_C_ewm"])
    df["TEAM_PACE_ewm"] = _asof(df[["TEAM", "GAME_DATE"]], pace, "TEAM", ["PACE_ewm"])["PACE_ewm"].values
    df["OPP_PACE_ewm"] = _asof(df[["OPP", "GAME_DATE"]].rename(columns={"OPP": "TEAM"}), pace, "TEAM",
                               ["PACE_ewm"])["PACE_ewm"].values

    # ---- compagni e avversari assenti
    state = played[["PLAYER_ID", "GAME_DATE", "PDK", "MIN"]].sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    gg = state.groupby("PLAYER_ID")
    state["S_MIN"] = gg["MIN"].transform(lambda s: s.ewm(halflife=10).mean())
    state["S_PDK"] = gg["PDK"].transform(lambda s: s.ewm(halflife=10).mean())
    state["LAST_DATE"] = state["GAME_DATE"]
    ab = absences.merge(df[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID"), on="GAME_ID")
    ab = ab.merge(players[["PLAYER_ID", "POS"]], on="PLAYER_ID", how="left")
    ab["POS"] = ab["POS"].fillna("F")
    ab[["S_MIN", "S_PDK", "LAST_DATE"]] = _asof(ab[["PLAYER_ID", "GAME_DATE"]], state, "PLAYER_ID",
                                                ["S_MIN", "S_PDK", "LAST_DATE"]).values
    ab[["S_MIN", "S_PDK"]] = ab[["S_MIN", "S_PDK"]].astype(float).fillna(0)
    # assenze "nuove": giocava fino a 10 giorni prima (quelle lunghe sono già nella forma dei compagni)
    ab["NEW"] = ((ab["GAME_DATE"] - pd.to_datetime(ab["LAST_DATE"])).dt.days <= 10).astype(int)
    ab["S_MIN_NEW"] = ab["S_MIN"] * ab["NEW"]
    ab["S_PDK_NEW"] = ab["S_PDK"] * ab["NEW"]
    outs = ab.groupby(["GAME_ID", "TEAM_ID"]).agg(
        OUT_MIN=("S_MIN", "sum"), OUT_PDK=("S_PDK", "sum"),
        OUT_MIN_NEW=("S_MIN_NEW", "sum"), OUT_PDK_NEW=("S_PDK_NEW", "sum")).reset_index()
    outs_pos = ab.groupby(["GAME_ID", "TEAM_ID", "POS"]).agg(OUT_MIN_POS=("S_MIN_NEW", "sum")).reset_index()
    df = df.merge(outs, on=["GAME_ID", "TEAM_ID"], how="left")
    df = df.merge(outs_pos, on=["GAME_ID", "TEAM_ID", "POS"], how="left")
    teams_in_game = df[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opp_out = outs.merge(teams_in_game, on="GAME_ID", suffixes=("_ab", ""))
    opp_out = (opp_out[opp_out["TEAM_ID_ab"] != opp_out["TEAM_ID"]]
               .rename(columns={"OUT_PDK_NEW": "OPP_OUT_PDK"})[["GAME_ID", "TEAM_ID", "OPP_OUT_PDK"]])
    df = df.merge(opp_out, on=["GAME_ID", "TEAM_ID"], how="left")
    oc = ["OUT_MIN", "OUT_PDK", "OUT_MIN_NEW", "OUT_PDK_NEW", "OUT_MIN_POS", "OPP_OUT_PDK"]
    df[oc] = df[oc].fillna(0)
    df["HOME"] = df["HOME"].astype(int)
    return df


def training_rows(df) -> pd.DataFrame:
    """
    Righe usate per allenare: partite giocate con almeno una partita precedente nel DB,
    più gli esordi dei veri rookie (dalla seconda stagione del DB in poi: nella prima
    stagione ogni veterano risulterebbe esordiente).
    """
    first_season = df["SEASON"].min()
    rookie_debut = (df["EXP"] == 0) & (df["SEASON"] > first_season)
    return df[(df["PLAYED"] == 1) & ((df["GP_TOTAL"] > 0) | rookie_debut)]
