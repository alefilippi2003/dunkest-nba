"""Costruzione della tabella finale `player_games`: una riga per giocatore-partita con il PDK."""

import numpy as np
import pandas as pd

from .db import table_exists
from .scoring import dunkest_score

ADV_KEEP = ["PLAYER_ID", "GAME_ID", "USG_PCT", "E_USG_PCT", "PACE", "E_PACE", "POSS",
            "OFF_RATING", "DEF_RATING", "AST_PCT", "OREB_PCT", "DREB_PCT", "TS_PCT", "PIE"]
TEAM_ADV_KEEP = ["TEAM_ID", "GAME_ID", "PACE", "POSS", "OFF_RATING", "DEF_RATING"]


def build_player_games(con, export_csv=None) -> pd.DataFrame | None:
    print("costruisco player_games")
    if not table_exists(con, "raw_player_base"):
        print("  manca raw_player_base: lancia prima `python main.py update --all`")
        return None
    df = pd.read_sql("SELECT * FROM raw_player_base", con)

    if table_exists(con, "raw_player_adv"):
        adv = pd.read_sql("SELECT * FROM raw_player_adv", con)
        adv = adv[[c for c in ADV_KEEP if c in adv.columns]]
        df = df.merge(adv, on=["PLAYER_ID", "GAME_ID"], how="left")

    # titolari dai boxscore (NaN dove il boxscore non è ancora stato scaricato)
    df["STARTER"] = np.nan
    if table_exists(con, "box_players"):
        box = (pd.read_sql("SELECT GAME_ID, PLAYER_ID, STARTER AS ST_BOX FROM box_players", con)
               .drop_duplicates(["GAME_ID", "PLAYER_ID"]))
        df = df.merge(box, on=["GAME_ID", "PLAYER_ID"], how="left")
        df["STARTER"] = df["ST_BOX"]
        df = df.drop(columns="ST_BOX")

    # contesto partita
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["OPP"] = df["MATCHUP"].str.split().str[-1]
    df = df.rename(columns={"TEAM_ABBREVIATION": "TEAM", "SEASON_YEAR": "SEASON_Y"})

    # ritmo e rating di squadra e avversario nella partita (le feature li useranno solo "dal passato")
    if table_exists(con, "raw_team_adv"):
        tadv = pd.read_sql("SELECT * FROM raw_team_adv", con)
        tadv = tadv[[c for c in TEAM_ADV_KEEP if c in tadv.columns]]
        tb = pd.read_sql("SELECT TEAM_ID, GAME_ID, TEAM_ABBREVIATION FROM raw_team_base", con)
        tadv = tadv.merge(tb, on=["TEAM_ID", "GAME_ID"], how="left")
        own = tadv.add_prefix("TEAM_").rename(columns={"TEAM_TEAM_ID": "TEAM_ID",
                                                      "TEAM_GAME_ID": "GAME_ID"})
        own = own.drop(columns=["TEAM_TEAM_ABBREVIATION"])
        opp = tadv.add_prefix("OPP_").rename(columns={"OPP_GAME_ID": "GAME_ID",
                                                     "OPP_TEAM_ABBREVIATION": "OPP"})
        opp = opp.drop(columns=["OPP_TEAM_ID"])
        df = df.merge(own, on=["TEAM_ID", "GAME_ID"], how="left")
        df = df.merge(opp, on=["GAME_ID", "OPP"], how="left")

    df["STARTER_KNOWN"] = df["STARTER"].notna().astype(int)
    df["PDK"] = dunkest_score(df)

    front = ["SEASON", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM",
             "OPP", "HOME", "WL", "STARTER", "MIN", "PDK"]
    df = df[front + [c for c in df.columns if c not in front]]
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df.to_sql("player_games", con, if_exists="replace", index=False)
    con.commit()

    # controlli di coerenza
    print(f"  player_games: {len(df)} righe, {df['PLAYER_ID'].nunique()} giocatori")
    summary = df.groupby("SEASON").agg(righe=("PDK", "size"), partite=("GAME_ID", "nunique"),
                                       titolari_noti=("STARTER_KNOWN", "mean"),
                                       titolari=("STARTER", "sum"),
                                       pdk_medio=("PDK", "mean"))
    summary["titolari_per_squadra"] = summary["titolari"] / (2 * summary["partite"])  # atteso: 5
    print(summary.drop(columns="titolari").round(3).to_string())
    last = df["SEASON"].max()
    top = (df[df["SEASON"] == last].groupby("PLAYER_NAME")
           .agg(gp=("PDK", "size"), pdk=("PDK", "mean"), min=("MIN", "mean"))
           .query("gp >= 5").sort_values("pdk", ascending=False).head(10))
    print(f"\n  top 10 PDK medio {last}:\n{top.round(1).to_string()}")

    if export_csv:
        df.to_csv(export_csv, index=False)
        print(f"  esportato {export_csv}")
    return df
