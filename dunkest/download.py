"""
Download dei dati NBA nel database.

Fonti:
  - stats.nba.com (tramite nba_api): log partita per partita di giocatori e squadre, anagrafica.
    Ha un rate limit severo: poche chiamate per stagione, con pause e retry.
  - cdn.nba.com: boxscore (titolari, motivi di assenza) e calendario. Nessun rate limit noto.
"""

import random
import re
import time

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import (
    boxscoretraditionalv3,
    playergamelogs,
    playerindex,
    teamgamelogs,
)

from .config import SEASON_TYPE
from .db import save_season, table_exists

SLEEP = 1.0            # pausa base tra le richieste a stats.nba.com
TIMEOUT = 45
MAX_TRIES = 4
COOLDOWN_MIN = 15      # pausa se arrivano errori di fila (probabile blocco)
BOX_SLEEP_STATS = 3.0  # boxscore da stats.nba.com: lento apposta per non farsi bloccare
BOX_SLEEP_CDN = 0.25
CDN_MISS_LIMIT = 20    # dopo tanti errori CDN di fila per una stagione si passa a stats.nba.com
CDN_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/",
               "Origin": "https://www.nba.com"}
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"


# ============================================================================ utilità

def api_call(endpoint_cls, tries=MAX_TRIES, sleep=SLEEP, **kwargs):
    """Chiama un endpoint nba_api con pausa e retry esponenziale. Ritorna la lista di DataFrame."""
    for attempt in range(tries):
        try:
            time.sleep(sleep + random.random() * 0.5)
            return endpoint_cls(timeout=TIMEOUT, **kwargs).get_data_frames()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # timeout, 429, JSON vuoto...
            wait = 5 * 2 ** attempt
            print(f"    ! {type(e).__name__}: {str(e)[:100]} -> riprovo tra {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{endpoint_cls.__name__} {kwargs} fallito dopo {tries} tentativi")


def parse_minutes(x) -> float:
    """Minuti nei formati '34:12', 'PT34M12.00S' o vuoto -> float."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "":
        return 0.0
    s = str(x)
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60
    if ":" in s:
        mm, ss = s.split(":")[:2]
        return int(mm) + float(ss) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


# ============================================================================ stats.nba.com

def fetch_player_logs(con, season) -> int:
    print(f"  log giocatori {season}")
    base = api_call(playergamelogs.PlayerGameLogs, season_nullable=season,
                    season_type_nullable=SEASON_TYPE)[0]
    save_season(con, "raw_player_base", base, season)
    if base.empty:
        return 0
    adv = api_call(playergamelogs.PlayerGameLogs, season_nullable=season,
                   season_type_nullable=SEASON_TYPE,
                   measure_type_player_game_logs_nullable="Advanced")[0]
    save_season(con, "raw_player_adv", adv, season)
    return len(base)


def fetch_team_logs(con, season) -> None:
    print(f"  log squadre {season}")
    base = api_call(teamgamelogs.TeamGameLogs, season_nullable=season,
                    season_type_nullable=SEASON_TYPE)[0]
    save_season(con, "raw_team_base", base, season)
    adv = api_call(teamgamelogs.TeamGameLogs, season_nullable=season,
                   season_type_nullable=SEASON_TYPE,
                   measure_type_player_game_logs_nullable="Advanced")[0]
    save_season(con, "raw_team_adv", adv, season)


def fetch_players(con, season) -> None:
    """Anagrafica di tutti i giocatori, con squadra e stato del roster attuali (una chiamata)."""
    print("  anagrafica giocatori")
    try:
        df = api_call(playerindex.PlayerIndex, season=season, historical_nullable="1")[0]
        df.to_sql("players", con, if_exists="replace", index=False)
        con.commit()
        print(f"    players: {len(df)} righe")
    except Exception as e:
        print(f"    anagrafica non disponibile: {e}")


# ============================================================================ cdn.nba.com

def fetch_schedule(con) -> None:
    """Calendario di regular season della stagione in corso."""
    print("  calendario stagione in corso")
    try:
        r = requests.get(SCHEDULE_URL, headers=CDN_HEADERS, timeout=30)
        r.raise_for_status()
        ls = r.json()["leagueSchedule"]
    except Exception as e:
        print(f"    calendario non disponibile: {e}")
        return
    rows = []
    for gd in ls.get("gameDates", []):
        for g in gd.get("games", []):
            rows.append({
                "GAME_ID": g.get("gameId"),
                "GAME_DATE_UTC": g.get("gameDateTimeUTC"),
                "GAME_DATE_EST": g.get("gameDateEst"),
                "WEEK": g.get("weekNumber"),
                "LABEL": g.get("gameLabel"),
                "SUBLABEL": g.get("gameSubLabel"),
                "HOME_ID": g.get("homeTeam", {}).get("teamId"),
                "HOME": g.get("homeTeam", {}).get("teamTricode"),
                "AWAY_ID": g.get("awayTeam", {}).get("teamId"),
                "AWAY": g.get("awayTeam", {}).get("teamTricode"),
                "STATUS": g.get("gameStatus"),
            })
    df = pd.DataFrame(rows)
    df = df[df["GAME_ID"].astype(str).str.startswith("002")]   # 002... = regular season
    df["SEASON"] = ls.get("seasonYear", "?")
    df.to_sql("schedule", con, if_exists="replace", index=False)
    con.commit()
    print(f"    schedule: {len(df)} partite ({df['SEASON'].iloc[0] if len(df) else '-'})")


def box_from_cdn(gid) -> pd.DataFrame:
    """Boxscore dal CDN NBA: titolari, attivi/inattivi e motivo dell'assenza."""
    time.sleep(BOX_SLEEP_CDN)
    r = requests.get(BOXSCORE_URL.format(gid=gid), headers=CDN_HEADERS, timeout=20)
    r.raise_for_status()
    game = r.json()["game"]
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = game[side]
        for p in t.get("players", []):
            rows.append({
                "GAME_ID": gid, "TEAM_ID": t["teamId"], "TEAM": t["teamTricode"],
                "PLAYER_ID": p["personId"], "NAME_SHORT": p.get("nameI"),
                "STARTER": int(str(p.get("starter", "0")) == "1"),
                "STATUS": p.get("status"),
                "COMMENT": " ".join(x for x in [p.get("notPlayingReason"),
                                                 p.get("notPlayingDescription")] if x),
                "MIN_BOX": parse_minutes(p.get("statistics", {}).get("minutes")),
            })
    return pd.DataFrame(rows)


def box_from_stats(gid) -> pd.DataFrame:
    """Riserva: boxscore V3 da stats.nba.com (lento, rate limit severo)."""
    df = api_call(boxscoretraditionalv3.BoxScoreTraditionalV3, tries=2, sleep=BOX_SLEEP_STATS,
                  game_id=gid)[0]
    if df.empty:
        return df
    return pd.DataFrame({
        "GAME_ID": gid, "TEAM_ID": df["teamId"], "TEAM": df["teamTricode"],
        "PLAYER_ID": df["personId"], "NAME_SHORT": df["nameI"],
        "STARTER": (df["position"].fillna("").str.strip() != "").astype(int),
        "STATUS": np.where(df["comment"].fillna("") == "", "ACTIVE", "INACTIVE"),
        "COMMENT": df["comment"].fillna(""),
        "MIN_BOX": df["minutes"].map(parse_minutes),
    })


def fetch_boxscores(con, seasons) -> None:
    """Scarica i boxscore mancanti (riprendibile: le partite già salvate vengono saltate)."""
    if not table_exists(con, "raw_team_base"):
        return
    con.execute("CREATE TABLE IF NOT EXISTS box_fetched (GAME_ID TEXT PRIMARY KEY)")
    ph = ",".join("?" * len(seasons))
    games = pd.read_sql(
        f"SELECT DISTINCT GAME_ID, SEASON FROM raw_team_base WHERE SEASON IN ({ph}) "
        f"AND GAME_ID NOT IN (SELECT GAME_ID FROM box_fetched) ORDER BY GAME_ID",
        con, params=list(seasons))
    n = len(games)
    print(f"  boxscore da scaricare: {n}")
    t0, fails = time.time(), 0
    cdn_miss = {}
    src = {"cdn": 0, "stats": 0}
    for i, (gid, season) in enumerate(games.itertuples(index=False), 1):
        df = None
        if cdn_miss.get(season, 0) < CDN_MISS_LIMIT:
            try:
                df = box_from_cdn(gid)
                cdn_miss[season] = 0
                src["cdn"] += 1
            except Exception:
                cdn_miss[season] = cdn_miss.get(season, 0) + 1
                if cdn_miss[season] == CDN_MISS_LIMIT:
                    print(f"    CDN non disponibile per {season}: passo a stats.nba.com (lento)")
        if df is None:
            try:
                df = box_from_stats(gid)
                if df is not None and not df.empty:
                    src["stats"] += 1
            except Exception:
                df = None
        if df is None or df.empty:
            fails += 1
            if fails >= 3:
                print(f"    ! {fails} errori di fila: probabile blocco, pausa {COOLDOWN_MIN} min "
                      f"(puoi anche interrompere e riprendere più tardi)")
                time.sleep(COOLDOWN_MIN * 60)
                fails = 0
            continue
        fails = 0
        df["SEASON"] = season
        df.to_sql("box_players", con, if_exists="append", index=False)
        con.execute("INSERT OR IGNORE INTO box_fetched VALUES (?)", (str(gid),))
        con.commit()
        if i % 50 == 0 or i == n:
            el = time.time() - t0
            print(f"    {i}/{n}  ({el / 60:.1f} min, ETA {el / i * (n - i) / 60:.1f} min)  "
                  f"[cdn {src['cdn']}, stats {src['stats']}]")


# ============================================================================ orchestrazione

def update(con, seasons, skip_boxscores=False) -> None:
    """Scarica/aggiorna le stagioni indicate, anagrafica, calendario e boxscore mancanti."""
    for season in seasons:
        print(f"== {season}")
        if fetch_player_logs(con, season) == 0:
            continue   # stagione non ancora iniziata
        fetch_team_logs(con, season)
    fetch_players(con, seasons[-1])
    fetch_schedule(con)
    if not skip_boxscores:
        fetch_boxscores(con, seasons)
