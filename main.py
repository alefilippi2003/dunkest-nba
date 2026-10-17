#!/usr/bin/env python3
"""
Dunkest NBA - riga di comando.

    python main.py update --all                  primo avvio: tutte le stagioni dal 2022-23 (~1 ora)
    python main.py update                        aggiorna la stagione in corso (dopo ogni turno)
    python main.py train                         validazione + allenamento dei modelli finali
    python main.py predict --from 2026-10-21 --to 2026-10-26 [--out "Tatum, Haliburton"]

Opzioni comuni: --db PERCORSO (default data/dunkest.db). Dettagli: python main.py <comando> -h
"""

import argparse
import sys

import pandas as pd

from dunkest.config import (DB_PATH, FIRST_SEASON, MODEL_PATH, OUTPUT_DIR, season_of,
                            seasons_between)
from dunkest.db import connect


def cmd_update(args, con):
    from dunkest.dataset import build_player_games
    if not args.build_only:
        from dunkest.download import update   # import qui: train/predict non richiedono nba_api
        current = season_of(pd.Timestamp.today())
        if args.seasons:
            seasons = args.seasons
        elif args.all:
            seasons = seasons_between(FIRST_SEASON, current)
        else:
            seasons = [current]
        update(con, seasons, skip_boxscores=args.skip_boxscores)
    build_player_games(con, export_csv=args.csv)


def cmd_train(args, con):
    from dunkest.train import run_train
    run_train(con, args.model, do_validate=not args.no_validate)


def cmd_predict(args, con):
    from dunkest.predict import run_predict
    run_predict(con, args.model, args.output, args.d0, args.d1, args.out, top=args.top)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dunkest NBA: dati, modello e previsioni del PDK")
    ap.add_argument("--db", default=str(DB_PATH), help=f"database SQLite (default {DB_PATH})")
    sub = ap.add_subparsers(dest="command", required=True)

    up = sub.add_parser("update", help="scarica/aggiorna i dati NBA nel database")
    up.add_argument("--all", action="store_true", help=f"tutte le stagioni da {FIRST_SEASON} a oggi")
    up.add_argument("--seasons", nargs="+", help="stagioni specifiche, es. 2024-25 2025-26")
    up.add_argument("--skip-boxscores", action="store_true", help="non scarica i boxscore")
    up.add_argument("--build-only", action="store_true",
                    help="nessun download: ricostruisce solo la tabella player_games")
    up.add_argument("--csv", metavar="FILE", help="esporta player_games anche in CSV")
    up.set_defaults(func=cmd_update)

    tr = sub.add_parser("train", help="valida e allena i modelli")
    tr.add_argument("--model", default=str(MODEL_PATH), help=f"file dei modelli (default {MODEL_PATH})")
    tr.add_argument("--no-validate", action="store_true", help="salta la validazione (più veloce)")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="previsioni per le partite tra due date")
    pr.add_argument("--from", dest="d0", required=True, help="data iniziale YYYY-MM-DD")
    pr.add_argument("--to", dest="d1", required=True, help="data finale YYYY-MM-DD (inclusa)")
    pr.add_argument("--out", default="", help='giocatori assenti, es. "Tatum, Haliburton"')
    pr.add_argument("--model", default=str(MODEL_PATH))
    pr.add_argument("--output", default=str(OUTPUT_DIR), help=f"cartella dei CSV (default {OUTPUT_DIR})")
    pr.add_argument("--top", type=int, default=25, help="quanti giocatori stampare a schermo")
    pr.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    con = connect(args.db)
    try:
        args.func(args, con)
    except KeyboardInterrupt:
        print("\ninterrotto: rilancia lo stesso comando per riprendere")
        sys.exit(1)
    finally:
        con.close()


if __name__ == "__main__":
    main()
