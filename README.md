# Dunkest NBA

Previsione del punteggio [Dunkest](https://www.dunkest.com) (PDK) dei giocatori NBA, partita per partita, con modelli di machine learning allenati sulle ultime stagioni.

Il progetto:
1. scarica i dati NBA (statistiche partita per partita, boxscore, assenze, calendario) in un database SQLite locale;
2. ricalcola il PDK di ogni prestazione con il regolamento Dunkest;
3. allena modelli gradient boosting che stimano PDK, minuti, pavimento (P10) e tetto (P90) di ogni giocatore;
4. produce le previsioni per le partite di un periodo, ad esempio una giornata Dunkest.

## Installazione

Richiede Python 3.10 o superiore.

```bash
git clone https://github.com/<utente>/dunkest-nba.git
cd dunkest-nba
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Primo avvio

```bash
python main.py update --all     # stagioni dal 2022-23 a oggi (~1 ora, soprattutto boxscore)
python main.py train            # validazione + modelli finali (~1-2 minuti)
```

`update` si può interrompere (Ctrl-C) e rilanciare: riprende da dove si era fermato.
Se hai già un `dunkest.db`, copialo in `data/` e salta il primo comando.

### Durante la stagione

```bash
python main.py update                                         # aggiorna la stagione in corso
python main.py predict --from 2026-10-21 --to 2026-10-26      # previsioni per il periodo
python main.py predict --from 2026-10-21 --to 2026-10-26 --out "Tatum, Haliburton"
python main.py train                                          # riallena ogni 1-2 settimane
```

`--out` segna dei giocatori come assenti: vengono esclusi e il loro spazio passa ai compagni tramite le feature sugli assenti. I nomi si possono scrivere senza accenti e anche solo in parte (`"jokic"`).

Ogni comando ha l'help: `python main.py predict -h`.

### Output

`predict` scrive due file in `output/`.

**`predizioni_<da>_<a>.csv`**: una riga per giocatore, ordinata per PDK totale nel periodo.

| Colonna | Significato |
|---|---|
| `PARTITE`, `AVVERSARI` | partite nel periodo e avversari (`vs` = in casa, `@` = in trasferta) |
| `MIN_MEDI` | minuti attesi |
| `PDK_MEDIO`, `PDK_TOTALE` | PDK atteso per partita e totale nel periodo |
| `P10_MEDIO`, `P90_MEDIO` | pavimento e tetto: il PDK reale cade tra i due nell'80% dei casi |
| `NOTE` | `pochi dati` (rookie o quasi), `fermo da tempo: verifica` (probabile infortunio lungo) |

**`predizioni_partite_<da>_<a>.csv`**: le stesse informazioni, una riga per giocatore e partita.

Le previsioni presuppongono che il giocatore scenda in campo.

## Dati

Fonti: [stats.nba.com](https://www.nba.com/stats) tramite [`nba_api`](https://github.com/swar/nba_api), e il CDN pubblico di nba.com per boxscore e calendario. Solo regular season.

| Tabella | Contenuto |
|---|---|
| `raw_player_base`, `raw_player_adv` | statistiche classiche e avanzate di ogni giocatore in ogni partita |
| `raw_team_base`, `raw_team_adv` | lo stesso a livello di squadra |
| `box_players` | titolari, attivi e inattivi con il motivo dell'assenza |
| `players` | anagrafica, ruolo, draft, squadra e stato del roster attuali |
| `schedule` | calendario della stagione in corso |
| `player_games` | tabella finale: una riga per giocatore-partita con il PDK ricalcolato |

### Punteggio Dunkest

Implementato in `dunkest/scoring.py` dal regolamento ufficiale e verificato su 24.000 punteggi reali della stagione 2023-24 (corrispondenza 99.75%):

| Azione | Punti | Azione | Punti |
|---|---|---|---|
| punto | +1 | palla persa | −1.5 |
| rimbalzo difensivo | +1 | stoppata subita | −0.5 |
| rimbalzo offensivo | +1.25 | tiro sbagliato (campo o libero) | −1 |
| assist, recupero, stoppata | +1.5 | quintetto base | +1 |
| 3 / 4 / 5+ triple segnate | +3 / +4 / +5 | doppia / tripla / quadrupla doppia | +5 / +10 / +50 |
| vittoria di squadra | +5% del valore assoluto | capitano | ×1.5 (non applicato qui) |

## Modello

### Feature

44 feature per ogni coppia giocatore-partita, calcolate **solo con informazioni precedenti alla partita**. Un test automatico verifica che cambiare il risultato di una partita non modifichi le sue feature.

- **Forma:** medie mobili esponenziali di PDK, minuti, titolarità, PDK al minuto, usage e tiri (emivita 3, 10 e 30 partite). Proseguono tra una stagione e l'altra, quindi la forma vecchia pesa sempre meno man mano che arrivano partite nuove.
- **Contesto:** riposo, back-to-back, casa o trasferta, esperienza, cambio di squadra, ruolo, scelta al draft.
- **Avversario:** PDK concessi dalla difesa, in totale e per ruolo (G/F/C), e ritmo di gioco delle due squadre.
- **Assenti:** minuti e PDK dei compagni e degli avversari assenti per motivi noti prima della partita (infortuni, riposo, sospensioni), separando le assenze recenti da quelle lunghe e indicando quanti minuti mancano nel ruolo del giocatore.

### Modelli

Cinque `HistGradientBoostingRegressor` di scikit-learn:

| Modello | Target |
|---|---|
| `direct` | PDK |
| `min` | minuti |
| `rate` | PDK al minuto (righe con almeno 3 minuti, pesate per i minuti) |
| `q10`, `q90` | 10° e 90° percentile del PDK (loss quantilica) |

Previsione = (`direct` + `min` × `rate`) / 2.

### Validazione

Si allena sulle stagioni precedenti e si testa su una stagione intera mai vista, tenendo il modello fisso per tutta la stagione. Il confronto è con la media mobile esponenziale del PDK sulle ultime 10 partite. *Spearman/giorno* misura quanto bene il modello ordina i giocatori che giocano lo stesso giorno.

| Stagione di test | Metodo | MAE | R² | Spearman/giorno |
|---|---|---|---|---|
| 2024-25 | media mobile | 7.56 | 0.439 | 0.648 |
| 2024-25 | **modello** | **7.36** | **0.472** | **0.673** |
| 2025-26 | media mobile | 7.52 | 0.412 | 0.637 |
| 2025-26 | **modello** | **7.28** | **0.454** | **0.669** |

Errore sui minuti: 4.7 contro 5.25 della media mobile. L'intervallo P10-P90 contiene il valore reale nell'80% dei casi.

## Limiti noti

- **Assenti:** in allenamento il modello conosce gli assenti reali di ogni partita, in previsione solo quelli passati con `--out`.
- **Inizio stagione:** le stelle partono in media 2-4 punti sotto la forma di fine stagione precedente e il modello ne tiene conto. Sopra i 50 PDK di media (livello Jokić) la previsione resta un po' prudente.
- **Rientri da infortuni lunghi:** la previsione usa la forma precedente allo stop (vedi la colonna `NOTE`).
- **Scelte tecniche:** le esclusioni decise dall'allenatore a partita iniziata non sono prevedibili.

## Roadmap

- [ ] Injury report ufficiale NBA, per compilare `--out` in automatico
- [ ] Ottimizzatore della formazione (programmazione lineare intera) con crediti e ruoli Dunkest
- [ ] Backtest sulle formazioni
- [ ] Quote delle scommesse (spread e totale) come feature

## Struttura

```
main.py                 riga di comando: update, train, predict
dunkest/
  config.py             percorsi e stagioni
  db.py                 accesso al database SQLite
  download.py           download da stats.nba.com e cdn.nba.com
  dataset.py            costruzione di player_games
  scoring.py            punteggio Dunkest
  features.py           feature senza leakage
  model.py              modelli, metriche, salvataggio
  train.py              validazione e allenamento
  predict.py            previsioni e CSV
tests/                  test del punteggio e anti-leakage (pytest)
data/ models/ output/   generati, esclusi da git
```

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

## Note

Progetto personale non affiliato a NBA o Dunkest. I dati provengono da endpoint pubblici di nba.com: lo script fa poche richieste, con pause tra l'una e l'altra.
