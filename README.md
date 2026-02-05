# music-ai

Generate music mixes from audio embeddings, store metadata in Postgres + pgvector, and optionally sync playlists to Navidrome via Subsonic.

This project is pretty heavily vibe coded as I don't know a lot about running models and embedding.

When I first started running the embedding with `MuQ-MuLan-large`, the family/genre tagging was not very great. Luckily I had a few thousand tracks that had good genre tagging that I trained a supervised genre classifier on. This classifier resolve a lot of the bad tagging from the model.

My goal with this project is to tag/embed all of my local music tracks then provide a way to do the following:

- Generate instant mixes from a seed track
- Generate daily/weekly playlists of many genres/families

Using the custon genre classifier I can ensure the genres/families is how I want them to be.

The Navidrome/Subsonic mix playlist logic assumes access to a copy of the navidrome.db sqlite file to fetch track IDs to use in Subsonic API queries.

## Requirements

- Python 3.12
- Postgres with `pgvector` enabled
- FFmpeg and libsndfile (for audio decoding)
- Intel XPU users: host drivers installed and `/dev/dri` available

## Local Dev Setup (uv)

Install uv:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the venv and install deps:
```
uv sync --frozen
```

Run project entrypoints with `uv run`:
```
uv run musicai-mix --help
```

## Configuration

Primary config is `config.yaml`. You can override key values with env vars:

- `MUSICAI_CONFIG`: path to config file (default `config.yaml`)
- `MUSICAI_DB_DSN`: overrides `db_dsn`
- `MUSICAI_DEVICE`: overrides `device` (`cpu` or `xpu`)

Important config fields:

- `db_dsn`: Postgres DSN for tracks and embeddings
- `device`: `cpu` or `xpu`
- `music_path_prefix`: local root for resolving seed paths
- `subsonic_url`, `subsonic_user`, `subsonic_password`: Navidrome/Subsonic
- `subsonic_sqlite_db_path`: Navidrome sqlite DB for fast ID mapping
- `genre_clf_path`: optional supervised genre classifier
- `genre_clf_min_score`: minimum score to override prompt model

## Tag Bank

`app/tag_bank.py` defines the label sets and prompts used by the zero-shot tagger:

- Families (broad buckets)
- Genres (per-family subgenres and a global genre bank)
- Moods

If you change labels or prompts, re-ingest to recompute tags, and expect mixes to shift because tag filters depend on these labels.

## Supervised Genre Classifier

The project supports an optional supervised genre classifier that can override the prompt-based genre when confidence is high enough.

Train the classifier from a labeled CSV:
```
uv run musicai-train-genre --csv path-title-genre.csv --out models/genre_clf.joblib
```

Notes:
- The CSV should include a filepath, title, and one or more genre labels per row. See `path-title-genre.csv` for the expected format.
  - I pulled this directly from the Navidrome database by querying media_file that had a genre(s) associated with it.
- Set `genre_clf_path` and `genre_clf_min_score` in `config.yaml` to enable it during ingest.

## Database

Initialize tables and pgvector extension:
```
uv run musicai-init
```

Example queries:
```sql
select distinct genre from tracks;
select distinct family from tracks;
```

## Ingest (embeddings + tags)

Index a music library and compute embeddings:
```
uv run musicai-ingest --root "/path/to/music"
```

Notes:
- This writes `tracks` and `embeddings` in Postgres.
- Use `skip_if_ingested` in `config.yaml` to avoid reprocessing.

## Mixes (local output)

Mix from seed:
```
uv run musicai-mix --seed "track" --k 20
```

Daily mix:
```
uv run musicai-mix-schedule --genre dnb --period daily --k 50
```

Weekly mix:
```
uv run musicai-mix-schedule --genre techno --period weekly --k 50
```

Specific date:
```
uv run musicai-mix-schedule --genre dnb --period daily --date 2026-02-02 --k 50
```

Don’t persist history:
```
uv run musicai-mix-schedule --genre dnb --period daily --dry-run
```

Blend multiple genres:
```
uv run musicai-mix-schedule --genres "dnb,techno,house" --period weekly --k 50
```

Family mix:
```
uv run musicai-mix-schedule --family electronic --period weekly --k 50
```

Blend families:
```
uv run musicai-mix-schedule --families "electronic,hiphop_rnb" --period daily --k 40 --even-split
```

Blend families with ratios:
```
uv run musicai-mix-schedule --families "electronic,hiphop_rnb" --ratios "electronic=0.7,hiphop_rnb=0.3" --period weekly --k 50
```

## Navidrome / Subsonic Playlist Sync

Create or update a Navidrome playlist from a mix:
```
uv run musicai-navidrome --mode mix --seed "track" --k 75 --playlist-name "Mix - Test"
```

Schedule mode (uses tag targets):
```
uv run musicai-navidrome --mode schedule --genre dnb --period daily --k 50
```

Paths mode (explicit list):
```
uv run musicai-navidrome --mode paths --paths-file /path/to/paths.txt
```

SQLite ID mapping is always enabled when `subsonic_sqlite_db_path` exists.

## Docker

CPU:
```
docker compose -f compose.cpu.yml build
docker compose -f compose.cpu.yml run --rm musicai musicai-navidrome --mode mix --seed "track" --k 20
```

Intel XPU:
```
docker compose -f compose.xpu.yml build
docker compose -f compose.xpu.yml run --rm musicai musicai-navidrome --mode mix --seed "track" --k 20
```

Notes:
- Compose sets `MUSICAI_DEVICE` and `MUSICAI_DB_DSN`.
- Ensure `/path/to/music` and `./navidrome.db` exist on the host.

## Other Entry Points

All commands should be prefixed with `uv run`:

- `musicai-init`
- `musicai-ingest`
- `musicai-learn-stats`
- `musicai-recommend`
- `musicai-discover`
- `musicai-sweep`
- `musicai-train-genre`
- `musicai-compare-genre`
- `musicai-mix`
- `musicai-mix-schedule`
- `musicai-prefetch-mulan`
- `musicai-navidrome`
