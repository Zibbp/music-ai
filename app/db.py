import os
import yaml
import psycopg
from psycopg.rows import dict_row

def _load_cfg():
    cfg_path = os.getenv("MUSICAI_CONFIG", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_dsn = os.getenv("MUSICAI_DB_DSN")
    if db_dsn:
        cfg["db_dsn"] = db_dsn
    device = os.getenv("MUSICAI_DEVICE")
    if device:
        cfg["device"] = device
    return cfg

def init_db():
    cfg = _load_cfg()
    dsn = cfg["db_dsn"]
    ddl = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS tracks (
      id           BIGSERIAL PRIMARY KEY,
      path         TEXT UNIQUE NOT NULL,
      sr           INT NOT NULL,
      duration_s   REAL,
      bpm          REAL,
      rms          REAL,
      genre        TEXT,
      genre_score  REAL,
      genre_margin REAL,
      mood         TEXT,
      mood_score   REAL,
      mood_margin  REAL,
      family      TEXT,
      family_score REAL,
      family_margin REAL,
      family_confident BOOLEAN,
      genre_confident  BOOLEAN,
      mood_confident   BOOLEAN,
      family_topk JSONB,
      genre_topk JSONB,
      mood_topk JSONB,
      ingest_sig TEXT,
      created_at   TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS embeddings (
      track_id BIGINT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
      emb      vector(512) NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tag_stats (
      genre TEXT NOT NULL,
      mood  TEXT NOT NULL,
      bpm_p_low  REAL,
      bpm_p_high REAL,
      n BIGINT NOT NULL,
      PRIMARY KEY (genre, mood)
    );

    CREATE TABLE IF NOT EXISTS mix_history (
      id BIGSERIAL PRIMARY KEY,
      period TEXT NOT NULL,
      genre TEXT NOT NULL,
      period_start DATE NOT NULL,
      seed TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS mix_items (
      mix_id BIGINT REFERENCES mix_history(id) ON DELETE CASCADE,
      path TEXT NOT NULL,
      PRIMARY KEY (mix_id, path)
    );

    CREATE INDEX IF NOT EXISTS mix_history_period_genre_start
      ON mix_history(period, genre, period_start);
    CREATE INDEX IF NOT EXISTS mix_items_path
      ON mix_items(path);

    -- HNSW cosine index (fast ANN)
    CREATE INDEX IF NOT EXISTS embeddings_hnsw_cos
      ON embeddings USING hnsw (emb vector_cosine_ops);

    -- Backfill columns for existing DBs
    ALTER TABLE tracks ADD COLUMN IF NOT EXISTS family_topk JSONB;
    ALTER TABLE tracks ADD COLUMN IF NOT EXISTS genre_topk JSONB;
    ALTER TABLE tracks ADD COLUMN IF NOT EXISTS mood_topk JSONB;
    ALTER TABLE tracks ADD COLUMN IF NOT EXISTS ingest_sig TEXT;
    """

    with psycopg.connect(dsn) as conn:
        conn.execute(ddl)
        conn.commit()

    print("DB initialized.")
