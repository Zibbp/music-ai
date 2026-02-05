import argparse
import os
from typing import List

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .muq_embed import MuQEngine, load_cfg


def _parse_seeds(seed: str | None, seeds: str | None) -> List[str]:
    paths = []
    if seed:
        paths.append(seed)
    if seeds:
        for p in seeds.split(","):
            p = p.strip()
            if p:
                paths.append(p)
    if not paths:
        raise SystemExit("Provide --seed or --seeds")
    return paths


def _resolve_paths(cfg: dict, paths: List[str]) -> List[str]:
    root = cfg.get("music_path_prefix") or cfg.get("subsonic_path_strip_prefix")
    out = []
    for p in paths:
        if os.path.exists(p):
            out.append(p)
            continue
        if root:
            cand = os.path.join(root, p.lstrip("/\\"))
            if os.path.exists(cand):
                out.append(cand)
                continue
        out.append(p)
    return out


def _avg_embed(engine: MuQEngine, paths: List[str], sr: int, chunk_s: float, hop_s: float, active_keep: int):
    embs = []
    for p in paths:
        emb, _dur, _wav = engine.embed_track(
            p,
            sr=sr,
            chunk_s=chunk_s,
            hop_s=hop_s,
            active_keep=active_keep,
        )
        embs.append(emb.squeeze(0).detach().cpu().numpy())
    X = np.stack(embs, axis=0)
    v = X.mean(axis=0, keepdims=True)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v.squeeze(0)


def get_mix_rows(
    cfg: dict,
    seed_paths: List[str],
    k: int = 50,
    same_genre: bool = False,
    same_family: bool = False,
    same_mood: bool = False,
    bpm_window: float | None = None,
    exclude_seeds: bool = False,
):
    sr = int(cfg["sr"])
    engine = MuQEngine(
        cfg["device"],
        model_path=cfg.get("mulan_model_path"),
        offline=bool(cfg.get("mulan_offline", False)),
    )

    seed_vec = _avg_embed(
        engine,
        seed_paths,
        sr=sr,
        chunk_s=float(cfg["chunk_s"]),
        hop_s=float(cfg["hop_s"]),
        active_keep=int(cfg.get("active_keep", 4)),
    )

    with psycopg.connect(cfg["db_dsn"]) as conn:
        register_vector(conn)
        conn.row_factory = psycopg.rows.dict_row
        # Ensure ANN search can return at least k results (HNSW defaults can be too low).
        # Guard against missing settings without aborting the transaction.
        try:
            desired_ef = int(cfg.get("hnsw_ef_search", 0))
            if desired_ef < int(k):
                desired_ef = int(k)
            has_hnsw = conn.execute(
                "SELECT 1 FROM pg_settings WHERE name = 'hnsw.ef_search'"
            ).fetchone()
            if has_hnsw:
                conn.execute("SET LOCAL hnsw.ef_search = %s", (desired_ef,))
        except Exception:
            conn.rollback()
        # Optional: if using IVFFlat, allow configuring probes.
        try:
            probes = int(cfg.get("ivfflat_probes", 0))
            if probes > 0:
                has_ivf = conn.execute(
                    "SELECT 1 FROM pg_settings WHERE name = 'ivfflat.probes'"
                ).fetchone()
                if has_ivf:
                    conn.execute("SET LOCAL ivfflat.probes = %s", (probes,))
        except Exception:
            conn.rollback()

        seed_rows = conn.execute(
            "SELECT path, genre, family, mood, bpm FROM tracks WHERE path = ANY(%s)",
            (seed_paths,),
        ).fetchall()
        seed_genre = seed_rows[0]["genre"] if seed_rows else None
        seed_family = seed_rows[0]["family"] if seed_rows else None
        seed_mood = seed_rows[0]["mood"] if seed_rows else None
        seed_bpm = seed_rows[0]["bpm"] if seed_rows else None

        where = []
        params = {"seed": seed_vec, "k": int(k)}

        if same_genre and seed_genre:
            where.append("t.genre = %(genre)s")
            params["genre"] = seed_genre
        if same_family and seed_family:
            where.append("t.family = %(family)s")
            params["family"] = seed_family
        if same_mood and seed_mood:
            where.append("t.mood = %(mood)s")
            params["mood"] = seed_mood
        if bpm_window and seed_bpm:
            where.append("t.bpm BETWEEN %(bpm_lo)s AND %(bpm_hi)s")
            params["bpm_lo"] = float(seed_bpm) - float(bpm_window)
            params["bpm_hi"] = float(seed_bpm) + float(bpm_window)
        if exclude_seeds:
            where.append("t.path <> ALL(%(seed_paths)s)")
            params["seed_paths"] = seed_paths

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
        SELECT
          t.path, t.genre, t.family, t.mood, t.bpm,
          1 - (e.emb <=> %(seed)s) AS sim
        FROM embeddings e
        JOIN tracks t ON t.id = e.track_id
        {where_sql}
        ORDER BY e.emb <=> %(seed)s
        LIMIT %(k)s
        """
        rows = conn.execute(sql, params).fetchall()
        # If ANN search returns fewer than requested, fall back to a full scan to
        # avoid HNSW ef_search limits silently capping results.
        if len(rows) < int(k):
            try:
                conn.execute("SET LOCAL enable_indexscan = off")
                conn.execute("SET LOCAL enable_bitmapscan = off")
                conn.execute("SET LOCAL enable_seqscan = on")
            except Exception:
                conn.rollback()
            rows = conn.execute(sql, params).fetchall()

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", help="Seed track path")
    ap.add_argument("--seeds", help="Comma-separated seed paths")
    ap.add_argument("--k", type=int, default=50, help="Number of results")
    ap.add_argument("--same-genre", action="store_true", help="Restrict to the seed genre")
    ap.add_argument("--same-family", action="store_true", help="Restrict to the seed family")
    ap.add_argument("--same-mood", action="store_true", help="Restrict to the seed mood")
    ap.add_argument("--bpm-window", type=float, default=None, help="Restrict to seed bpm +/- window")
    ap.add_argument("--exclude-seeds", action="store_true", help="Exclude seed tracks from results")
    args = ap.parse_args()

    seed_paths = _parse_seeds(args.seed, args.seeds)
    cfg = load_cfg()
    seed_paths = _resolve_paths(cfg, seed_paths)
    rows = get_mix_rows(
        cfg,
        seed_paths,
        k=int(args.k),
        same_genre=bool(args.same_genre),
        same_family=bool(args.same_family),
        same_mood=bool(args.same_mood),
        bpm_window=args.bpm_window,
        exclude_seeds=bool(args.exclude_seeds),
    )

    for r in rows:
        bpm = r["bpm"]
        bpm_s = f"{bpm:.1f}" if isinstance(bpm, (int, float)) else "n/a"
        print(f"{r['sim']:.4f} | {r['genre'] or 'n/a'} | {r['mood'] or 'n/a'} | {bpm_s} | {r['path']}")


if __name__ == "__main__":
    main()
