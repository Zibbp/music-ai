# app/ingest.py

import argparse
import os
import time
import hashlib
import json

import psycopg
from tqdm import tqdm
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

from .muq_embed import MuQEngine, load_cfg
from .tag_bank import (
    FAMILY_LABELS, FAMILY_PROMPTS,
    MOOD_LABELS, MOOD_PROMPTS,
    GENRE_ALL_LABELS, GENRE_ALL_PROMPTS,
    STAGE2,
)

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aiff", ".aif"}


def iter_audio_files(root):
    for base, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in AUDIO_EXTS:
                yield os.path.join(base, fn)


def batched(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def upsert_track(conn, path, sr, duration_s, bpm, rms, tags, emb_vec, ingest_sig):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tracks(
              path, sr, duration_s, bpm, rms,
              family, family_score, family_margin, family_confident,
              genre, genre_score, genre_margin, genre_confident,
              mood, mood_score, mood_margin, mood_confident,
              family_topk, genre_topk, mood_topk,
              ingest_sig
            )
            VALUES (
              %s,%s,%s,%s,%s,
              %s,%s,%s,%s,
              %s,%s,%s,%s,
              %s,%s,%s,%s,
              %s,%s,%s,
              %s
            )
            ON CONFLICT (path) DO UPDATE SET
              sr=EXCLUDED.sr,
              duration_s=EXCLUDED.duration_s,
              bpm=EXCLUDED.bpm,
              rms=EXCLUDED.rms,

              family=EXCLUDED.family,
              family_score=EXCLUDED.family_score,
              family_margin=EXCLUDED.family_margin,
              family_confident=EXCLUDED.family_confident,

              genre=EXCLUDED.genre,
              genre_score=EXCLUDED.genre_score,
              genre_margin=EXCLUDED.genre_margin,
              genre_confident=EXCLUDED.genre_confident,

              mood=EXCLUDED.mood,
              mood_score=EXCLUDED.mood_score,
              mood_margin=EXCLUDED.mood_margin,
              mood_confident=EXCLUDED.mood_confident,

              family_topk=EXCLUDED.family_topk,
              genre_topk=EXCLUDED.genre_topk,
              mood_topk=EXCLUDED.mood_topk,
              ingest_sig=EXCLUDED.ingest_sig
            RETURNING id
            """,
            (
                path,
                sr,
                duration_s,
                bpm,
                rms,
                tags.get("family"),
                tags.get("family_score"),
                tags.get("family_margin"),
                tags.get("family_confident"),
                tags.get("genre"),
                tags.get("genre_score"),
                tags.get("genre_margin"),
                tags.get("genre_confident"),
                tags.get("mood"),
                tags.get("mood_score"),
                tags.get("mood_margin"),
                tags.get("mood_confident"),
                Json(tags.get("family_topk")),
                Json(tags.get("genre_topk")),
                Json(tags.get("mood_topk")),
                ingest_sig,
            ),
        )
        track_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO embeddings(track_id, emb)
            VALUES (%s, %s)
            ON CONFLICT (track_id) DO UPDATE SET emb=EXCLUDED.emb
            """,
            (track_id, emb_vec),
        )


def rms_only_from_wav(wav_cpu) -> float:
    import torch

    return float(torch.sqrt(torch.mean(wav_cpu.float().square())).item())


def bpm_from_wav_fast(wav_cpu, sr: int, max_seconds: int = 60, down_sr: int = 12000) -> float:
    import librosa

    n = min(wav_cpu.numel(), sr * int(max_seconds))
    y = wav_cpu[:n].cpu().numpy()

    if down_sr and int(down_sr) < sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=int(down_sr))
        sr = int(down_sr)

    return float(librosa.beat.tempo(y=y, sr=sr)[0])


def _ingest_signature(cfg: dict) -> str:
    payload = {
        "model": "OpenMuQ/MuQ-MuLan-large",
        "sr": cfg.get("sr"),
        "chunk_s": cfg.get("chunk_s"),
        "hop_s": cfg.get("hop_s"),
        "active_keep": cfg.get("active_keep"),
        "family_threshold": cfg.get("family_threshold"),
        "family_margin": cfg.get("family_margin"),
        "genre_threshold": cfg.get("genre_threshold"),
        "genre_margin": cfg.get("genre_margin"),
        "mood_threshold": cfg.get("mood_threshold"),
        "mood_margin": cfg.get("mood_margin"),
        "family_allow_fallback": cfg.get("family_allow_fallback"),
        "genre_allow_fallback": cfg.get("genre_allow_fallback"),
        "genre_global": cfg.get("genre_global"),
        "family_from_genre": cfg.get("family_from_genre"),
        "genre_clf_path": cfg.get("genre_clf_path"),
        "genre_clf_min_score": cfg.get("genre_clf_min_score"),
        "tag_bank": {
            "family_labels": FAMILY_LABELS,
            "family_prompts": FAMILY_PROMPTS,
            "mood_labels": MOOD_LABELS,
            "mood_prompts": MOOD_PROMPTS,
            "genre_all_labels": GENRE_ALL_LABELS,
            "genre_all_prompts": GENRE_ALL_PROMPTS,
            "stage2": STAGE2,
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Music library root folder")
    args = ap.parse_args()

    cfg = load_cfg()
    sr = int(cfg["sr"])
    engine = MuQEngine(
        cfg["device"],
        model_path=cfg.get("mulan_model_path"),
        offline=bool(cfg.get("mulan_offline", False)),
    )
    ingest_sig = _ingest_signature(cfg)

    # Perf knobs
    batch_size = int(cfg.get("batch_size", 16))                # tracks per GPU batch
    microbatch_chunks = int(cfg.get("microbatch_chunks", 64))  # chunks per forward pass
    active_keep = int(cfg.get("active_keep", 4))
    commit_every = int(cfg.get("commit_every", 200))
    load_workers = int(cfg.get("load_workers", 0))
    timing_breakdown = bool(cfg.get("timing_breakdown", True))
    skip_if_ingested = bool(cfg.get("skip_if_ingested", True))

    # Optional BPM
    compute_bpm = bool(cfg.get("compute_bpm", False))
    bpm_max_seconds = int(cfg.get("bpm_max_seconds", 60))
    bpm_down_sr = int(cfg.get("bpm_down_sr", 12000))

    chunk_s = float(cfg["chunk_s"])
    hop_s = float(cfg["hop_s"])

    with psycopg.connect(cfg["db_dsn"]) as conn:
        register_vector(conn)
        conn.autocommit = False

        pending = 0

        # Throughput stats
        total_ok = 0
        t0 = time.perf_counter()
        last_print_t = t0
        last_print_ok = 0
        print_every_s = float(cfg.get("tps_print_every_s", 5.0))

        for batch_paths in tqdm(batched(iter_audio_files(args.root), batch_size), desc="Ingest"):
            try:
                if skip_if_ingested:
                    rows = conn.execute(
                        "SELECT path, ingest_sig FROM tracks WHERE path = ANY(%s)",
                        (batch_paths,),
                    ).fetchall()
                    done = {r[0] for r in rows if r[1] == ingest_sig}
                    if done:
                        batch_paths = [p for p in batch_paths if p not in done]
                        if not batch_paths:
                            continue
                if timing_breakdown:
                    batch_results, timings = engine.embed_tracks(
                        batch_paths,
                        sr=sr,
                        chunk_s=chunk_s,
                        hop_s=hop_s,
                        active_keep=active_keep,
                        microbatch_chunks=microbatch_chunks,
                        num_workers=load_workers,
                        return_timings=True,
                    )
                else:
                    batch_results = engine.embed_tracks(
                        batch_paths,
                        sr=sr,
                        chunk_s=chunk_s,
                        hop_s=hop_s,
                        active_keep=active_keep,
                        microbatch_chunks=microbatch_chunks,
                        num_workers=load_workers,
                    )
                    timings = None

                post_t0 = time.perf_counter()
                for path, (pooled, dur, wav_cpu) in zip(batch_paths, batch_results):
                    tags = engine.classify(pooled, cfg)

                    rms = rms_only_from_wav(wav_cpu)
                    if compute_bpm:
                        bpm = bpm_from_wav_fast(
                            wav_cpu,
                            sr=sr,
                            max_seconds=bpm_max_seconds,
                            down_sr=bpm_down_sr,
                        )
                    else:
                        bpm = None

                    emb_vec = pooled.squeeze(0).detach().cpu().numpy()
                    upsert_track(conn, path, sr, dur, bpm, rms, tags, emb_vec, ingest_sig)

                    pending += 1
                    total_ok += 1

                    if pending >= commit_every:
                        conn.commit()
                        pending = 0

                    # Print tracks/sec periodically (doesn't interfere with tqdm)
                    now = time.perf_counter()
                    if (now - last_print_t) >= print_every_s:
                        interval = now - last_print_t
                        done = total_ok - last_print_ok
                        tps = (done / interval) if interval > 0 else 0.0

                        total_elapsed = now - t0
                        total_tps = (total_ok / total_elapsed) if total_elapsed > 0 else 0.0

                        msg = (
                            f"Throughput: {tps:.2f} tracks/s (last {interval:.1f}s), "
                            f"{total_tps:.2f} tracks/s overall, {total_ok} ok"
                        )
                        if timings is not None:
                            post_s = time.perf_counter() - post_t0
                            msg += (
                                f" | timing: load {timings['load_s']:.2f}s, "
                                f"pack {timings['pack_s']:.2f}s, infer {timings['infer_s']:.2f}s, "
                                f"pool {timings['pool_s']:.2f}s, post {post_s:.2f}s, "
                                f"chunks {timings['chunks']}"
                            )
                        tqdm.write(msg)

                        last_print_t = now
                        last_print_ok = total_ok

            except Exception as e:
                conn.rollback()
                pending = 0
                tqdm.write(f"FAIL batch starting with {batch_paths[0]}: {e}")

        if pending:
            conn.commit()

        # Final stats
        t1 = time.perf_counter()
        total_elapsed = t1 - t0
        total_tps = (total_ok / total_elapsed) if total_elapsed > 0 else 0.0
        tqdm.write(f"Done: {total_ok} tracks in {total_elapsed:.1f}s => {total_tps:.2f} tracks/s")


if __name__ == "__main__":
    main()
