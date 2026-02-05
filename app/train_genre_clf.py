import argparse
import csv
import json
import os
import hashlib

import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .muq_embed import MuQEngine, load_cfg


def _normalize_genre(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("&", "and")
    s = " ".join(s.split())
    return s


def _load_genre_map(path: str | None):
    if not path:
        return None, {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    mode = (data.get("mode") or "exact").lower()
    mapping = data.get("targets") if isinstance(data, dict) and "targets" in data else data
    mapping = mapping or {}

    norm_map = {}
    for target, vals in mapping.items():
        if isinstance(vals, str):
            vals = [vals]
        norm_vals = [_normalize_genre(v) for v in vals]
        norm_map[target] = norm_vals
    return mode, norm_map


def _map_genres(raw_genres, mode: str | None, mapping: dict) -> list[str]:
    if not mapping:
        return [_normalize_genre(g) for g in raw_genres if _normalize_genre(g)]

    out = set()
    for g in raw_genres:
        ng = _normalize_genre(g)
        if not ng:
            continue
        for target, vals in mapping.items():
            if mode == "contains":
                if any(v in ng for v in vals):
                    out.add(target)
            else:
                if any(v == ng for v in vals):
                    out.add(target)
    return sorted(out)


def _iter_csv_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _cache_path_for(path: str, cache_dir: str, cache_params: str) -> str:
    real_path = os.path.realpath(path)
    st = os.stat(real_path)
    key = f"{real_path}|{st.st_mtime_ns}|{st.st_size}|{cache_params}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{h}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="path-title-genre.csv", help="CSV with columns: path,title,genres")
    ap.add_argument("--root", default=".", help="Optional root to prefix relative paths")
    ap.add_argument("--map", dest="map_path", default=None, help="Optional YAML mapping of raw genres to targets")
    ap.add_argument("--out", default="models/genre_clf.joblib", help="Output model path")
    ap.add_argument("--min-per-class", type=int, default=10, help="Drop classes with < N examples")
    ap.add_argument("--val-size", type=float, default=0.2, help="Validation split fraction")
    ap.add_argument("--batch-size", type=int, default=32, help="Tracks per embed batch")
    ap.add_argument("--skip-unsupported", action="store_true", help="Skip files with unsupported extensions")
    ap.add_argument("--cache-dir", default="cache/embeddings", help="Directory for per-track embedding cache")
    ap.add_argument("--use-cache", action="store_true", help="Use on-disk embedding cache (resume-safe)")
    ap.add_argument("--print-files", action="store_true", help="Print each file successfully used")
    args = ap.parse_args()

    cfg = load_cfg()
    engine = MuQEngine(
        cfg["device"],
        model_path=cfg.get("mulan_model_path"),
        offline=bool(cfg.get("mulan_offline", False)),
    )

    mode, mapping = _load_genre_map(args.map_path)

    supported_exts = {".flac", ".wav", ".ogg", ".opus", ".m4a", ".aiff", ".aif"}
    paths = []
    labels = []
    for row in _iter_csv_rows(args.csv):
        path = row.get("path") or row.get("Path") or row.get("PATH")
        genres = row.get("genres") or row.get("genre") or row.get("Genres") or row.get("Genre")
        if not path or not genres:
            continue
        raw = [g.strip() for g in genres.split(",") if g.strip()]
        mapped = _map_genres(raw, mode, mapping)
        if not mapped:
            continue
        if not os.path.isabs(path):
            path = os.path.join(args.root, path)
        if args.skip_unsupported:
            ext = os.path.splitext(path)[1].lower()
            if ext not in supported_exts:
                continue
        paths.append(path)
        labels.append(mapped)

    if not paths:
        raise SystemExit("No labeled rows found. Check CSV columns and mapping.")

    # Filter rare classes
    counts = {}
    for labs in labels:
        for l in labs:
            counts[l] = counts.get(l, 0) + 1
    keep = {k for k, v in counts.items() if v >= int(args.min_per_class)}
    paths_f = []
    labels_f = []
    for p, labs in zip(paths, labels):
        kept = [l for l in labs if l in keep]
        if kept:
            paths_f.append(p)
            labels_f.append(kept)

    if not paths_f:
        raise SystemExit("All labels filtered by min-per-class.")

    # Embed tracks (skip unreadable files)
    embs = []
    labels_f2 = []
    total = len(paths_f)
    pbar = tqdm(total=total, desc="Embedding", unit="track")
    if args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
    cache_params = (
        f"{cfg.get('sr')}|{cfg.get('chunk_s')}|{cfg.get('hop_s')}|"
        f"{cfg.get('active_keep')}|{engine.mulan.__class__.__name__}"
    )
    cache_hits = 0
    cache_misses = 0
    cache_saved = 0
    for i in range(0, len(paths_f), int(args.batch_size)):
        batch = paths_f[i : i + int(args.batch_size)]
        batch_labels = labels_f[i : i + int(args.batch_size)]
        try:
            cached_embs = {}
            if args.use_cache:
                for path in batch:
                    try:
                        cache_path = _cache_path_for(path, args.cache_dir, cache_params)
                        if os.path.exists(cache_path):
                            cached_embs[path] = np.load(cache_path)
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    except Exception:
                        cache_misses += 1
                        pass

            to_process = [p for p in batch if p not in cached_embs]
            if to_process:
                results = engine.embed_tracks(
                    to_process,
                    sr=int(cfg["sr"]),
                    chunk_s=float(cfg["chunk_s"]),
                    hop_s=float(cfg["hop_s"]),
                    active_keep=int(cfg.get("active_keep", 4)),
                    microbatch_chunks=int(cfg.get("microbatch_chunks", 64)),
                    num_workers=int(cfg.get("load_workers", 0)),
                )
                for path, (pooled, _dur, _wav) in zip(to_process, results):
                    arr = pooled.squeeze(0).detach().cpu().numpy()
                    cached_embs[path] = arr
                    if args.use_cache:
                        try:
                            cache_path = _cache_path_for(path, args.cache_dir, cache_params)
                            np.save(cache_path, arr)
                            cache_saved += 1
                        except Exception:
                            pass

            for path, labs in zip(batch, batch_labels):
                if path in cached_embs:
                    embs.append(cached_embs[path])
                    labels_f2.append(labs)
                    if args.print_files:
                        print(f"OK: {path}")
                pbar.update(1)
        except Exception:
            # Fall back to per-track to isolate failures
            for path, labs in zip(batch, batch_labels):
                try:
                    pooled, _dur, _wav = engine.embed_track(
                        path,
                        sr=int(cfg["sr"]),
                        chunk_s=float(cfg["chunk_s"]),
                        hop_s=float(cfg["hop_s"]),
                        active_keep=int(cfg.get("active_keep", 4)),
                    )
                    arr = pooled.squeeze(0).detach().cpu().numpy()
                    embs.append(arr)
                    labels_f2.append(labs)
                    if args.print_files:
                        print(f"OK: {path}")
                    if args.use_cache:
                        try:
                            cache_path = _cache_path_for(path, args.cache_dir, cache_params)
                            np.save(cache_path, arr)
                            cache_saved += 1
                        except Exception:
                            pass
                    pbar.update(1)
                except Exception as e2:
                    print(f"SKIP: {path} ({e2})")
                    pbar.update(1)
    pbar.close()
    if args.use_cache:
        print(f"Cache: hits={cache_hits} misses={cache_misses} saved={cache_saved}")

    if not embs:
        raise SystemExit("No embeddings produced. All files failed or were skipped.")

    X = np.stack(embs, axis=0)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_f2)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=float(args.val_size), random_state=42, shuffle=True
    )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    )
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)
    metrics = {
        "f1_micro": float(f1_score(Y_val, Y_pred, average="micro")),
        "f1_macro": float(f1_score(Y_val, Y_pred, average="macro")),
        "n_labels": int(len(mlb.classes_)),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "model": clf,
        "mlb": mlb,
        "metrics": metrics,
        "config": {
            "csv": args.csv,
            "root": args.root,
            "map": args.map_path,
            "min_per_class": int(args.min_per_class),
            "val_size": float(args.val_size),
        },
    }
    joblib.dump(payload, args.out)

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
