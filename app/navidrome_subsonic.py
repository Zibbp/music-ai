import argparse
import json
import os
import re
import sqlite3
import unicodedata
import urllib.parse
import urllib.request

import psycopg

from .muq_embed import load_cfg
from .mix import get_mix_rows, _resolve_paths
from .mix_schedule import get_schedule_rows


def _base_url(cfg: dict) -> str:
    url = cfg.get("subsonic_url")
    if not url:
        raise SystemExit("Missing subsonic_url in config.yaml")
    url = url.rstrip("/")
    if not url.endswith("/rest"):
        url = url + "/rest"
    return url


def _auth_params(cfg: dict) -> dict:
    user = cfg.get("subsonic_user")
    if not user:
        raise SystemExit("Missing subsonic_user in config.yaml")

    token = cfg.get("subsonic_token")
    salt = cfg.get("subsonic_salt")
    password = cfg.get("subsonic_password")

    params = {"u": user}
    if token and salt:
        params["t"] = token
        params["s"] = salt
    elif password:
        params["p"] = password
    else:
        raise SystemExit("Provide subsonic_password or subsonic_token+subsonic_salt in config.yaml")

    params["v"] = cfg.get("subsonic_version", "1.16.1")
    params["c"] = cfg.get("subsonic_client", "musicai")
    params["f"] = "json"
    return params


def _subsonic_call(cfg: dict, endpoint: str, params: dict) -> dict:
    base = _base_url(cfg)
    all_params = _auth_params(cfg)
    all_params.update(params)
    url = f"{base}/{endpoint}"
    qs = urllib.parse.urlencode(all_params, doseq=True)
    req = urllib.request.Request(url + "?" + qs)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
    payload = json.loads(data)
    status = payload.get("subsonic-response", {}).get("status")
    if status != "ok":
        raise RuntimeError(payload)
    return payload["subsonic-response"]


def _normalize_path(p: str) -> str:
    p = unicodedata.normalize("NFC", (p or ""))
    p = p.replace("\\", "/")
    parts = []
    for part in p.split("/"):
        part = unicodedata.normalize("NFC", part)
        part = part.replace("\ufe0e", "").replace("\ufe0f", "").replace("\u200d", "")
        part = part.casefold().strip()
        if part:
            parts.append(part)
    return "/".join(parts)


def _normalize_path_nospace(p: str) -> str:
    return _normalize_path(p).replace(" ", "")


def _path_suffix_match(path_a: str, path_b: str) -> bool:
    a = _normalize_path(path_a)
    b = _normalize_path(path_b)
    if not a or not b:
        return False
    if a == b or a.endswith(f"/{b}") or b.endswith(f"/{a}"):
        return True

    a_ns = _normalize_path_nospace(path_a)
    b_ns = _normalize_path_nospace(path_b)
    return a_ns == b_ns or a_ns.endswith(f"/{b_ns}") or b_ns.endswith(f"/{a_ns}")


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _strip_prefix(path: str, prefix: str | None) -> str:
    if not prefix:
        return path
    if path.startswith(prefix):
        return path[len(prefix) :].lstrip("/\\")
    return path


def _lookup_ids_sqlite(db_path: str, paths: list[str], strip_prefix: str | None):
    if not db_path or not os.path.exists(db_path):
        return {}, {}, []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    id_map = {}
    meta_map = {}
    missing = []
    try:
        select_sql = "SELECT id, path, title, artist FROM media_file WHERE path=?"
        select_like_sql = "SELECT id, path, title, artist FROM media_file WHERE path LIKE ? ESCAPE '\\' LIMIT 300"
        has_meta = True
        cur.execute("SELECT title, artist FROM media_file LIMIT 1")
    except Exception:
        select_sql = "SELECT id, path FROM media_file WHERE path=?"
        select_like_sql = "SELECT id, path FROM media_file WHERE path LIKE ? ESCAPE '\\' LIMIT 300"
        has_meta = False

    for p in paths:
        rel = _strip_prefix(p, strip_prefix)
        row = cur.execute(select_sql, (rel,)).fetchone()
        if not row:
            rel_norm = rel.replace("\\", "/").lstrip("/")
            parts = [x for x in rel_norm.split("/") if x]
            patterns = []
            if parts:
                patterns.append(f"%{_escape_like(parts[-1])}")
            if len(parts) >= 2:
                patterns.append(f"%{_escape_like('/'.join(parts[-2:]))}")

            candidates = []
            seen_paths = set()
            for pattern in patterns:
                rows = cur.execute(select_like_sql, (pattern,)).fetchall()
                for cand in rows:
                    cpath = cand["path"]
                    if cpath in seen_paths:
                        continue
                    seen_paths.add(cpath)
                    candidates.append(cand)
            row = next((cand for cand in candidates if _path_suffix_match(cand["path"], rel)), None)
        if row:
            id_map[p] = str(row["id"])
            if has_meta:
                meta_map[p] = {"title": row["title"], "artist": row["artist"]}
        else:
            missing.append(p)

    conn.close()
    return id_map, meta_map, missing


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _parse_title_artist(path: str) -> tuple[str, str]:
    base = os.path.splitext(os.path.basename(path))[0]
    # strip leading track numbers like "01 - "
    base = re.sub(r"^[0-9]+\\s*[-_.]\\s*", "", base)
    # common separators
    for sep in [" - ", " – ", " — ", " · ", " • "]:
        if sep in base:
            parts = [p.strip() for p in base.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]
    return "", base


def _score_candidate(title_hint: str, artist_hint: str, dur_hint: float | None, cand: dict, tol_s: float) -> float:
    score = 0.0
    c_title = _normalize_text(cand.get("title", ""))
    c_artist = _normalize_text(cand.get("artist", ""))

    if title_hint:
        if c_title == title_hint:
            score += 2.0
        elif title_hint in c_title or c_title in title_hint:
            score += 1.0
    if artist_hint:
        if c_artist == artist_hint:
            score += 2.0
        elif artist_hint in c_artist or c_artist in artist_hint:
            score += 1.0
    if dur_hint is not None and cand.get("duration") is not None:
        try:
            diff = abs(float(cand["duration"]) - float(dur_hint))
            if diff <= tol_s:
                score += 1.0
            elif diff <= tol_s * 2:
                score += 0.5
        except Exception:
            pass
    return score


def _find_song_id(
    cfg: dict,
    path: str,
    duration_s: float | None = None,
    *,
    debug: bool = False,
) -> str | None:
    prefix = cfg.get("subsonic_path_strip_prefix")
    local_rel = _strip_prefix(path, prefix)
    local_norm = _normalize_path(local_rel)
    artist_hint, title_hint = _parse_title_artist(path)
    artist_hint = _normalize_text(artist_hint)
    title_hint = _normalize_text(title_hint)

    song_limit = int(cfg.get("subsonic_search_limit", 50))
    dur_tol = float(cfg.get("subsonic_duration_tolerance_s", 3.0))
    res = _subsonic_call(
        cfg,
        "search3",
        {"query": title_hint or os.path.splitext(os.path.basename(path))[0], "songCount": song_limit, "albumCount": 0, "artistCount": 0},
    )
    songs = res.get("searchResult3", {}).get("song", [])
    if isinstance(songs, dict):
        songs = [songs]

    if debug:
        has_path = any("path" in s for s in songs)
        print(f"[debug] search3 query='{title_hint or os.path.splitext(os.path.basename(path))[0]}' results={len(songs)} path_field={has_path}")

    for s in songs:
        spath = s.get("path") or ""
        if _path_suffix_match(spath, local_norm):
            return s.get("id")

    best_id = None
    best_score = -1.0
    for s in songs:
        score = _score_candidate(title_hint, artist_hint, duration_s, s, dur_tol)
        if score > best_score:
            best_score = score
            best_id = s.get("id")

    if debug and songs:
        ranked = sorted(
            (
                (
                    _score_candidate(title_hint, artist_hint, duration_s, s, dur_tol),
                    s.get("id"),
                    s.get("artist"),
                    s.get("title"),
                    s.get("duration"),
                    s.get("path"),
                )
                for s in songs
            ),
            reverse=True,
        )[:5]
        for sc, sid, art, title, dur, spath in ranked:
            print(f"[debug] cand score={sc:.2f} id={sid} artist={art} title={title} dur={dur} path={spath}")

    if best_score >= 2.0:
        return best_id
    return None


def _get_playlist_id(cfg: dict, name: str) -> str | None:
    res = _subsonic_call(cfg, "getPlaylists", {})
    playlists = res.get("playlists", {}).get("playlist", [])
    if isinstance(playlists, dict):
        playlists = [playlists]
    for p in playlists:
        if p.get("name") == name:
            return p.get("id")
    return None


def _delete_playlist(cfg: dict, playlist_id: str) -> None:
    _subsonic_call(cfg, "deletePlaylist", {"id": playlist_id})


def _create_playlist(cfg: dict, name: str, song_ids: list[str]) -> str:
    params = {"name": name}
    if song_ids:
        params["songId"] = song_ids
    res = _subsonic_call(cfg, "createPlaylist", params)
    return res.get("playlist", {}).get("id")


def _add_to_playlist(cfg: dict, playlist_id: str, song_ids: list[str]) -> None:
    if not song_ids:
        return
    _subsonic_call(cfg, "updatePlaylist", {"playlistId": playlist_id, "songIdToAdd": song_ids})


def _chunks(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mix", "schedule", "paths"], required=True)
    ap.add_argument("--playlist-name", help="Playlist name (default: generated)")
    ap.add_argument("--replace", action="store_true", help="Replace existing playlist with same name")
    ap.add_argument("--dry-run", action="store_true", help="Do not create playlists")

    # mix args
    ap.add_argument("--seed", help="Seed track path")
    ap.add_argument("--seeds", help="Comma-separated seed paths")
    ap.add_argument("--same-genre", action="store_true")
    ap.add_argument("--same-family", action="store_true")
    ap.add_argument("--same-mood", action="store_true")
    ap.add_argument("--bpm-window", type=float, default=None)

    # schedule args
    ap.add_argument("--genre", help="Target genre label")
    ap.add_argument("--genres", help="Comma-separated list of genres to blend")
    ap.add_argument("--family", help="Target family label")
    ap.add_argument("--families", help="Comma-separated list of families to blend")
    ap.add_argument("--ratios", help="Comma-separated ratios")
    ap.add_argument("--even-split", action="store_true")
    ap.add_argument("--period", choices=["daily", "weekly"], default="daily")
    ap.add_argument("--date", default=None)
    ap.add_argument("--lookback-days", type=int, default=None)

    # common
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--paths-file", help="File with one track path per line")
    ap.add_argument("--debug-missing", action="store_true", help="Print debug info for missing matches")
    args = ap.parse_args()

    cfg = load_cfg()

    rows = []
    meta = {}
    target_k = int(args.k)
    if args.mode == "mix":
        seed_paths = []
        if args.seed:
            seed_paths.append(args.seed)
        if args.seeds:
            seed_paths.extend([p.strip() for p in args.seeds.split(",") if p.strip()])
        if not seed_paths:
            raise SystemExit("Provide --seed or --seeds for mix mode")
        seed_paths = _resolve_paths(cfg, seed_paths)
        overfetch = int(cfg.get("mix_overfetch", 3))
        if overfetch < 1:
            overfetch = 1
        candidate_k = target_k * overfetch
        rows = get_mix_rows(
            cfg,
            seed_paths,
            k=candidate_k,
            same_genre=bool(args.same_genre),
            same_family=bool(args.same_family),
            same_mood=bool(args.same_mood),
            bpm_window=args.bpm_window,
            exclude_seeds=True,
        )
        meta = {"name": f"mix:{os.path.basename(seed_paths[0])}"}
    elif args.mode == "schedule":
        rows, meta = get_schedule_rows(
            cfg,
            period=args.period,
            k=int(args.k),
            date=args.date,
            genre=args.genre,
            genres=args.genres,
            family=args.family,
            families=args.families,
            ratios=args.ratios,
            even_split=bool(args.even_split),
            lookback_days=args.lookback_days,
        )
    else:
        if not args.paths_file:
            raise SystemExit("Provide --paths-file for paths mode")
        with open(args.paths_file, "r", encoding="utf-8") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        rows = [{"path": p} for p in paths]
        meta = {"name": os.path.basename(args.paths_file)}
        target_k = len(rows)

    if not rows:
        print("No rows to build playlist.")
        return

    playlist_name = args.playlist_name
    if not playlist_name:
        if args.mode == "schedule":
            playlist_name = f"{meta['period']}-{meta['target_key']}-{meta['period_start']}"
        else:
            playlist_name = meta.get("name", "musicai-mix")

    song_ids = []
    missing = []
    missing_set = set()
    duplicates = []
    seen_keys = set()

    duration_map = {}
    try:
        dsn = cfg.get("db_dsn")
        if dsn:
            with psycopg.connect(dsn) as conn:
                rows_for_paths = [r.get("path") for r in rows if r.get("path")]
                if rows_for_paths:
                    q = conn.execute(
                        "SELECT path, duration_s FROM tracks WHERE path = ANY(%s)",
                        (rows_for_paths,),
                    ).fetchall()
                    duration_map = {p: d for p, d in q}
    except Exception:
        duration_map = {}

    paths = [r.get("path") for r in rows if r.get("path")]
    id_map = {}
    meta_map = {}
    db_path = cfg.get("subsonic_sqlite_db_path")
    strip_prefix = cfg.get("subsonic_sqlite_path_strip_prefix")
    id_map, meta_map, sqlite_missing = _lookup_ids_sqlite(db_path, paths, strip_prefix)
    if sqlite_missing:
        print(f"SQLite mapping missed {len(sqlite_missing)} paths; falling back to Subsonic search.")

    def _add_missing(path: str) -> None:
        if path not in missing_set:
            missing_set.add(path)
            missing.append(path)

    for r in rows:
        path = r.get("path")
        if not path:
            continue
        sid = id_map.get(path)
        if not sid:
            sid = _find_song_id(
                cfg,
                path,
                duration_s=duration_map.get(path),
                debug=bool(args.debug_missing),
            )
        if not sid:
            _add_missing(path)
            continue
        key = sid
        track_meta = meta_map.get(path)
        if track_meta:
            title = _normalize_text(track_meta.get("title"))
            artist = _normalize_text(track_meta.get("artist"))
            if title and artist:
                key = f"{title}::{artist}"
        if key in seen_keys:
            duplicates.append(path)
            continue
        seen_keys.add(key)
        song_ids.append(sid)
        if len(song_ids) >= target_k:
            break

    if missing:
        print(f"Missing {len(missing)} tracks in Subsonic index.")
    if duplicates:
        print(f"Skipped {len(duplicates)} duplicates by title+artist.")
    if len(song_ids) < target_k:
        print(
            f"Only {len(song_ids)} tracks available for requested {target_k} "
            f"(candidates={len(rows)}, missing={len(missing)}, duplicates={len(duplicates)})."
        )

    if args.dry_run:
        print(f"Would create playlist '{playlist_name}' with {len(song_ids)} tracks.")
        return

    if args.mode == "schedule":
        try:
            dsn = cfg.get("db_dsn")
            if dsn:
                with psycopg.connect(dsn) as conn:
                    conn.row_factory = psycopg.rows.dict_row
                    mix_id = conn.execute(
                        """
                        INSERT INTO mix_history(period, genre, period_start, seed)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (meta["period"], meta["target_key"], meta["period_start"], meta["seed"]),
                    ).fetchone()["id"]
                    for r in rows:
                        conn.execute(
                            "INSERT INTO mix_items(mix_id, path) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (mix_id, r["path"]),
                        )
                    conn.commit()
        except Exception as e:
            print(f"Warning: failed to persist mix history ({e}).")

    existing = _get_playlist_id(cfg, playlist_name)
    if existing:
        _delete_playlist(cfg, existing)

    playlist_id = _create_playlist(cfg, playlist_name, song_ids[:100])
    for chunk in _chunks(song_ids[100:], 100):
        _add_to_playlist(cfg, playlist_id, chunk)

    print(f"Playlist '{playlist_name}' updated with {len(song_ids)} tracks.")
    if missing:
        print("Missing paths:")
        for p in missing[:20]:
            print(f"- {p}")
    if duplicates:
        print("Duplicate paths (sample):")
        for p in duplicates[:20]:
            print(f"- {p}")
