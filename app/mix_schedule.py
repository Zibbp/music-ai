import argparse
import datetime as dt

import psycopg
from pgvector.psycopg import register_vector

from .muq_embed import load_cfg


def _period_start(period: str, day: dt.date) -> dt.date:
    if period == "daily":
        return day
    if period == "weekly":
        # ISO week starts Monday
        return day - dt.timedelta(days=day.weekday())
    raise ValueError("period must be daily or weekly")


def get_schedule_rows(
    cfg: dict,
    *,
    period: str,
    k: int,
    date: str | None = None,
    genre: str | None = None,
    genres: str | None = None,
    family: str | None = None,
    families: str | None = None,
    ratios: str | None = None,
    even_split: bool = False,
    lookback_days: int | None = None,
):
    today = dt.date.today() if not date else dt.date.fromisoformat(date)
    period_start = _period_start(period, today)
    genres_list = []
    if genre:
        genres_list.append(genre)
    if genres:
        genres_list.extend([g.strip() for g in genres.split(",") if g.strip()])

    families_list = []
    if family:
        families_list.append(family)
    if families:
        families_list.extend([f.strip() for f in families.split(",") if f.strip()])

    if genres_list and families_list:
        raise SystemExit("Use either genres or families, not both.")
    if not genres_list and not families_list:
        raise SystemExit("Provide --genre/--genres or --family/--families")

    target_field = "genre" if genres_list else "family"
    targets_list = genres_list if genres_list else families_list
    targets_list = sorted(set(targets_list))
    target_key = ",".join(targets_list)
    seed = f"{period}:{period_start.isoformat()}:{target_field}:{target_key}"

    if lookback_days is None:
        if period == "daily":
            lookback_days = int(cfg.get("mix_lookback_days_daily", 30))
        else:
            lookback_days = int(cfg.get("mix_lookback_days_weekly", 90))

    with psycopg.connect(cfg["db_dsn"]) as conn:
        register_vector(conn)
        conn.row_factory = psycopg.rows.dict_row

        def _base_exclude_clause():
            return """
            t.path NOT IN (
              SELECT mi.path
              FROM mix_items mi
              JOIN mix_history mh ON mh.id = mi.mix_id
              WHERE mh.genre = %(target_key)s
                AND mh.period = %(period)s
                AND mh.period_start >= (CURRENT_DATE - (%(lookback)s || ' days')::interval)
            )
            """

        def _fetch_target(value: str, limit: int, exclude: list[str]):
            if limit <= 0:
                return []
            where = [f"t.{target_field} = %(target)s", _base_exclude_clause()]
            params = {
                "target": value,
                "target_key": target_key,
                "seed": seed,
                "period": period,
                "period_start": period_start,
                "lookback": int(lookback_days),
                "k": int(limit),
            }
            if exclude:
                where.append("t.path <> ALL(%(exclude)s)")
                params["exclude"] = exclude
            where_sql = " AND ".join(where)
            sql = f"""
            SELECT t.path, t.genre, t.family, t.mood, t.bpm
            FROM tracks t
            WHERE {where_sql}
            ORDER BY md5(t.path || %(seed)s || %(target)s)
            LIMIT %(k)s
            """
            return conn.execute(sql, params).fetchall()

        def _fetch_blended(limit: int, exclude: list[str]):
            if limit <= 0:
                return []
            where = [f"t.{target_field} = ANY(%(targets)s)", _base_exclude_clause()]
            params = {
                "targets": targets_list,
                "target_key": target_key,
                "seed": seed,
                "period": period,
                "period_start": period_start,
                "lookback": int(lookback_days),
                "k": int(limit),
            }
            if exclude:
                where.append("t.path <> ALL(%(exclude)s)")
                params["exclude"] = exclude
            where_sql = " AND ".join(where)
            sql = f"""
            SELECT t.path, t.genre, t.family, t.mood, t.bpm
            FROM tracks t
            WHERE {where_sql}
            ORDER BY md5(t.path || %(seed)s)
            LIMIT %(k)s
            """
            return conn.execute(sql, params).fetchall()

        rows = []
        if len(targets_list) == 1:
            rows = _fetch_target(targets_list[0], int(k), [])
        else:
            ratio_map = None
            if ratios:
                ratio_map = {}
                for part in ratios.split(","):
                    if not part.strip():
                        continue
                    if "=" not in part:
                        raise SystemExit("Invalid --ratios format. Use genre=0.5,other=0.5")
                    g, v = part.split("=", 1)
                    g = g.strip()
                    if g not in targets_list:
                        raise SystemExit(f"Ratio target '{g}' not in requested targets.")
                    ratio_map[g] = float(v)
                if not ratio_map:
                    ratio_map = None

            targets = {}
            if ratio_map:
                total = sum(ratio_map.values())
                if total <= 0:
                    raise SystemExit("Ratios must sum to > 0")
                fracs = []
                for g in targets_list:
                    raw = (ratio_map.get(g, 0.0) / total) * int(k)
                    base = int(raw)
                    targets[g] = base
                    fracs.append((raw - base, g))
                diff = int(k) - sum(targets.values())
                if diff > 0:
                    fracs.sort(reverse=True)
                    for _i in range(diff):
                        targets[fracs[_i % len(fracs)][1]] += 1
            else:
                if even_split or not ratio_map:
                    base = int(k) // len(targets_list)
                    rem = int(k) % len(targets_list)
                    for i, g in enumerate(targets_list):
                        targets[g] = base + (1 if i < rem else 0)

            selected = []
            selected_paths = []
            for g in targets_list:
                picked = _fetch_target(g, targets.get(g, 0), selected_paths)
                for r in picked:
                    selected.append(r)
                    selected_paths.append(r["path"])

            # Fill remainder if any genre had insufficient tracks
            remaining = int(k) - len(selected)
            if remaining > 0:
                picked = _fetch_blended(remaining, selected_paths)
                selected.extend(picked)

            rows = selected

        return rows, {
            "target_key": target_key,
            "seed": seed,
            "period_start": period_start,
            "period": period,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", help="Target genre label")
    ap.add_argument("--genres", help="Comma-separated list of genres to blend")
    ap.add_argument("--family", help="Target family label")
    ap.add_argument("--families", help="Comma-separated list of families to blend")
    ap.add_argument("--ratios", help="Comma-separated ratios, e.g. dnb=0.5,techno=0.3,house=0.2")
    ap.add_argument("--even-split", action="store_true", help="Evenly split across targets")
    ap.add_argument("--period", choices=["daily", "weekly"], default="daily")
    ap.add_argument("--k", type=int, default=50, help="Number of tracks in mix")
    ap.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    ap.add_argument("--lookback-days", type=int, default=None, help="Exclude tracks used in past N days")
    ap.add_argument("--dry-run", action="store_true", help="Do not persist mix history")
    args = ap.parse_args()

    cfg = load_cfg()
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

    if not rows:
        print("No tracks found for this mix (after filters).")
        return

    if not args.dry_run:
        with psycopg.connect(cfg["db_dsn"]) as conn:
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

    for r in rows:
        bpm = r["bpm"]
        bpm_s = f"{bpm:.1f}" if isinstance(bpm, (int, float)) else "n/a"
        print(f"{r['genre'] or 'n/a'} | {r['mood'] or 'n/a'} | {bpm_s} | {r['path']}")


if __name__ == "__main__":
    main()
