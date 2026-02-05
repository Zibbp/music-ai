import numpy as np
import psycopg
from .muq_embed import load_cfg

def main():
    cfg = load_cfg()
    p_lo = float(cfg["bpm_p_low"])
    p_hi = float(cfg["bpm_p_high"])

    with psycopg.connect(cfg["db_dsn"]) as conn:
        # only use confident tags (threshold+margin already enforced by tag decision)
        rows = conn.execute("""
        SELECT genre, mood, bpm
        FROM tracks
        WHERE genre_confident IS TRUE AND mood_confident IS TRUE AND bpm IS NOT NULL
        """).fetchall()

        buckets = {}
        for g, m, bpm in rows:
            buckets.setdefault((g, m), []).append(float(bpm))

        conn.execute("TRUNCATE tag_stats")
        for (g, m), bpms in buckets.items():
            bpms = np.array(bpms, dtype=np.float32)
            if len(bpms) < 20:
                continue
            lo = float(np.percentile(bpms, p_lo))
            hi = float(np.percentile(bpms, p_hi))
            conn.execute(
                "INSERT INTO tag_stats(genre,mood,bpm_p_low,bpm_p_high,n) VALUES (%s,%s,%s,%s,%s)",
                (g, m, lo, hi, int(len(bpms))),
            )
        conn.commit()

    print("Learned tag_stats.")
