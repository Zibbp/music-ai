import argparse
import json

from .muq_embed import MuQEngine, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Audio file path")
    args = ap.parse_args()

    cfg = load_cfg()
    engine = MuQEngine(
        cfg["device"],
        model_path=cfg.get("mulan_model_path"),
        offline=bool(cfg.get("mulan_offline", False)),
    )

    emb, _dur, _wav = engine.embed_track(
        args.path,
        sr=int(cfg["sr"]),
        chunk_s=float(cfg["chunk_s"]),
        hop_s=float(cfg["hop_s"]),
        active_keep=int(cfg.get("active_keep", 4)),
    )

    # Classify once with the classifier enabled (current config)
    out_clf = engine.classify(emb, cfg)

    # Classify again with classifier disabled
    cfg_no = dict(cfg)
    cfg_no["genre_clf_path"] = None
    out_prompt = engine.classify(emb, cfg_no)

    report = {
        "track": args.path,
        "with_classifier": {
            "family": out_clf.get("family"),
            "genre": out_clf.get("genre"),
            "genre_score": out_clf.get("genre_score"),
            "genre_margin": out_clf.get("genre_margin"),
            "mood": out_clf.get("mood"),
        },
        "prompt_only": {
            "family": out_prompt.get("family"),
            "genre": out_prompt.get("genre"),
            "genre_score": out_prompt.get("genre_score"),
            "genre_margin": out_prompt.get("genre_margin"),
            "mood": out_prompt.get("mood"),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
