import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Local directory to store the model snapshot")
    ap.add_argument("--repo", default="OpenMuQ/MuQ-MuLan-large", help="HF repo id")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise SystemExit(
            "huggingface_hub is required for prefetch. Install it or use transformers cache."
        ) from e

    snapshot_download(
        repo_id=args.repo,
        local_dir=args.out,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.repo} to {args.out}")


if __name__ == "__main__":
    main()
