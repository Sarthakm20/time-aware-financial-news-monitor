"""Publish the model to the Hugging Face Hub and/or the dashboard to a Hugging Face Space.

Dry run by default: prints exactly what would be uploaded. Nothing leaves your machine
without --push. Needs `huggingface-cli login` (or HF_TOKEN) with a write token.

    python deploy/publish.py model --user <hf-user>            # dry run
    python deploy/publish.py model --user <hf-user> --push
    python deploy/publish.py space --user <hf-user> --push     # after the model is on the Hub
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "distilbert_headline_adapted"
MODEL_REPO = "distilbert-financial-headline-sentiment"
SPACE_REPO = "news-sentiment-monitor"
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

# What the Space needs to serve the dashboard site and its API (no raw data, no models, no secrets).
SPACE_FILES = [
    "src/__init__.py", "src/config.py", "src/explorer.py", "src/aggregation.py", "src/service.py",
    "src/inference.py", "src/relevance.py", "src/company_names.py", "src/labels.py", "src/onnx_backend.py",
    "src/live_feed.py", "src/env.py",
    "src/api.py", "src/dashboard_api.py", "src/dashboard_data.py", "src/dashboard_metrics.py", "src/dashboard_live.py",
    "outputs/predictions/nasdaq100_scored_headlines_adapted.csv",
    "outputs/predictions/nasdaq100_scored_headlines_base.csv",
    "outputs/metrics/event_study_summary.json", "outputs/metrics/event_study_buckets.csv",
    "outputs/metrics/test_set_model_comparison.csv", "outputs/metrics/tsla_headline_model_comparison.csv",
    "outputs/metrics/seed_summary.json",
    "outputs/metrics/multi_company_pooled_accuracy.csv", "outputs/metrics/multi_company_per_company_accuracy.csv",
    "outputs/metrics/multi_company_relevance_filter.csv",
    "outputs/metrics/event_study_car_paths.csv", "outputs/metrics/corpus_summary.json",
    "outputs/metrics/event_study_buckets_by_year.csv", "outputs/metrics/event_study_car_paths_by_year.csv",
    "outputs/metrics/adapted_test_metrics.json",
]
# Directories copied whole: the static site (HTML, CSS, JS, vendored libraries, fonts).
SPACE_TREES = ["web"]


def stage_model(staging: Path, user: str) -> str:
    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        sys.exit(f"{MODEL_DIR} is missing {missing}; run `python -m src.pipeline adapt` first")
    for f in MODEL_FILES:
        shutil.copy2(MODEL_DIR / f, staging / f)
    card = (ROOT / "deploy" / "model_card.md").read_text(encoding="utf-8").replace("<hf-user>", user)
    (staging / "README.md").write_text(card, encoding="utf-8")
    return f"{user}/{MODEL_REPO}"


def stage_space(staging: Path, user: str) -> str:
    for rel in SPACE_FILES:
        src = ROOT / rel
        if not src.exists():
            sys.exit(f"missing {rel}; run the pipeline stages that produce it first")
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in SPACE_TREES:
        if not (ROOT / rel / "index.html").exists():
            sys.exit(f"missing {rel}/index.html; the dashboard site is not built")
        shutil.copytree(ROOT / rel, staging / rel, ignore=shutil.ignore_patterns("__pycache__", ".*"))
    for name in ["Dockerfile", "README.md", "requirements.txt"]:
        text = (ROOT / "deploy" / "space" / name).read_text(encoding="utf-8").replace("<hf-user>", user)
        (staging / name).write_text(text, encoding="utf-8")
    return f"{user}/{SPACE_REPO}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target", choices=["model", "space"])
    parser.add_argument("--user", required=True, help="your Hugging Face username or organisation")
    parser.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        repo_id = stage_model(staging, args.user) if args.target == "model" else stage_space(staging, args.user)
        files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
        size_mb = sum((staging / f).stat().st_size for f in files) / 1e6
        print(f"{'Uploading' if args.push else 'Would upload'} {len(files)} files ({size_mb:.1f} MB) "
              f"to {args.target} repo {repo_id}:")
        for f in files:
            print("  ", f)
        if not args.push:
            print("\nDry run. Re-run with --push to upload.")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        repo_type = "model" if args.target == "model" else "space"
        api.create_repo(repo_id, repo_type=repo_type, private=args.private, exist_ok=True,
                        **({"space_sdk": "docker"} if repo_type == "space" else {}))
        if repo_type == "space":
            api.add_space_variable(repo_id, "SENTIMENT_MODEL", f"{args.user}/{MODEL_REPO}")
        api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type=repo_type,
                          commit_message="Publish from time-aware-financial-news-monitor")
        url = f"https://huggingface.co/{'spaces/' if repo_type == 'space' else ''}{repo_id}"
        print(f"\nDone: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
