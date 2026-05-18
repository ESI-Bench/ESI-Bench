#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def validate_dataset_file(repo_root: Path) -> None:
    dataset_file = repo_root / "data/questions.jsonl"
    if not dataset_file.exists():
        raise SystemExit(f"Missing {dataset_file}")
    if dataset_file.stat().st_size == 0:
        raise SystemExit(f"{dataset_file} is empty. Regenerate it before uploading.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload the Croissant-friendly ESI-Bench table to Hugging Face.")
    parser.add_argument("--repo-id", default="ESI-Bench/esi-bench")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private if it does not exist.")
    parser.add_argument("--commit-message", default="Replace with Croissant-friendly questions table")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_dataset_file(repo_root)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=repo_root,
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "data/questions.jsonl",
            "scripts/export_hf_questions.py",
            "scripts/upload_hf_dataset.py",
        ],
        delete_patterns=[
            "data/**",
            "dataset/**",
            "scripts/**",
        ],
        commit_message=args.commit_message,
    )
    print(f"Uploaded {repo_root} to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
