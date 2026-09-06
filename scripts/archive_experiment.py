"""Archive selected model artifacts as a named experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ARTIFACT_NAMES = (
    "xgb_quantile_model.json",
    "feature_columns.json",
    "metrics.json",
    "metrics_by_segment.json",
    "conformal_Q.json",
    "feature_importance.csv",
)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive model artifacts as a named experiment.")
    parser.add_argument("--name", required=True, help="Experiment directory name")
    parser.add_argument("--description", default="", help="Optional experiment description")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing archive")
    args = parser.parse_args()

    experiment_name = Path(args.name)
    if not args.name or experiment_name.name != args.name or str(experiment_name) in {".", ".."}:
        parser.error("--name must be a simple directory name without path separators")

    source_dir = Path("models")
    destination_dir = source_dir / "experiments" / experiment_name
    if destination_dir.exists() and not args.overwrite:
        parser.error(f"Destination already exists: {destination_dir}; use --overwrite to replace it")

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for artifact_name in ARTIFACT_NAMES:
        source_path = source_dir / artifact_name
        if source_path.exists():
            shutil.copy2(source_path, destination_dir / artifact_name)
            copied_files.append(artifact_name)
            print(f"Copied: {source_path} -> {destination_dir / artifact_name}")
        else:
            print(f"Skipped missing source: {source_path}")

    metadata = {
        "experiment_name": args.name,
        "description": args.description,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": copied_files,
        "git_commit_if_available": get_git_commit(),
    }
    metadata_path = destination_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Archived {len(copied_files)} artifact(s) in {destination_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
