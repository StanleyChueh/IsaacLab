#!/usr/bin/env python

import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    repo_id = "ethanCSL/openarm_visuomotor_VR_pringles_V13_no_background_strong"  # The name of the dataset repo on Hugging Face Hub
    dataset_dir = Path("~/Stanley_ws/IsaacLab/datasets").expanduser() / repo_id

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    api = HfApi()
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
    )

    # LeRobot resolves a dataset by git *revision*, not by branch: LeRobotDatasetMetadata
    # calls get_safe_version(repo_id, codebase_version), which raises RevisionNotFoundError
    # unless a tag named after info.json's codebase_version exists. lerobot's own
    # LeRobotDataset.push_to_hub() creates that tag; upload_large_folder() does not, so
    # without this the upload looks complete and lerobot-train still fails to load it.
    codebase_version = json.loads((dataset_dir / "meta" / "info.json").read_text())["codebase_version"]
    api.create_tag(repo_id, tag=codebase_version, repo_type="dataset", exist_ok=True)
    print(f"Pushed {repo_id} and tagged it {codebase_version}")

if __name__ == "__main__":
    main()
