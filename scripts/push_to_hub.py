#!/usr/bin/env python

from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    repo_id = "ethanCSL/openarm_visuomotor_VR_pringles_V6"  # The name of the dataset repo on Hugging Face Hub
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

if __name__ == "__main__":
    main()
