#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the three public datasets used by this project into the local
`data/` directory so the scripts under `code/` can run out of the box.

Datasets (all public, third-party):
  - CED     : https://github.com/thunlp/Chinese_Rumor_Dataset        -> data/CED_Dataset/
  - LTCR    : https://github.com/Enderfga/DoubleCheck  (data/LTCR.csv) -> data/LTCR/data/
  - CHECKED : https://github.com/cyang03/CHECKED       (dataset/)     -> data/CHECKED/

Usage:
    python scripts/fetch_datasets.py

Requires `git` to be available on PATH.
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")

CED_SRC = "https://github.com/thunlp/Chinese_Rumor_Dataset.git"
LTCR_SRC = "https://github.com/Enderfga/DoubleCheck.git"
CHECKED_SRC = "https://github.com/cyang03/CHECKED.git"


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def clone_and_copy(url, tmp_dir, src_rel, dst_abs):
    """Shallow-clone a repo, copy the needed subtree to `dst_abs`, drop the rest."""
    name = url.rsplit("/", 1)[-1].replace(".git", "")
    dest = os.path.join(tmp_dir, name)
    run(["git", "clone", "--depth", "1", url, dest])
    src = os.path.join(dest, src_rel)
    if not os.path.exists(src):
        raise RuntimeError(f"Source path not found: {src}")
    if os.path.isdir(src):
        shutil.copytree(src, dst_abs, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        shutil.copy2(src, dst_abs)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="rumor_data_")
    try:
        print("=== Downloading CED ===")
        clone_and_copy(
            CED_SRC, tmp, "CED_Dataset", os.path.join(DATA_DIR, "CED_Dataset")
        )
        print("=== Downloading LTCR ===")
        clone_and_copy(
            LTCR_SRC,
            tmp,
            os.path.join("data", "LTCR.csv"),
            os.path.join(DATA_DIR, "LTCR", "data", "LTCR.csv"),
        )
        print("=== Downloading CHECKED ===")
        clone_and_copy(
            CHECKED_SRC,
            tmp,
            "dataset",
            os.path.join(DATA_DIR, "CHECKED", "dataset"),
        )
        print("\nDone. Data ready at:")
        for rel in ["data/CED_Dataset", "data/LTCR/data/LTCR.csv", "data/CHECKED/dataset"]:
            print("  -", os.path.join(REPO_DIR, rel))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
