"""
hdfs_upload.py
==============
Upload project datasets/assets to HDFS for Big Data infrastructure proof.
"""

from __future__ import annotations

import subprocess
import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_hdfs(command: list[str]) -> None:
    logger.info("Running: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Hadoop CLI was not found. Install Hadoop locally or add its bin "
            "directory to PATH, then retry. On Windows, this usually means "
            "HADOOP_HOME is set and %HADOOP_HOME%\\bin contains hdfs.cmd."
        ) from exc


def resolve_hdfs_bin(hdfs_bin: str) -> str:
    resolved = shutil.which(hdfs_bin)
    if resolved:
        return resolved

    if Path(hdfs_bin).exists():
        return hdfs_bin

    raise RuntimeError(
        f"Could not find Hadoop CLI executable '{hdfs_bin}'. Pass --hdfs-bin "
        "with the full path to hdfs.cmd, or add Hadoop's bin directory to PATH."
    )


def upload_path(local_path: Path, hdfs_path: str, hdfs_bin: str) -> None:
    if not local_path.exists():
        logger.warning("Skipping missing path: %s", local_path)
        return
    run_hdfs([hdfs_bin, "dfs", "-mkdir", "-p", hdfs_path.rsplit("/", 1)[0]])
    run_hdfs([hdfs_bin, "dfs", "-put", "-f", str(local_path), hdfs_path])


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload stock project assets to HDFS")
    parser.add_argument("--hdfs-root", default="/stock_data")
    parser.add_argument(
        "--hdfs-bin",
        default="hdfs",
        help="Hadoop CLI executable name or full path, e.g. C:\\hadoop\\bin\\hdfs.cmd",
    )
    args = parser.parse_args()

    try:
        hdfs_bin = resolve_hdfs_bin(args.hdfs_bin)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    hdfs_root = args.hdfs_root.rstrip("/")
    run_hdfs([hdfs_bin, "dfs", "-mkdir", "-p", hdfs_root])

    upload_map = {
        PROJECT_ROOT / "data" / "stock_market_dataset" / "stocks": f"{hdfs_root}/stocks",
        PROJECT_ROOT / "data" / "stock_market_dataset" / "etfs": f"{hdfs_root}/etfs",
        PROJECT_ROOT / "data" / "final_featured_data.csv": f"{hdfs_root}/final_featured_data.csv",
        PROJECT_ROOT / "images": f"{hdfs_root}/candlestick_images",
        PROJECT_ROOT / "data" / "candlestick_image_dataset.csv": f"{hdfs_root}/candlestick_image_dataset.csv",
    }
    for local, remote in upload_map.items():
        upload_path(local, remote, hdfs_bin)

    logger.info("HDFS upload pipeline complete.")


if __name__ == "__main__":
    main()

