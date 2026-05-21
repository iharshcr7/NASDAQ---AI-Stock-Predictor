"""
upload_images_to_hdfs.py
=========================
Upload candlestick images to HDFS for Big Data proof.

Uploads all images from local images/ directory to HDFS at:
    hdfs://localhost:9000/stock_data/images/bullish/
    hdfs://localhost:9000/stock_data/images/bearish/

This demonstrates distributed storage of unstructured data.

Usage:
    python scripts/upload_images_to_hdfs.py
    
    # With custom HDFS path
    python scripts/upload_images_to_hdfs.py --hdfs-path /custom/path/images/
    
    # Skip verification (faster)
    python scripts/upload_images_to_hdfs.py --skip-verify
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
BULLISH_DIR = IMAGES_DIR / "bullish"
BEARISH_DIR = IMAGES_DIR / "bearish"

HDFS_NAMENODE = "hdfs://localhost:9000"
DEFAULT_HDFS_PATH = "/stock_data/images/"


def check_hdfs_available() -> bool:
    """Check if HDFS is available and accessible."""
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-test", "-d", "/"],
            capture_output=True,
            timeout=10,
            shell=True
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.error(f"HDFS availability check failed: {e}")
        return False


def create_hdfs_directory(hdfs_path: str) -> bool:
    """Create HDFS directory if it doesn't exist."""
    try:
        logger.info(f"Creating HDFS directory: {hdfs_path}")
        
        cmd = ["hdfs", "dfs", "-mkdir", "-p", hdfs_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )
        
        if result.returncode == 0:
            logger.info(f"Directory created: {hdfs_path}")
            return True
        else:
            logger.warning(f"mkdir returned {result.returncode} (may already exist)")
            return True  # Directory might already exist
            
    except Exception as e:
        logger.error(f"Failed to create HDFS directory: {e}")
        return False


def upload_directory_to_hdfs(
    local_dir: Path,
    hdfs_path: str,
    verify: bool = True
) -> Tuple[bool, int]:
    """
    Upload entire directory to HDFS.
    
    Args:
        local_dir: Local directory path
        hdfs_path: HDFS destination path
        verify: Whether to verify upload
        
    Returns:
        Tuple of (success, file_count)
    """
    if not local_dir.exists():
        logger.error(f"Local directory not found: {local_dir}")
        return False, 0
    
    # Count files
    files = list(local_dir.glob("*.png"))
    file_count = len(files)
    
    if file_count == 0:
        logger.warning(f"No PNG files found in {local_dir}")
        return True, 0
    
    logger.info(f"Uploading {file_count} files from {local_dir.name}...")
    
    try:
        # Upload entire directory
        # Using -put with -f flag to overwrite if exists
        cmd = ["hdfs", "dfs", "-put", "-f", str(local_dir), hdfs_path]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout for large uploads
            shell=True
        )
        
        if result.returncode == 0:
            logger.info(f"Upload successful: {file_count} files")
            
            # Verify if requested
            if verify:
                hdfs_dir_path = f"{hdfs_path}{local_dir.name}/"
                verify_cmd = ["hdfs", "dfs", "-count", hdfs_dir_path]
                verify_result = subprocess.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=True
                )
                
                if verify_result.returncode == 0:
                    logger.info(f"Verification passed: {hdfs_dir_path}")
                    # Parse count output
                    output = verify_result.stdout.strip()
                    if output:
                        parts = output.split()
                        if len(parts) >= 3:
                            hdfs_file_count = int(parts[2])
                            logger.info(f"  HDFS file count: {hdfs_file_count}")
                else:
                    logger.warning("Verification failed but upload may have succeeded")
            
            return True, file_count
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"Upload failed: {error_msg}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False, 0
            
    except subprocess.TimeoutExpired:
        logger.error("Upload timed out (>5 minutes)")
        return False, 0
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        return False, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload candlestick images to HDFS"
    )
    parser.add_argument(
        "--hdfs-path",
        type=str,
        default=DEFAULT_HDFS_PATH,
        help=f"HDFS destination path (default: {DEFAULT_HDFS_PATH})"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification after upload (faster)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("HDFS IMAGE UPLOAD")
    logger.info("=" * 70)
    
    # Check local directories
    if not IMAGES_DIR.exists():
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        logger.error("Run generate_candlestick_images.py first")
        sys.exit(1)
    
    if not BULLISH_DIR.exists() or not BEARISH_DIR.exists():
        logger.error("Bullish or bearish directory not found")
        sys.exit(1)
    
    # Count local images
    bullish_count = len(list(BULLISH_DIR.glob("*.png")))
    bearish_count = len(list(BEARISH_DIR.glob("*.png")))
    total_count = bullish_count + bearish_count
    
    logger.info(f"Local images:")
    logger.info(f"  Bullish: {bullish_count}")
    logger.info(f"  Bearish: {bearish_count}")
    logger.info(f"  Total: {total_count}")
    
    if total_count == 0:
        logger.error("No images found to upload")
        sys.exit(1)
    
    # Check HDFS availability
    logger.info("\nChecking HDFS availability...")
    if not check_hdfs_available():
        logger.error("HDFS is not available or not running")
        logger.error("\nTo start HDFS:")
        logger.error("  start-dfs.sh")
        logger.error("\nTo verify HDFS is running:")
        logger.error("  jps  # Should show NameNode and DataNode")
        sys.exit(1)
    
    logger.info("HDFS is available")
    
    # Create base directory
    if not create_hdfs_directory(args.hdfs_path):
        logger.error("Failed to create HDFS base directory")
        sys.exit(1)
    
    # Upload bullish images
    logger.info("\n" + "=" * 70)
    logger.info("UPLOADING BULLISH IMAGES")
    logger.info("=" * 70)
    
    bullish_success, bullish_uploaded = upload_directory_to_hdfs(
        BULLISH_DIR,
        args.hdfs_path,
        verify=not args.skip_verify
    )
    
    if not bullish_success:
        logger.error("Failed to upload bullish images")
        sys.exit(1)
    
    # Upload bearish images
    logger.info("\n" + "=" * 70)
    logger.info("UPLOADING BEARISH IMAGES")
    logger.info("=" * 70)
    
    bearish_success, bearish_uploaded = upload_directory_to_hdfs(
        BEARISH_DIR,
        args.hdfs_path,
        verify=not args.skip_verify
    )
    
    if not bearish_success:
        logger.error("Failed to upload bearish images")
        sys.exit(1)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("UPLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Bullish images uploaded: {bullish_uploaded}")
    logger.info(f"Bearish images uploaded: {bearish_uploaded}")
    logger.info(f"Total images uploaded: {bullish_uploaded + bearish_uploaded}")
    logger.info(f"\nHDFS location: {HDFS_NAMENODE}{args.hdfs_path}")
    logger.info("\nVerify with:")
    logger.info(f"  hdfs dfs -ls {args.hdfs_path}")
    logger.info(f"  hdfs dfs -count {args.hdfs_path}")
    logger.info(f"  hdfs dfs -ls {args.hdfs_path}bullish/ | head -5")
    logger.info(f"  hdfs dfs -ls {args.hdfs_path}bearish/ | head -5")
    logger.info("=" * 70)
    logger.info("HDFS upload complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"HDFS upload failed: {e}")
        sys.exit(1)
