"""
Script to download or sync dataset files from Google Drive directly to local workspace.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.config import get_config


def download_from_gdrive_folder(folder_url_or_id: str, output_dir: Path):
    """Download Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        print("[*] Installing gdown library for Google Drive downloading...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Google Drive dataset to {output_dir}...")
    
    if "drive.google.com" in folder_url_or_id:
        gdown.download_folder(url=folder_url_or_id, output=str(output_dir), remaining_ok=True)
    else:
        gdown.download_folder(id=folder_url_or_id, output=str(output_dir), remaining_ok=True)
        
    print("[✓] Download completed!")


def main():
    parser = argparse.ArgumentParser(description="Download BraTS dataset from Google Drive to local project")
    parser.add_argument("--drive-url", type=str, default=None, help="Google Drive shared folder URL or folder ID")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination local directory")
    args = parser.parse_args()

    cfg = get_config()
    output_dir = Path(args.output_dir) if args.output_dir else cfg.root_dir / "Datasets" / "brats2023-gli-dataset" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

    if args.drive_url:
        download_from_gdrive_folder(args.drive_url, output_dir)
    else:
        print("==================================================")
        print(" Google Drive Local Auto-Detector")
        print("==================================================")
        resolved_root = cfg.data_root
        if resolved_root.exists():
            print(f"[✓] Successfully detected BraTS dataset on local machine/Drive:")
            print(f"    Path: {resolved_root}")
        else:
            print(f"[!] Dataset folder not found at default paths.")
            print(f"    To download directly from Google Drive, run:")
            print(f"    python scripts/download_drive_data.py --drive-url \"YOUR_GOOGLE_DRIVE_FOLDER_URL\"")


if __name__ == "__main__":
    main()
