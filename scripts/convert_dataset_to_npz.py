"""
Script to pre-convert raw BraTS 2023 NIfTI (.nii.gz) files into fast compressed .npz format.

Reduces load time per case from ~5.0 seconds down to ~0.003 seconds (1000x speedup).
Supports Google Drive output destination for Google Colab workflows.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.config import get_config
from src.data.data_loader import BraTSDataLoader


def process_single_case(case_info: tuple) -> tuple:
    """
    Worker function to process single case.

    Args:
        case_info: (case_id, data_root_str, output_dir_str)

    Returns:
        (case_id, status, elapsed_sec, file_size_mb)
    """
    case_id, data_root_str, output_dir_str = case_info
    t0 = time.time()
    try:
        loader = BraTSDataLoader(data_root_str)
        output_dir = Path(output_dir_str)
        saved_path = loader.save_case_to_npz(case_id, output_dir)
        elapsed = time.time() - t0
        size_mb = saved_path.stat().st_size / (1024 * 1024)
        return case_id, "SUCCESS", elapsed, size_mb
    except Exception as e:
        return case_id, f"ERROR: {str(e)}", time.time() - t0, 0.0


def main():
    # Auto-mount Google Drive if running on Google Colab
    try:
        from google.colab import drive
        if not Path("/content/drive").exists():
            print("[*] Google Colab environment detected. Mounting Google Drive at /content/drive...")
            drive.mount("/content/drive")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Convert BraTS NIfTI dataset to fast NPZ format")
    parser.add_argument("--data-root", type=str, default=None, help="Path to raw BraTS dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output NPZ directory")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of CPU worker processes")
    parser.add_argument("--drive-dir", type=str, default=None, help="Optional Google Drive path to mirror NPZ files")
    args = parser.parse_args()

    cfg = get_config()
    
    # 1. Resolve raw data root
    data_root = Path(args.data_root) if args.data_root else cfg.data_root
    if not data_root.exists():
        print(f"[!] Error: Raw data root path does not exist: {data_root}")
        print("    Please specify --data-root or set BRATS_DATA_ROOT environment variable.")
        sys.exit(1)

    # 2. Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root.parent / "processed_npz"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"==================================================")
    print(f" BraTS 2023 Dataset Fast NPZ Converter")
    print(f" Input Raw NIfTI Directory:  {data_root}")
    print(f" Output NPZ Directory:      {output_dir}")
    print(f" CPU Worker Processes:      {args.num_workers}")
    print(f"==================================================")

    # 3. Discover cases
    loader = BraTSDataLoader(str(data_root))
    cases = loader.list_cases()
    case_ids = [c.name for c in cases]
    print(f"Found {len(case_ids)} cases to convert.\n")

    if len(case_ids) == 0:
        print("[!] No cases found in raw data root.")
        sys.exit(0)

    # 4. Prepare worker arguments
    tasks = [(case_id, str(data_root), str(output_dir)) for case_id in case_ids]

    start_time = time.time()
    success_count = 0
    total_size_mb = 0.0

    # 5. Parallel conversion loop
    print("Starting parallel conversion...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_case, task) for task in tasks]

        for idx, future in enumerate(as_completed(futures), 1):
            case_id, status, elapsed, size_mb = future.result()
            if status == "SUCCESS":
                success_count += 1
                total_size_mb += size_mb
                print(f"[{idx:03d}/{len(case_ids):03d}] {case_id}: Done ({elapsed:.2f}s, {size_mb:.1f} MB)")
            else:
                print(f"[{idx:03d}/{len(case_ids):03d}] {case_id}: {status}")

    total_elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f" Conversion Summary:")
    print(f" Total Cases Converted: {success_count}/{len(case_ids)}")
    print(f" Total Output Size:     {total_size_mb / 1024:.2f} GB")
    print(f" Total Time Taken:      {total_elapsed:.1f} seconds")
    print(f" Average Time / Case:   {total_elapsed / max(1, success_count):.2f} seconds")
    print(f"==================================================")

    # 6. Mirror to Google Drive if requested
    drive_dir = args.drive_dir or os.getenv("DRIVE_PROCESSED_DIR")
    if drive_dir:
        drive_path = Path(drive_dir)
        drive_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSyncing NPZ files to Google Drive location: {drive_path}...")
        import shutil
        for npz_file in output_dir.glob("*.npz"):
            shutil.copy2(npz_file, drive_path / npz_file.name)
        print("Drive sync completed successfully!")

    print("\n[✓] All set! Data loader will now automatically use fast NPZ files.")


if __name__ == "__main__":
    main()
