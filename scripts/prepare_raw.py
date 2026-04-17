"""
scripts/prepare_raw.py

Convenience script: scans a source folder for all supported audio files
and copies / converts them into data/D0_raw/ as properly-named 16-bit
16 kHz mono WAV files.

Usage:
    python scripts/prepare_raw.py --src /path/to/your/recordings
    python scripts/prepare_raw.py --src /path/to/recordings --dry-run
"""

import os
import sys
import argparse
import shutil
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils


def main():
    parser = argparse.ArgumentParser(description="Prepare raw audio for D0")
    parser.add_argument("--src", required=True, help="Source folder with raw recordings")
    parser.add_argument("--dry-run", action="store_true", help="Only list files, don't copy")
    args = parser.parse_args()

    files = utils.find_audio_files(args.src)
    print(f"Found {len(files)} audio files in {args.src}")

    if args.dry_run:
        for f in files:
            print(f"  {f}")
        return

    os.makedirs(config.RAW_DIR, exist_ok=True)

    for i, src_path in enumerate(tqdm(files, desc="Preparing D0")):
        stem = f"rec_{i:04d}"
        ext  = os.path.splitext(src_path)[1].lower()

        if ext == ".wav":
            # Still re-encode to ensure correct format
            dst = os.path.join(config.RAW_DIR, f"{stem}.wav")
            try:
                utils.convert_to_wav(src_path, dst)
            except Exception as e:
                print(f"  WARN: {src_path}: {e}")
        else:
            dst = os.path.join(config.RAW_DIR, f"{stem}.wav")
            try:
                utils.convert_to_wav(src_path, dst)
                print(f"  Converted: {os.path.basename(src_path)} → {stem}.wav")
            except Exception as e:
                print(f"  WARN: {src_path}: {e}")

    print(f"\nAll files prepared in: {config.RAW_DIR}")


if __name__ == "__main__":
    main()
