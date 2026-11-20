#!/usr/bin/env python3
"""Run FFN preparation, inference, or postprocessing from an inference JSON."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from sub_inference import FFNInference, prepare_target_image, run_postprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'inference', 'postprocess', 'all'], required=True)
    parser.add_argument('--overwrite-hdf5', action='store_true')
    parser.add_argument('--clear-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Preview inference or export using an existing HDF5/result')
    args = parser.parse_args()
    if args.dry_run and args.stage in ('prepare', 'all'):
        parser.error('--dry-run requires --stage inference or postprocess')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.stage in ('prepare', 'all'):
        prepare_target_image(config, overwrite=args.overwrite_hdf5)
    if args.stage in ('inference', 'all'):
        FFNInference()._Run(config, clear_existing=args.clear_existing, dry_run=args.dry_run)
    if args.stage in ('postprocess', 'all'):
        run_postprocess(config, dry_run=args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f'FFN inference launcher error: {error}')
