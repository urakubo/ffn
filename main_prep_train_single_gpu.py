#!/usr/bin/env python3
"""Prepare FFN data or train on one GPU using the shared FFN JSON."""
import argparse
import glob
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


STAGES = ('hdf5', 'compute_partitions', 'build_coordinates', 'train')
TRAIN_OPTIONS = set('''train_coords data_volumes label_volumes model_name model_args
train_dir batch_size max_steps summary_rate_secs seed_pad threshold fov_policy
fov_moves shuffle_moves image_mean image_stddev image_offset_scale_map
permutable_axes reflectable_axes optimizer learning_rate momentum adam_beta1
adam_beta2 epsilon rmsprop_decay learning_rate_decay_factor
decay_steps num_epochs_per_decay'''.split())


def flag(name, value):
    if isinstance(value, (dict, bool)):
        value = json.dumps(value, allow_nan=False, separators=(',', ':'))
    elif isinstance(value, list):
        value = ','.join(map(str, value))
    return f'--{name}={value}'


def volume_map(value):
    result = {}
    for item in value if isinstance(value, list) else value.split(','):
        tag, path, dataset = item.split(':', 2)
        if not all((tag, path, dataset)) or tag in result:
            raise ValueError(f'invalid or repeated volume entry: {item}')
        result[tag] = (path, dataset)
    return result


def write_hdf5(pattern, destination, dataset, labels, overwrite):
    import cv2
    import h5py
    import numpy as np

    files = sorted(path for path in glob.glob(pattern) if Path(path).is_file())
    if not files:
        raise ValueError(f'no input slices matched {pattern}')
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED if labels else cv2.IMREAD_GRAYSCALE)
              for path in files]
    if any(image is None or image.ndim != 2 for image in images):
        raise ValueError(f'could not read all slices as 2-D images: {pattern}')
    volume = np.stack(images)
    if labels:
        volume = volume.astype(np.int32)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, 'w' if overwrite else 'x') as output:
        output.create_dataset(dataset, data=volume, compression='gzip')
    if not labels:
        mean = volume.mean(dtype=np.float64)
        stddev = volume.std(dtype=np.float64)
        print(f'EM intensity {destination}:{dataset} '
              f'image_mean={mean:.6f} image_stddev={stddev:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=(*STAGES, 'prepare', 'all'))
    parser.add_argument('--volume', action='append', help='Preparation volume tag; repeat to select more')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite-hdf5', action='store_true')
    a = parser.parse_args()
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    options = config['options']
    prep = config.get('preparation', {})
    repo = Path(prep.get('repo_dir', 'repo'))
    env = {**os.environ, **{key: str(value) for key, value in
           config.get('environment', {}).items()}}
    if a.stage == 'train' and a.volume:
        parser.error('--volume selects preparation volumes only')
    stages = STAGES if a.stage == 'all' else STAGES[:3] if a.stage == 'prepare' else (a.stage,)

    def invoke(script, flags, gpu):
        command = [sys.executable, str(repo / script), *flags]
        print(f'CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {shlex.join(command)}', flush=True)
        if not a.dry_run:
            subprocess.run(command, env={**env, 'CUDA_VISIBLE_DEVICES': gpu}, check=True)

    if any(stage != 'train' for stage in stages):
        volumes = prep['volumes']
        selected = a.volume or list(volumes)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('select at least one preparation volume, with no duplicate tags')
        inputs = volume_map(options['data_volumes'])
        labels = volume_map(options['label_volumes'])
        for tag in selected:
            spec = volumes[tag]
            image_path, image_key = inputs[tag]
            label_path, label_key = labels[tag]
            partition = spec['partition_volume'] if any(
                stage in stages for stage in ('compute_partitions', 'build_coordinates')) else None
            if 'hdf5' in stages:
                outputs = ((spec['em_glob'], image_path, image_key, False),
                           (spec['seg_glob'], label_path, label_key, True))
                if not a.dry_run and not a.overwrite_hdf5:
                    for _, path, _, _ in outputs:
                        if Path(path).exists():
                            raise FileExistsError(f'{path} already exists; use --overwrite-hdf5 to replace')
                for pattern, path, key, is_label in outputs:
                    print(f'HDF5 [{tag}]: {pattern} -> {path}:{key}', flush=True)
                    if not a.dry_run:
                        write_hdf5(pattern, path, key, is_label, a.overwrite_hdf5)
            if 'compute_partitions' in stages:
                fields = {k: prep['compute_partitions'][k]
                          for k in ('thresholds', 'lom_radius', 'min_size')}
                fields.update(input_volume=f'{label_path}:{label_key}', output_volume=partition)
                if not a.dry_run:
                    Path(partition.rsplit(':', 1)[0]).parent.mkdir(parents=True, exist_ok=True)
                invoke('compute_partitions.py', [flag(k, v) for k, v in fields.items()], '')
            if 'build_coordinates' in stages:
                output = spec['coordinate_output']
                if not a.dry_run:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                invoke('build_coordinates.py', [flag('partition_volumes', f'{tag}:{partition}'),
                       flag('coordinate_output', output),
                       flag('margin', prep['build_coordinates']['margin'])], '')
    if 'train' in stages:
        gpu = str(config.get('single_gpu', {}).get('gpu', '0'))
        if not gpu or ',' in gpu:
            raise ValueError('single_gpu.gpu must select exactly one GPU')
        for key in ('train_coords', 'data_volumes', 'label_volumes', 'model_name',
                    'model_args', 'train_dir'):
            if not options.get(key):
                raise ValueError(f'options.{key} is required for training')
        flags = [flag(key, value) for key, value in options.items()
                 if key in TRAIN_OPTIONS and value is not None]
        invoke('train.py', flags, gpu)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except (OSError, KeyError, ValueError, TypeError) as error:
        sys.exit(f'FFN launcher error: {error}')
