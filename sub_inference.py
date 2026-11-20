"""Image preparation, FFN inference, and label export for main_inference.py."""
import glob
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

import h5py
import numpy as np


BOUND_KEYS = ('Start X', 'Start Y', 'Start Z', 'End X', 'End Y', 'End Z')


def bounds_for(shape, overrides, stored=None):
    """Bounds are XYZ, array dimensions are ZYX, and ends are exclusive."""
    stored = stored or {}
    defaults = dict(zip(BOUND_KEYS, (0, 0, 0, *shape[::-1])))
    values = {}
    for key in BOUND_KEYS:
        value = overrides.get(key)
        if value is None:
            value = stored.get(key, defaults[key])
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{key} must be an integer or null')
        values[key] = int(value)
    for axis, limit in zip('XYZ', shape[::-1]):
        if not 0 <= values[f'Start {axis}'] < values[f'End {axis}'] <= limit:
            raise ValueError(f'Invalid {axis} bounds for axis length {limit}: {values}')
    return values


def read_slice(path):
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        image = tifffile.imread(path)
    else:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            code = cv2.COLOR_BGR2GRAY if image.shape[-1] == 3 else cv2.COLOR_BGRA2GRAY
            image = cv2.cvtColor(image, code)
    if image.ndim != 2 or image.dtype.kind not in 'uif':
        raise ValueError(f'Expected one 2-D grayscale slice in {path}, got {image.shape}')
    if not np.isfinite(image).all():
        raise ValueError(f'Non-finite image values in {path}')
    return image


def merge_stats(count, mean, m2, image):
    n = image.size
    image_mean = float(image.mean(dtype=np.float64))
    image_m2 = float(image.var(dtype=np.float64)) * n
    delta = image_mean - mean
    total = count + n
    return total, mean + delta * n / total, m2 + image_m2 + delta * delta * count * n / total


def prepare_target_image(config, overwrite=False):
    settings = config['inference']
    source = Path(settings['target_image_dir'])
    supported = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}
    files = [p for p in source.glob(settings.get('image_pattern', '*'))
             if p.is_file() and p.suffix.lower() in supported]
    # Numeric filename components are compared numerically (z2 precedes z10).
    files.sort(key=lambda p: [int(x) if x.isdigit() else x.lower()
                             for x in re.split(r'(\d+)', str(p))])
    if not files:
        raise ValueError(f'No supported image slices found in {source}')
    destination = Path(settings['target_image_file_h5'])
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} exists. Use --overwrite-hdf5 to replace it.')
    first = read_slice(files[0])
    shape = (len(files), *first.shape)
    bounds = bounds_for(shape, settings.get('bounds', {}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.h5.tmp')
    os.close(fd)
    temporary = Path(temporary)
    try:
        count, mean, m2 = 0, 0.0, 0.0
        with h5py.File(temporary, 'w') as output:
            raw = output.create_dataset(settings.get('hdf5_dataset', 'raw'), shape,
                                        dtype=first.dtype, compression='gzip',
                                        chunks=(1, min(first.shape[0], 256), min(first.shape[1], 256)))
            for z, path in enumerate(files):
                image = first if z == 0 else read_slice(path)
                if image.shape != first.shape or image.dtype != first.dtype:
                    raise ValueError(f'Slice shape/dtype differs at {path}')
                raw[z] = image
                count, mean, m2 = merge_stats(count, mean, m2, image)
            computed_stddev = math.sqrt(max(m2 / count, 0))
            options = config.get('options', {})
            effective_mean = options.get('image_mean')
            effective_stddev = options.get('image_stddev')
            effective_mean = mean if effective_mean is None else float(effective_mean)
            effective_stddev = computed_stddev if effective_stddev is None else float(effective_stddev)
            if not math.isfinite(effective_mean) or not math.isfinite(effective_stddev) or effective_stddev < 0:
                raise ValueError('Image normalization values must be finite; stddev must be nonnegative')
            for key, value in {**bounds, 'image_mean': effective_mean,
                               'image_stddev': effective_stddev,
                               'image_mean_computed': mean,
                               'image_stddev_computed': computed_stddev}.items():
                output.create_dataset(key, data=value)
            output.attrs['axis_order'] = 'ZYX'
            output.attrs['end_coordinates'] = 'exclusive'
            output.attrs['source_directory'] = str(source.resolve())
            output.create_dataset('source_files', data=np.asarray(
                [str(p) for p in files], dtype=h5py.string_dtype('utf-8')))
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == 'nt':
            # Windows rename refuses to replace an existing destination.
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
        print(f'HDF5 saved: {destination}; shape ZYX={shape}', flush=True)
        print(f'Computed image_mean={mean:.8f} image_stddev={computed_stddev:.8f}', flush=True)
        print(f'Effective image_mean={effective_mean:.8f} image_stddev={effective_stddev:.8f}', flush=True)
        print('Bounds:', bounds, flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def image_metadata(config):
    settings = config['inference']
    options = config.get('options', {})
    with h5py.File(settings['target_image_file_h5'], 'r') as source:
        raw = source[settings.get('hdf5_dataset', 'raw')]
        if raw.ndim != 3:
            raise ValueError(f'HDF5 image must have ZYX dimensions, got {raw.shape}')
        stored = {key: source[key][()].item() for key in BOUND_KEYS if key in source}
        bounds = bounds_for(raw.shape, settings.get('bounds', {}), stored)
        values = {}
        for key, legacy in [('image_mean', 'mean'), ('image_stddev', 'std')]:
            value = options.get(key)
            if value is None:
                dataset = key if key in source else legacy if legacy in source else None
                value = float(source[dataset][()]) if dataset else None
            values[key] = value
        if any(value is None for value in values.values()):
            count, mean, m2 = 0, 0.0, 0.0
            for z in range(raw.shape[0]):
                count, mean, m2 = merge_stats(count, mean, m2, raw[z])
            if values['image_mean'] is None:
                values['image_mean'] = mean
            if values['image_stddev'] is None:
                values['image_stddev'] = math.sqrt(max(m2 / count, 0))
    values = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(v) for v in values.values()) or values['image_stddev'] <= 0:
        raise ValueError('image_mean must be finite and image_stddev must be positive. '
                         'Set options.image_stddev for a constant image.')
    return bounds, values


def checkpoint_complete(prefix):
    prefix = Path(prefix)
    index = Path(str(prefix) + '.index')
    shards = list(prefix.parent.glob(prefix.name + '.data-*-of-*'))
    if not index.is_file() or index.stat().st_size == 0 or not shards:
        return False
    numbers = []
    totals = set()
    for shard in shards:
        match = re.fullmatch(re.escape(prefix.name) + r'\.data-(\d+)-of-(\d+)', shard.name)
        if not match or not shard.is_file() or shard.stat().st_size == 0:
            return False
        numbers.append(int(match[1]))
        totals.add(int(match[2]))
    return len(totals) == 1 and sorted(numbers) == list(range(next(iter(totals))))


def select_checkpoint(config):
    directory = Path(config['options']['train_dir'])
    chosen = config['inference'].get('ckpt')
    if chosen is not None:
        prefix = directory / f'model.ckpt-{chosen}' if str(chosen).isdigit() else Path(str(chosen))
        if not checkpoint_complete(prefix):
            raise FileNotFoundError(f'Checkpoint index/data shard set is incomplete: {prefix}')
        return prefix
    candidates = []
    for index in directory.glob('model.ckpt-*.index'):
        match = re.fullmatch(r'model\.ckpt-(\d+)\.index', index.name)
        if match:
            candidates.append((int(match[1]), Path(str(index)[:-6])))
    for step, prefix in sorted(candidates, reverse=True):
        if checkpoint_complete(prefix):
            print(f'Selected checkpoint step {step}: {prefix}', flush=True)
            return prefix
    raise FileNotFoundError(f'No checkpoint with a complete index/data shard set in {directory}')


def proto_text(fields, indent=''):
    lines = []
    for key, value in fields.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', key):
            raise ValueError(f'Invalid protobuf field name: {key}')
        if isinstance(value, dict):
            lines.extend([indent + key + ' {', proto_text(value, indent + '  '), indent + '}'])
        else:
            if isinstance(value, str):
                text = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, bool):
                text = str(value).lower()
            elif isinstance(value, (int, float)) and math.isfinite(value):
                text = str(value)
            else:
                raise ValueError(f'Unsupported protobuf value for {key}')
            lines.append(f'{indent}{key}: {text}')
    return '\n'.join(lines)


def result_paths(output_dir, bounds):
    x, y, z = (bounds['Start ' + axis] for axis in 'XYZ')
    root = Path(output_dir)
    name = f'seg-{x}_{y}_{z}'
    return [root / str(x) / str(y) / (name + suffix)
            for suffix in ('.npz', '.prob', '.cpoint')] + [
                root / (name + suffix) for suffix in ('.npz', '.prob', '.cpoint')]


class FFNInference:
    def _Run(self, config, clear_existing=False, dry_run=False):
        settings = config['inference']
        options = config['options']
        bounds, normalization = image_metadata(config)
        checkpoint = select_checkpoint(config)
        model_args = options['model_args']
        if isinstance(model_args, str):
            model_args = json.loads(model_args)
        if not isinstance(model_args, dict):
            raise ValueError('options.model_args must be an object or JSON object string')
        request = {
            'image': {'hdf5': settings['target_image_file_h5'] + ':' + settings.get('hdf5_dataset', 'raw')},
            **normalization,
            'model_name': options['model_name'],
            'model_args': json.dumps(model_args, separators=(',', ':'), allow_nan=False),
            'model_checkpoint_path': str(checkpoint),
            'segmentation_output_dir': settings['ffn_inference_folder'],
            'checkpoint_interval': settings.get('checkpoint_interval', 1800),
            'seed_policy': settings.get('seed_policy', 'PolicySparsePeaks'),
            'inference_options': settings['inference_options'],
        }
        if settings.get('seed_policy_args') is not None:
            seed_args = settings['seed_policy_args']
            if isinstance(seed_args, str):
                seed_args = json.loads(seed_args)
            request['seed_policy_args'] = json.dumps(seed_args, allow_nan=False)
        text = proto_text(request)
        box = ('start { ' + ' '.join(f'{a.lower()}: {bounds["Start " + a]}' for a in 'XYZ') +
               ' } size { ' + ' '.join(f'{a.lower()}: {bounds["End " + a] - bounds["Start " + a]}'
                                      for a in 'XYZ') + ' }')
        runner = Path(settings['repo_dir']) / 'run_inference.py'
        if not runner.is_file():
            raise FileNotFoundError(runner)
        command = [sys.executable, str(runner), '--inference_request', text, '--bounding_box', box]
        output = Path(settings['ffn_inference_folder'])
        existing = [p for p in result_paths(output, bounds) if p.exists()]
        print('Bounds:', bounds, flush=True)
        print('Normalization:', normalization, flush=True)
        print(shlex.join(command), flush=True)
        for path in existing:
            print('Existing result: ' + str(path), flush=True)
        if existing and not (clear_existing or settings.get('clear_existing', False)):
            raise FileExistsError('Existing FFN output found. Use --clear-existing to rerun this region.')
        if dry_run:
            return
        output.mkdir(parents=True, exist_ok=True)
        for path in existing:
            if not path.resolve().is_relative_to(output.resolve()) or not path.is_file():
                raise ValueError(f'Refusing to delete an unexpected path: {path}')
        for path in existing:
            path.unlink()
            print(f'Deleted: {path}', flush=True)
        (output / 'inference_params.pbtxt').write_text(text + '\n', encoding='utf-8')
        manifest = {'checkpoint': str(checkpoint), 'bounds': bounds, **normalization,
                    'request': request, 'status': 'started'}
        manifest_path = output / 'inference_run.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        environment = {**os.environ, **{key: str(value) for key, value in
                                       config.get('environment', {}).items()}}
        try:
            subprocess.run(command, env=environment, check=True)
        except BaseException:
            manifest['status'] = 'failed'
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            raise
        manifest['status'] = 'completed'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def labels_to_rgb(labels):
    """A YX label slice becomes YXC RGB. Zero is black; IDs retain stable colors."""
    x = labels.astype(np.uint64, copy=False)
    rgb = np.empty(x.shape + (3,), dtype=np.uint8)
    for channel, multiplier, shift, bias in ((0, 37, 5, 17), (1, 73, 9, 29), (2, 17, 3, 71)):
        rgb[..., channel] = ((x * np.uint64(multiplier)) ^ (x >> np.uint64(shift)) ^
                             np.uint64(bias)).astype(np.uint8)
    rgb[x == 0] = 0
    return rgb


def run_postprocess(config, dry_run=False):
    settings = config['inference']
    post = config['postprocess']
    path = post.get('result_file')
    bounds = None
    if path is None:
        bounds, _ = image_metadata(config)
        path = next((p for p in result_paths(settings['ffn_inference_folder'], bounds)
                     if p.suffix == '.npz' and p.is_file()), None)
        if path is None:
            raise FileNotFoundError('No segmentation NPZ found for the selected region')
    path = Path(path)
    data = np.load(path, allow_pickle=False, mmap_mode='r' if path.suffix == '.npy' else None)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            labels = data[post.get('segmentation_key', 'segmentation')]
    else:
        labels = data
    if labels.ndim != 3 or labels.dtype.kind not in 'uib' or labels.min() < 0:
        raise ValueError(f'Segmentation must contain nonnegative integer labels in ZYX order: {labels.shape}')
    if bounds:
        expected = tuple(bounds['End ' + a] - bounds['Start ' + a] for a in 'ZYX')
        if labels.shape != expected:
            raise ValueError(f'Segmentation shape {labels.shape} differs from ROI ZYX {expected}')
    extension = post.get('output_filetype', 'tif').lower().lstrip('.')
    extension = 'tif' if extension == 'tiff' else extension
    if extension not in ('tif', 'png'):
        raise ValueError('output_filetype must be tif or png')
    coloring = post.get('coloring', False)
    if not isinstance(coloring, bool) or not isinstance(post.get('fill_hole', False), bool):
        raise ValueError('coloring and fill_hole must be JSON booleans')
    dtype_name = post.get('label_dtype', 'uint32')
    if dtype_name not in ('uint16', 'uint32'):
        raise ValueError('label_dtype must be uint16 or uint32')
    maximum = int(labels.max())
    if not coloring and maximum > np.iinfo(dtype_name).max:
        raise ValueError(f'Maximum label ID {maximum} exceeds {dtype_name}; no image files were written')
    if not coloring and extension == 'png' and dtype_name != 'uint16':
        raise ValueError('Use uint16 for grayscale PNG or use TIFF for uint32 labels')
    destination = Path(post['output_segmentation_folder'])
    print(f'Segmentation: {path}; shape ZYX={labels.shape}; max ID={maximum}', flush=True)
    print(f'Export: {destination}; format={extension}; coloring={coloring}; label_dtype={dtype_name}', flush=True)
    if dry_run:
        return
    if post.get('fill_hole', False):
        import fastmorph
        labels = fastmorph.fill_holes(np.array(labels, copy=True), fix_borders=True,
                                     morphological_closing=True, remove_enclosed=True)
    destination.mkdir(parents=True, exist_ok=True)
    start_z = bounds['Start Z'] if bounds else 0
    for z in range(labels.shape[0]):
        image = labels_to_rgb(labels[z]) if coloring else labels[z].astype(dtype_name)
        filename = destination / f'z{start_z + z:06d}.{extension}'
        if extension == 'tif':
            import tifffile
            tifffile.imwrite(filename, image, photometric='rgb' if coloring else 'minisblack',
                             metadata={'axes': 'YXC' if coloring else 'YX'})
        else:
            from PIL import Image
            Image.fromarray(image).save(filename)
    print(f'Saved {labels.shape[0]} slices to {destination}', flush=True)
