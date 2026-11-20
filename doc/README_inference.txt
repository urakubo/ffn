FFN inference launcher

Place main_inference.py and FFNInference3.py in the same directory.
Run commands from the FFN working directory containing repo_urakubo/.
Relative paths in the JSON are resolved from the working directory.
Use the same Python venv that already runs your FFN code.

Required packages for preparation/export are numpy, h5py, opencv-python,
tifffile and Pillow. FFN inference uses the dependencies already installed
for repo_urakubo/run_inference.py. fastmorph is needed only for fill_hole=true.

Edit inference_config.json first. In particular, set target_image_dir,
target_image_file_h5, train_dir, model_args, ffn_inference_folder and the
output_segmentation_folder to your real paths and model settings.
The example uses the sparse-Z model and thresholds from the supplied launcher.

Run the stages independently

  python main_inference.py --config inference_config.json --stage prepare
  python main_inference.py --config inference_config.json --stage inference
  python main_inference.py --config inference_config.json --stage postprocess

Run them consecutively

  python main_inference.py --config inference_config.json --stage all

Replace an existing HDF5 explicitly

  python main_inference.py --config inference_config.json --stage prepare --overwrite-hdf5

Rerun inference for the configured region

  python main_inference.py --config inference_config.json --stage inference --clear-existing

Preview inference using an existing HDF5

  python main_inference.py --config inference_config.json --stage inference --clear-existing --dry-run

Preview export using an existing segmentation

  python main_inference.py --config inference_config.json --stage postprocess --dry-run

HDF5 preparation and coordinates

Input is a directory of 2-D slices. TIFF slices must be grayscale. For other
supported image types, RGB/RGBA input is converted to grayscale. Filenames are
sorted numerically within their text components, so z2 precedes z10.
The file contains raw image data in ZYX order, original pixel dtype, population
mean/stddev, computed mean/stddev, Start/End X/Y/Z, axis order and source names.
Statistics use all voxels of the stored volume and float64 accumulation.
The entire image stack is stored. Bounds select an inference ROI inside this
stack. Start coordinates are inclusive; End coordinates are exclusive.
For example, Start X=10 and End X=30 selects 20 voxels along X.
With null bounds, Start defaults to zero and End to the image dimensions.
Image filenames do not define an external/global voxel origin.

Normalization and bounds precedence

A non-null JSON value overrides HDF5 metadata. Otherwise inference reads
HDF5 metadata. For an older HDF5, missing bounds fall back to the array shape
and missing statistics are computed from its raw data. Older mean/std scalar
dataset names are also accepted.
options.image_mean and options.image_stddev use the same names as the training
JSON. They default to null in this inference example, selecting image-derived
normalization. To use the training normalization, enter its numeric values.
During preparation, manual values are saved alongside the computed statistics.
Later JSON changes take effect at inference time and are recorded in
inference_run.json. Existing HDF5 metadata is updated by rerunning preparation.
An automatically measured zero stddev is saved, but inference requires a
positive stddev. Set a positive manual value when using a constant image.

Checkpoint selection

options.train_dir is the training checkpoint directory.
inference.ckpt=null selects the highest numeric model.ckpt-N with a nonempty
index file and a complete nonempty data shard set. An incomplete highest
checkpoint is skipped. The .meta file is optional because FFN builds its graph.
Use a number such as 38420160 to select train_dir/model.ckpt-38420160.
An explicit checkpoint prefix string is also accepted, relative to the working
directory or absolute. Use the prefix without .index or .data suffixes.
This checks the file set; TensorFlow validates checkpoint contents on restore.

Existing inference output

Set inference.clear_existing=true or pass --clear-existing to delete outputs
for the selected Start X/Y/Z immediately before inference. The exact .npz,
.prob and .cpoint files are removed in both current x/y/seg-x_y_z locations
and the legacy flat location. Other regions and training checkpoints are kept.
The runner aborts with an explanation when matching output exists without
explicit clearing. Postprocess can read existing output independently.
Inference parameters and the selected checkpoint/bounds are recorded as
inference_params.pbtxt and inference_run.json in the output directory.

Postprocessing

FFN normally writes a compressed NPZ containing the segmentation array under
the segmentation key. Only that array is loaded; object-valued origins do not
require enabling pickle. For a standalone NPY, set postprocess.result_file
to its path. Set result_file=null to locate the NPZ for the configured region.
Input label arrays must have ZYX order. Every exported slice has YX order.
The optional fastmorph fill_hole operation uses the settings in the supplied
launcher. It is disabled in the example and changes segmentation labels when
enabled. It requires enough memory for a working copy of the label volume.

32-bit grayscale TIFF (the default)
  output_filetype="tif", coloring=false, label_dtype="uint32"

16-bit grayscale TIFF
  output_filetype="tif", coloring=false, label_dtype="uint16"

16-bit grayscale PNG
  output_filetype="png", coloring=false, label_dtype="uint16"

Color TIFF or PNG
  output_filetype="tif" or "png", coloring=true

RGB output uses uint8 YXC arrays, a stable ID-based color function and black
for background label zero. Colors are visualizations; the original NPZ/NPY
retains label IDs. Colors are not guaranteed unique for every ID.
Grayscale export preserves integer IDs and checks the maximum before casting.
It fails instead of truncating IDs above 65535 for uint16 or 4294967295 for
uint32. PNG does not support this launcher's uint32 grayscale output.
Result slices are named z000000.tif/png, using the ROI's Start Z when it is
derived from the HDF5. Explicit NPY/NPZ paths use zero-based slice filenames.
Export overwrites image files with the same names in the selected destination.

Validation

Tested with synthetic asymmetric image volumes and label arrays. Checks include
natural slice ordering, HDF5 dtype/statistics/metadata, manual overrides,
maximum-complete checkpoint selection, protobuf request parsing, scoped result
cleanup, grayscale uint16 PNG, grayscale uint32 TIFF, RGB TIFF/PNG, and overflow
rejection. CLI stages were also exercised with a simulated inference runner.
Actual GPU FFN inference and the optional fastmorph operation have not been
executed in this Windows workspace.

FFN source references

https://github.com/google/ffn/blob/master/run_inference.py
https://github.com/google/ffn/blob/master/ffn/inference/storage.py
https://github.com/google/ffn/blob/master/ffn/inference/inference.proto
