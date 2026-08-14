# Pixel Sharpness Regression

Small U-Net baseline that predicts an `H x W` score map in `[0, 1]`. Its two
network input channels are the Y luminance image and the variance map. The mask
is not a network channel: it filters both inputs before inference and limits the
loss and metrics to pixels whose mask value equals `1`.

## Environment

The code is compatible with Python 3.8, CUDA 11.0, and PyTorch 1.7.1+cu110.
Install the matching package versions with:

```powershell
python -m pip install -r requirements.txt
```

If PyTorch 1.7.1+cu110 is already installed correctly, it can be left in place;
the remaining versions in `requirements.txt` are selected for compatibility
with this older PyTorch/Python environment.

## Data layout

Files are matched by stem, for example all files for one sample use `0001`.

```text
data/
  train/
    y_images/*.png
    variances/*.npy
    masks/*.png
    labels/*.npy
  val/
    y_images/*.png
    variances/*.npy
    masks/*.png
    labels/*.npy
```

Y, variance, mask, and label must have exactly the same `H x W` shape. Class-1
label values must be finite and in `[0, 1]`. Class-1 variance values must be
finite and non-negative.

Before entering the network, Y is divided by 255 and variance is linearly
normalized as `clip(variance / variance_scale, 0, 1)`. Both channels are set to
zero outside `mask == 1`. The default variance scale is 7000 and can be changed
with `--variance-scale`; the selected value is saved in the checkpoint and
automatically reused during prediction. The previous logarithmic formula is
retained as commented code in both training and prediction data loaders.

## Extract Y channel

Extract the Y (luminance) channel from every JPG/JPEG under `train/images` and
`val/images` and save it losslessly as a single-channel PNG:

```powershell
python extract_y_channel.py --data-root D:\path\to\data
```

The default output layout is:

```text
data/
  train/y_images/*.png
  val/y_images/*.png
```

Files keep the input stem, for example `train/images/0001.jpg` becomes
`train/y_images/0001.png`. Existing files with the same name are overwritten.
To save under another root while preserving the split layout, use:

```powershell
python extract_y_channel.py --data-root D:\path\to\data `
  --output-root D:\path\to\y_output
```

## Train

From this directory:

```powershell
python train.py --data-root D:\path\to\data --epochs 100 --batch-size 1
```

For example, to use a different fixed variance scale:

```powershell
python train.py --data-root D:\path\to\data --variance-scale 6500
```

Every training and validation step prints the current and epoch-average loss,
cumulative MAE/RMSE, `Acc@0.05`, `Acc@0.10`, step time, elapsed time, and an ETA
for the complete run. The ETA is updated from measured step times and becomes
more stable after the first several steps.

The default uses full-resolution images. It applies horizontal/vertical flips and
rotations by multiples of 90 degrees. These are index-only transforms and do not
resample or blur the image. Validation has no augmentation.

Optional fixed-size random crops can reduce GPU memory usage and allow larger
batches, but they are disabled by default. Each crop is anchored on a randomly
selected class-1 pixel so it always contributes supervision:

```powershell
python train.py --data-root D:\path\to\data --crop-size 512 512 --batch-size 4
```

Use `--no-augment` to disable all training augmentation. Checkpoints `best.pt`
(lowest validation MAE) and `last.pt`, plus `history.csv`, are written under
`runs/small_unet` by default.

## TensorBoard

Training writes epoch-level train/validation curves for loss, MAE, RMSE,
`Acc@0.05`, and `Acc@0.10`, as well as the learning rate. The two accuracy
metrics are the proportions of valid class-1 pixels whose absolute prediction
error is at most `0.05` and `0.10` respectively.

All curves from one training process are written through a single TensorBoard
event writer and stored in one event file. Starting a new training process in
the same log directory creates a new event file, as required by TensorBoard.

Start TensorBoard from this directory while training or after training:

```powershell
tensorboard --logdir runs\small_unet\tensorboard --port 6006
```

Then open `http://localhost:6006`. A different log directory can be selected
with `--tensorboard-dir D:\path\to\logs`.

## Prediction

Predict one sample. The Y PNG, variance NPY, and mask PNG must have matching
dimensions; only `mask == 1` is retained in the saved full-size float32 NPY map:

```powershell
python predict.py --checkpoint runs\small_unet\best.pt `
  --y-input D:\data\test\y_images\0001.png `
  --variance D:\data\test\variances\0001.npy `
  --mask D:\data\test\masks\0001.png `
  --output-dir predictions
```

Predict every Y PNG in a directory. Variances and masks are matched by stem:

```powershell
python predict.py --checkpoint runs\small_unet\best.pt `
  --y-input D:\data\test\y_images `
  --variance D:\data\test\variances `
  --mask D:\data\test\masks `
  --output-dir predictions
```

This performs full-image inference without resizing or sliding windows. For a
`0001.png` Y input, the output is `predictions\0001.npy` with the original `H x W`
shape. Pixels outside class 1 are saved as zero.

## Visualization

Convert one prediction into a color PNG:

```powershell
python visualize.py --prediction predictions\0001.npy `
  --mask D:\data\test\masks\0001.png `
  --output-dir visualizations
```

Or process a complete prediction directory:

```powershell
python visualize.py --prediction predictions `
  --mask D:\data\test\masks `
  --output-dir visualizations
```

The visualization uses Matplotlib's standard `jet` colormap with fixed
`vmin=0` and `vmax=1`, matching `imshow(score, cmap="jet", vmin=0, vmax=1)`.
Therefore, the same score has the same color in every image. Non-class-1 pixels
are black. Output files use names such as `0001_color.png`.

Full `4096 x 3072` training can still require substantial GPU memory because
activations dominate memory use. If it does not fit, first reduce
`--base-channels` from `16` to `8`, or enable cropping explicitly.
