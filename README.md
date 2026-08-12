# Pixel Sharpness Regression

Small U-Net baseline that predicts an `H x W` score map in `[0, 1]`. Only pixels
whose PNG mask value equals `1` contribute to the loss and validation metrics.

## Data layout

Files are matched by stem, for example `0001.jpg`, `0001.png`, `0001.npy`.

```text
data/
  train/
    images/*.jpg
    masks/*.png
    labels/*.npy
  val/
    images/*.jpg
    masks/*.png
    labels/*.npy
```

Each mask and NPY array must have exactly the same height and width as its image.
Class-1 NPY values must be finite and in `[0, 1]`; values outside class 1 are
ignored even though they are normally zero.

## Train

From this directory:

```powershell
python train.py --data-root D:\path\to\data --epochs 100 --batch-size 1
```

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

Full `4096 x 3072` training can still require substantial GPU memory because
activations dominate memory use. If it does not fit, first reduce
`--base-channels` from `16` to `8`, or enable cropping explicitly.
