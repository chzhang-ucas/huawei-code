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

## TensorBoard

Training writes epoch-level train/validation curves for loss, MAE, RMSE,
`Acc@0.05`, and `Acc@0.10`, as well as the learning rate. The two accuracy
metrics are the proportions of valid class-1 pixels whose absolute prediction
error is at most `0.05` and `0.10` respectively.

Start TensorBoard from this directory while training or after training:

```powershell
tensorboard --logdir runs\small_unet\tensorboard --port 6006
```

Then open `http://localhost:6006`. A different log directory can be selected
with `--tensorboard-dir D:\path\to\logs`.

Full `4096 x 3072` training can still require substantial GPU memory because
activations dominate memory use. If it does not fit, first reduce
`--base-channels` from `16` to `8`, or enable cropping explicitly.
