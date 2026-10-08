# RecNet

PyTorch implementation of a reconstruction network with a CDFFormer-M36 encoder.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Data layout

Training and validation images are paired by sorted file name. Supported image extensions are `.jpg` and `.png`.

```text
train_datasets/
  images/
  targets/
test_datasets/
  images/
  targets/
```

For inference, each dataset uses this layout:

```text
test_datasets/
  MR/
    Edge/
    GT/
```

## Train

```bash
python train_rec.py
```

All paths can be overridden from the command line. Run `python train_rec.py --help` for available options. Checkpoints and TensorBoard logs are written to `cpts/` by default.

## Test

```bash
python test.py --checkpoint cpts/RecNet_epoch_best.pth
```

Predictions are written to `outputs/`. Use `--test_path`, `--output_dir`, and `--device cpu` when needed.

## Pretrained encoder

`model/dfformer.py` is adapted from the Apache-2.0-licensed DFFFormer implementation. RecNet disables automatic pretrained-weight downloads so the repository runs predictably offline.
