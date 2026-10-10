# RecNet

RecNet is a PyTorch image-to-image reconstruction model built with a CDFFormer-M36 encoder and a convolutional decoder.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version when GPU acceleration is required.

## Data layout

The default training paths are relative to the repository:

```text
train_datasets/
└── MR/
    ├── Edge/          # training inputs
    ├── Img/           # training targets
    └── val/
        ├── Edge/      # validation inputs
        └── Img/       # validation targets
```

Input and target filenames must sort into matching pairs. Supported extensions are `.jpg`, `.jpeg`, and `.png`.

## Training

```bash
python train_rec.py
```

Paths and hyperparameters can be overridden from the command line:

```bash
python train_rec.py \
  --train-images /path/to/train/inputs \
  --train-targets /path/to/train/targets \
  --val-images /path/to/validation/inputs \
  --val-targets /path/to/validation/targets \
  --output-dir ./ckpts
```

Run `python train_rec.py --help` for all options. The pretrained CDFFormer weights are downloaded by PyTorch on first use. Pass `--no-pretrained-backbone` to train without them.

## Inference

```bash
python test.py \
  --checkpoint ./ckpts/RecNet_epoch_best.pth \
  --test-images /path/to/test/inputs \
  --test-targets /path/to/test/targets \
  --output-dir ./test_maps
```

Predicted grayscale maps are written to the output directory. Run `python test.py --help` for all options.

## Acknowledgements

The encoder implementation is adapted from [CDFFormer](https://github.com/okojoalg/dfformer). Its original Apache 2.0 notice is retained in `model/dfformer.py`.
