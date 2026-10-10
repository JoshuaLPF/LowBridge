# Single-domain supervised segmentation

This repository trains a U-Net on one labeled domain with the average of
cross-entropy loss and Dice loss. Training, validation, and testing domains are
selected explicitly from the command line.

Domain IDs for the current cardiac dataset are `0=MR` and `1=CT`. Update
`data_root` in `configs/prostate.yaml` before running the commands below.

## Train

```bash
python train.py --train-domain 0 --val-domain 0 --save-path outputs/mr_unet
```

`--val-domain` defaults to the training domain. Use `--resume` to continue from
`latest.pth` in the selected output directory.

## Test

Select the checkpoint produced during training and the domain to evaluate:

```bash
python infer.py --checkpoint outputs/mr_unet/best.pth --test-domain 1
```

To also save input, label, and prediction images:

```bash
python infer.py --checkpoint outputs/mr_unet/best.pth --test-domain 1 \
  --save-dir outputs/mr_unet/test_ct
```
