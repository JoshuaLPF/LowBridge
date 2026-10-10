from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.segmentation_dataset import SegmentationDataset
from evaluate_metrics import evaluate_volume_metrics
from model.unet import UNet
from utils import fix_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a selected domain")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/prostate.yaml")
    parser.add_argument("--test-domain", type=int, required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testset = SegmentationDataset(cfg["dataset"], cfg["data_root"],
                                  args.test_domain, "val", cfg["image_size"])
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1,
                        pin_memory=device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = UNet(cfg["n_channels"], cfg["n_classes"], cfg.get("dropout", False))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    collectors = evaluate_volume_metrics(model, loader, cfg, save_dir=save_dir, verbose=True)
    print(f"domain={args.test_domain} checkpoint={args.checkpoint}")
    for collector in collectors:
        mean, _, domain_classwise = collector.get()
        classwise = domain_classwise[0]
        print(f"[{collector.metric}] Mean: {mean:.4f}; Class: " +
              ",".join(f"{value:.4f}" for value in classwise))


if __name__ == "__main__":
    main()
