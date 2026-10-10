from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.segmentation_dataset import SegmentationDataset
from evaluate import evaluate
from model.unet import UNet
from utils import AverageMeter, DiceLoss, fix_seed, init_log
from utils.mask_convert import converter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fully supervised single-domain training")
    parser.add_argument("--config", default="configs/prostate.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--save-path", default="outputs/unet")
    parser.add_argument("--train-domain", type=int, required=True)
    parser.add_argument("--val-domain", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_optimizer(model: nn.Module, cfg: dict):
    name = cfg["optimizer"].lower()
    if name == "sgd":
        return SGD(model.parameters(), cfg["lr"], momentum=0.9)
    if name == "adamw":
        return AdamW(model.parameters(), cfg["lr"])
    if name == "adam":
        return Adam(model.parameters(), cfg["lr"])
    raise ValueError(f"Unsupported optimizer: {name}")


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8")))
    val_domain = args.train_domain if args.val_domain is None else args.val_domain
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(cfg.get("num_threads", 4))
    trainset = SegmentationDataset(cfg["dataset"], cfg["data_root"], args.train_domain,
                                   "train", cfg["image_size"], cfg.get("augment", True))
    valset = SegmentationDataset(cfg["dataset"], cfg["data_root"], val_domain,
                                 "val", cfg["image_size"])
    trainloader = DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True,
                             num_workers=cfg.get("num_workers", 2),
                             pin_memory=device.type == "cuda", drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1,
                           pin_memory=device.type == "cuda")
    if len(trainloader) == 0:
        raise ValueError("Training set is smaller than batch_size")
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = init_log("train", logging.INFO)
    writer = SummaryWriter(str(save_path))
    logger.info("Training domain %d: %s", args.train_domain, trainset)
    logger.info("Validation domain %d: %s", val_domain, valset)
    model = UNet(cfg["n_channels"], cfg["n_classes"], cfg.get("dropout", False)).to(device)
    optimizer = make_optimizer(model, cfg)
    criterion_ce, criterion_dice = nn.CrossEntropyLoss(), DiceLoss(cfg["n_classes"])
    convert = converter[cfg["dataset"]]
    start_epoch, best_dice = 0, 0.0
    latest_path = save_path / "latest.pth"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", checkpoint.get("previous_best", 0.0))
    total_steps = cfg["epochs"] * cfg["iters"]
    global_step = start_epoch * cfg["iters"]
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        losses, ce_losses, dice_losses = AverageMeter(), AverageMeter(), AverageMeter()
        iterator = iter(trainloader)
        for _ in range(cfg["iters"]):
            try:
                image, mask, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                image, mask, _, _ = next(iterator)
            image, mask = image.to(device), convert(mask.to(device))
            logits = model(image)
            ce_loss = criterion_ce(logits, mask)
            dice_loss = criterion_dice(logits, mask, softmax="softmax", onehot=True)
            loss = (ce_loss + dice_loss) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            lr = cfg["lr"] * max(0.0, 1 - global_step / total_steps) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses.update(loss.item()); ce_losses.update(ce_loss.item()); dice_losses.update(dice_loss.item())
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/ce_loss", ce_loss.item(), global_step)
            writer.add_scalar("train/dice_loss", dice_loss.item(), global_step)
        mean_dice, _, classwise_domains = evaluate(model, valloader, cfg)
        logger.info("Epoch %d/%d loss=%.4f ce=%.4f dice_loss=%.4f val_dice=%.2f",
                    epoch + 1, cfg["epochs"], losses.avg, ce_losses.avg, dice_losses.avg, mean_dice)
        writer.add_scalar("val/mean_dice", mean_dice, epoch)
        for class_index, value in enumerate(classwise_domains[0]):
            writer.add_scalar(f"val/class_{class_index}_dice", value, epoch)
        is_best = mean_dice > best_dice
        best_dice = max(best_dice, mean_dice)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "best_dice": best_dice, "train_domain": args.train_domain,
                      "val_domain": val_domain, "config": cfg}
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, save_path / "best.pth")
    writer.close()


if __name__ == "__main__":
    main()
