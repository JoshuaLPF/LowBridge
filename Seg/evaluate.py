from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.mask_convert import converter, pred_mask


def evaluate(model, dataloader: DataLoader, cfg: dict, **_) -> tuple[float, list[float], list[list[float]]]:
    model.eval()
    current_volume = None

    if cfg["dataset"] == "fundus":
        def is_next_volume(path: str) -> bool:
            return True
    elif cfg["dataset"] == "mnms":
        def is_next_volume(path: str) -> bool:
            nonlocal current_volume
            volume = path[:3]
            changed = current_volume != volume
            current_volume = volume
            return changed
    elif cfg["dataset"] == "prostate":
        def is_next_volume(path: str) -> bool:
            nonlocal current_volume
            volume = path.split("_")[0]
            changed = current_volume != volume
            current_volume = volume
            return changed
    else:
        raise ValueError(f"Unknown dataset {cfg['dataset']}")
    return evaluate_volumes(model, dataloader, cfg, is_next_volume)


@torch.no_grad()
def evaluate_volumes(model, dataloader: DataLoader, cfg: dict,
                     is_next_volume: Callable[[str], bool]):
    device = next(model.parameters()).device
    convert = converter[cfg["dataset"]]
    n_classes = cfg["n_classes"] - 1
    sums = [[0.0] * n_classes for _ in range(cfg["n_domains"])]
    counts = [0] * cfg["n_domains"]
    volume_pred, volume_mask = [], []
    volume_domain = -1

    def accumulate() -> None:
        if not volume_pred:
            return
        pred, mask = torch.cat(volume_pred), torch.cat(volume_mask)
        for cls in range(n_classes):
            p, m = pred_mask[cfg["dataset"]](pred, mask, cls)
            sums[volume_domain][cls] += (2 * (p * m).sum().item() + 1e-4) / (
                p.sum().item() + m.sum().item() + 1e-4)
        counts[volume_domain] += 1

    for image, mask, domain, path in dataloader:
        image, mask = image.to(device), mask.to(device)
        h, w = image.shape[-2:]
        if (h, w) != (cfg["image_size"], cfg["image_size"]):
            image = F.interpolate(image, (cfg["image_size"], cfg["image_size"]),
                                  mode="bilinear", align_corners=False)
        logits = model(image)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
        if is_next_volume(path[0]) and volume_pred:
            accumulate(); volume_pred.clear(); volume_mask.clear()
        volume_domain = int(domain.item())
        volume_pred.append(logits.argmax(1))
        volume_mask.append(convert(mask))
    accumulate()

    classwise = [[value * 100 / count for value in values]
                 for values, count in zip(sums, counts) if count]
    domain_means = [sum(values) / len(values) for values in classwise]
    mean = sum(map(sum, classwise)) / sum(map(len, classwise))
    return mean, domain_means, classwise


evaluate_slice = evaluate
