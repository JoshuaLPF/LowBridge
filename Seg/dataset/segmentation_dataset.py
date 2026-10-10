from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .prostate import Prostate
from .transform import random_rot_flip, random_rotate


DATASETS = {"prostate": Prostate}


class SegmentationDataset(Dataset):
    """A labeled, single-domain segmentation dataset."""

    def __init__(self, name: str, root: str | Path, domain: int,
                 mode: Literal["train", "val"], image_size: int,
                 augment: bool = False) -> None:
        name = name.lower()
        if name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {name}")
        self.domain = domain
        self.mode = mode
        self.image_size = image_size
        self.augment = augment and mode == "train"
        self.dataset = DATASETS[name](root, domain, mode)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask, path = self.dataset[index]
        image, mask = image.transpose(1, 2, 0), mask.transpose(1, 2, 0)
        if self.augment:
            transform = random.choice((None, random_rot_flip, random_rotate))
            if transform is not None:
                image, mask, *_ = transform(image, mask)
        image = zoom(image, (self.image_size / image.shape[0],
                             self.image_size / image.shape[1], 1),
                     order=1).transpose(2, 0, 1)
        mask = zoom(mask, (self.image_size / mask.shape[0],
                           self.image_size / mask.shape[1], 1),
                    order=0).transpose(2, 0, 1)
        return (torch.from_numpy(image.copy()).float(),
                torch.from_numpy(mask.copy()).float(), self.domain, path)

    def __repr__(self) -> str:
        return repr(self.dataset)
