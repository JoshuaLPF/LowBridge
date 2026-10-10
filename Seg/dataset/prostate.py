from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .base_dataset import BaseDataset, Stage


class Prostate(BaseDataset):  # 该数据集的功能和操作是建立在 BaseDataset 上的。
    """Prostate dataset. See https://liuquande.github.io/SAML/."""

    names = ["MR", "CT"] # domain 0, domain 1

    def __init__(self, root: str | Path, domain: int, stage: Stage):
        sub_dir = "train" if stage == "train" else "test"
        BaseDataset.__init__(
            self,
            data_dir=Path(root) / f"{self.names[domain]}" / sub_dir,
            domain=domain,
            stage=stage,
            image_filter=Prostate._prostate_image_filter,
            label_filter=Prostate._prostate_label_filter,
        )

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        image, label = np.load(str(image_path)), np.load(str(label_path))
        # [-1, 1] to [0, 1]
        image = (image + 1) / 2  # 图像的值范围是 [-1, 1]，通过 (image + 1) / 2 转换到 [0, 1] 范围。

        ## cardiac
        label = np.stack(
            [
                label == 1,  # index 0: myocardium of the left ventricle (Myo)  liver
                label == 2,  # index 1: left atrium blood cavity (LAC)          RK
                label == 3,  # index 2: left ventricle blood cavity (LVC)       LK
                label == 4,  # index 3: ascending aorta (AA)                    SP
            ],
            axis=0,
        ).astype(np.uint8)  # np.stack 用于将这两个布尔数组堆叠在一起形成一个新的数组，axis=0 表示在第一个轴上堆叠，结果是一个 2 通道的标签图。
        return image, label

        # ## liver
        # label = np.stack(
        #     [
        #         label == 1,  # index 0: myocardium of the left ventricle (Myo)  liver
        #     ],
        #     axis=0,
        # ).astype(np.uint8)  # np.stack 用于将这两个布尔数组堆叠在一起形成一个新的数组，axis=0 表示在第一个轴上堆叠，结果是一个 2 通道的标签图。
        # return image, label



        # return image, label[np.newaxis, ...]  # 标签使用 np.newaxis 在标签的第一个维度上增加一个新维度，使标签数据的形状符合预期。

    @staticmethod
    def _prostate_image_filter(path: Path) -> str | None:
        volume, slice, type = path.stem.rsplit("_")
        if path.suffix == ".npy" and type == "image":
            return volume + "_" + slice

    @staticmethod
    def _prostate_label_filter(path: Path) -> str | None:
        volume, slice, type = path.stem.rsplit("_")
        if path.suffix == ".npy" and type == "label":
            return volume + "_" + slice

    def random_split(  # 用于按比例随机划分数据集。接收若干比例 frac（例如，0.7、0.3）以及一个用于四舍五入的函数 rounding（默认为 round）。
        self: Prostate,
        *frac: float,
        rounding: Callable[[float], int] = round,
    ) -> tuple[list[Prostate], list[list[int]]]:
        def setup(dataset: Prostate, indices_: list[int]) -> None:
            """Setup the dataset and filter new paths with given indices."""
            dataset.image_paths = [self.image_paths[i] for i in indices_]
            dataset.label_paths = [self.label_paths[i] for i in indices_]

        n = len(self)  # for data loading

        # get all volume_id
        all_vols = set(int(p.stem.split("_")[0]) for p in self.image_paths)
        n_vols = len(all_vols)
        vol_indices = torch.randperm(n_vols).tolist()
        # 获取所有图像的 volume_id，并将其打乱（使用 torch.randperm），然后根据 frac 中的比例将这些图像按子集划分

        indices_split = []  # 每个子集的索引存储在 indices_split 中，并最终返回拆分后的数据集和索引。
        for f in frac:
            n_split = rounding(f * n_vols)
            vol_split = vol_indices[:n_split]
            vol_indices = vol_indices[n_split:]
            # get the indices that startswith any volume_id in vol_split
            indices = [
                i for i, p in enumerate(self.image_paths)
                if int(p.stem.split("_")[0]) in vol_split
            ]
            indices_split.append(indices)

        indices = [
            i for i, p in enumerate(self.image_paths)
            if int(p.stem.split("_")[0]) in vol_indices
        ]
        indices_split.append(indices)

        results = [deepcopy(self) for _ in range(len(indices_split))]

        [
            setup(dataset, indices_)
            for dataset, indices_ in zip(results, indices_split)
        ]
        return results, indices_split

    def random_split_k(
        self: Prostate,
        k: int,
    ) -> tuple[list[Prostate], list[int]]:
        raise NotImplementedError
