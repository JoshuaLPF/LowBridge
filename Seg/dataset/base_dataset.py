from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, TypeVar
# 类型提示，Callable 用于指定函数类型，Optional 表示一个值可以是某类型或 None，TypeVar 用于泛型编程。
import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import Literal  # 允许指定具体的常量类型（例如 "train" 或 "val"）

Stage = Literal["train", "val"]
FileFilter = Callable[[Path], Optional[str]]  # 定义了一个函数类型，它接受一个 Path 对象并返回一个字符串标识符（或 None），用于过滤文件。

Dataset_T = TypeVar("Dataset_T", bound="BaseDataset")  # 创建了一个泛型类型，用于表示 BaseDataset 类型的实例，可以用于方法签名中。


class BaseDataset(Dataset):
    """An image segmentation dataset that loads image from the disk lazily.

    Attributes:
        data_dir: the directory of the dataset
        stage: the stage of the dataset, "train" or "val"
        image_filter: the file filter to select images
        label_filter: the file filter to select labels
            A file filter should take a `Path` object and return an identifier
            (str) if the file is a valid image, otherwise None. The identifiers
            of image and label in the same pair should match.
        image_paths: the paths of the images
        label_paths: the paths of the labels
        path_loaded: whether the paths are loaded

    Loading Process:
        1. Paths of the images and labels are not loaded until `__getitem__` or
           `__len__` is called.
        2. Loading paths is done by `load_path` method. Files are selected by
           the file filters.
        3. Once the paths are loaded, they will be cached and will not change
           unless `load_path` is called again.
        4. Subclasses should implement `load_image_label` method to load image
           and label from a given path. `load_image_label` will be called by
           `__getitem__`.
    """
    def __init__(
        self,
        data_dir: str | Path,  # 数据集的根目录，可以是字符串或 Path 对象。
        domain: int,
        stage: Stage,
        image_filter: FileFilter | None = None,
        label_filter: FileFilter | None = None,
    ):
        self.data_dir = data_dir
        self.domain = domain
        self.stage = stage

        if image_filter is None or label_filter is None:
            raise ValueError("image_filter and label_filter cannot be None")

        self.image_filter = image_filter
        self.label_filter = label_filter
        self.image_paths = []  # 初始化为空列表，用于存储图像和标签的路径。
        self.label_paths = []  # 标记路径是否已加载
        self.path_loaded = False

        if stage not in ["train", "test", "val"]:
            raise ValueError(
                f"stage must be 'train', 'test' or 'val', got {stage}")

    def load_path(self, *args, **kwargs) -> None:
        """Load path info according to the filter and the data directory.

        """
        data_dir = Path(self.data_dir)
        self.image_paths = [
            p for p in data_dir.rglob("*")  # 通过 Path.rglob("*") 遍历数据目录中的所有文件
            if self.image_filter(p) and p.is_file()
        ]
        self.label_paths = [
            p for p in data_dir.rglob("*")
            if self.label_filter(p) and p.is_file()
        ]
        self.image_paths.sort()
        self.label_paths.sort()

        if len(self.image_paths) != len(self.label_paths):
            raise ValueError(
                "number of images and labels mismatch: {} != {}".format(
                    len(self.image_paths), len(self.label_paths)))
        # check identifier match
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            image_id = self.image_filter(image_path)
            label_id = self.label_filter(label_path)
            if image_id != label_id:
                raise ValueError("image identifier mismatch: {} != {}".format(
                    image_id, label_id))

        # check successfully loaded
        if len(self.image_paths) == 0:
            warnings.warn("load empty dataset paths from {}".format(
                self.data_dir))
        self.path_loaded = True

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:  # image 和 label 是 NumPy 数组。
        """Load image and label from the given paths."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:  # 获取数据项
        """(n_ch, h, w), (n_cl, h, w)"""
        if not self.path_loaded:
            self.load_path()
        image_path, label_path = (self.image_paths[index],
                                  self.label_paths[index])
        image, label = self.load_image_label(image_path, label_path)
        return image, label, image_path.name  # 图像、标签和图像的文件名。

    def __len__(self) -> int:  # 获取数据集大小
        if not self.path_loaded:
            self.load_path()
        return len(self.image_paths)

    def __repr__(self) -> str:
        parts = [
            self.__class__.__name__,
            f"Domain{self.domain}",
            self.stage,
        ]
        return "-".join(parts) + f"({len(self)})"  # 返回类的字符串表示，包含类名、域名和阶段，以及数据集的大小

    def random_split(  # 随机分割数据集
        self: Dataset_T,
        *frac: float,  # 可变参数，指定每个划分的比例。例如，frac=(0.8, 0.1, 0.1) 表示将数据集分成 80%、10%、10% 三部分。最后一部分会自动计算为剩余的部分。
        rounding: Callable[[float], int] = round,  # 可调用对象（例如 round），用于对每个部分的大小进行四舍五入
    ) -> tuple[list[Dataset_T], list[list[int]]]:
        """Random split the dataset into several parts.

        The `__class__` and attributes are copied to the new datasets
        except for the image_paths and label_paths.

        Args:
            frac: the fraction of each part, the last part will be the
                complement of the sum of the previous parts.
            rounding: a callable to round the number of the second part.

        Returns:
            (dataset, ..., indices): the datasets and the indices of each part.
        """
        def setup(dataset: BaseDataset, indices_: list[int]) -> None:
            """Setup the dataset and filter new paths with given indices."""
            # setup辅助函数用来根据给定的索引 indices_ 设置数据集。它根据这些索引筛选出新的 image_paths 和 label_paths，并将它们赋值给 dataset。
            dataset.image_paths = [self.image_paths[i] for i in indices_]
            dataset.label_paths = [self.label_paths[i] for i in indices_]

        n = len(self)  # 获取当前数据集的总长度
        indices = torch.randperm(n).tolist()  # 随机生成一个从 0 到 n-1 的随机排列，并将其转换为列表。这个列表用于随机打乱数据集的样本顺序。

        indices_split = []  # 列表存储划分后的索引子集。
        for f in frac:
            n_split = rounding(f * n)  # 对于每个 frac 中的比例 f，计算出该部分样本的数量
            indices_split.append(indices[:n_split])  # 将前 n_split 个索引添加到 indices_split 中。
            indices = indices[n_split:]  # 剩余的索引保存在 indices 中，继续处理下一个比例，直到所有的比例都处理完。
        indices_split.append(indices)  # 最后，将剩余的索引作为最后一个部分添加到 indices_split

        results = [deepcopy(self) for _ in range(len(indices_split))]
        # 创建 indices_split 长度的副本列表 results，每个副本是当前数据集 self 的深拷贝。
        [
            setup(dataset, indices_)
            for dataset, indices_ in zip(results, indices_split)
        ]  # 使用 zip 将每个副本 dataset 和相应的索引 indices_ 配对，通过 setup 函数为每个副本设置新的路径。
        return results, indices_split  # 返回 results（包含划分后的数据集）和 indices_split（包含每部分的索引）。

    def random_split_k(  # 将数据集随机分成两部分，第一部分包含 k 个样本，第二部分包含剩余的样本
        self: Dataset_T,
        k: int,
    ) -> tuple[list[Dataset_T], list[int]]:
        def setup(dataset: BaseDataset, indices_: list[int]) -> None:
            """Setup the dataset and filter new paths with given indices."""
            dataset.image_paths = [self.image_paths[i] for i in indices_]
            dataset.label_paths = [self.label_paths[i] for i in indices_]

        n = len(self)
        if k > n:
            raise ValueError(f"k must be less than {n}, got {k}")

        part1, part2 = deepcopy(self), deepcopy(self)
        indices = torch.randperm(n).tolist()  # 随机生成一个打乱的索引列表 indices，然后将前 k 个索引分配给 indices1（第一部分），剩余的索引分配给 indices2（第二部分）。
        indices1, indices2 = indices[:k], indices[k:]

        setup(part1, indices1)
        setup(part2, indices2)
        return [part1, part2], indices1  # 返回两个数据集（part1 和 part2）以及第一部分的索引 indices1。

# random_split 方法用于将数据集按照指定的比例随机划分为多个部分。每个子数据集是原始数据集的副本，但只包含一部分的路径。
# random_split_k 方法将数据集随机分成两部分，第一部分包含 k 个样本，第二部分包含剩余样本。