from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import asd, dc, hd95
from torch.utils.data import DataLoader
from typing_extensions import TypeVar

from utils.mask_convert import converter, pred_mask, to_image, to_label

from torchvision.transforms.functional import to_pil_image
from PIL import Image

Collector = TypeVar("Collector", bound="ResultCollector")


class ResultCollector(ABC):

    def __init__(self, metric: str, n_classes: int, n_domains: int):
        self.metric = metric
        self.n_classes = n_classes
        self.n_domains = n_domains

    @abstractmethod
    def update(self, instance_metric_per_cls, domain_id):
        raise NotImplementedError

    @abstractmethod
    def get(self):
        raise NotImplementedError


class VolumewiseCollector(ResultCollector):

    def __init__(self, metric: str, n_classes: int, n_domains: int):
        super().__init__(metric, n_classes, n_domains)
        self.domain_instances: list[list[list[float]]] = [
            [] for _ in range(n_domains)
        ]  # (domain, instance, class)

    @property
    def averaged_instances(self) -> list[list[float]]:
        """Return the instance metric averaged over all classes."""
        return [[
            sum(inst) / len(inst) if len(inst) > 0 else 0 for inst in instances
        ] for instances in self.domain_instances]

    def update(self, instance_metric_per_cls, domain_id):
        k = 100 if self.metric.lower() in ["dice", "iou", "jaccard"] else 1
        self.domain_instances[domain_id].append(
            [v * k for v in instance_metric_per_cls])

    def get(self):
        sum_domain_class = [[
            sum(inst[c] for inst in instances) for c in range(self.n_classes)
        ] for instances in self.domain_instances]
        count_domain = [len(v) for v in self.domain_instances]

        metric_domain_classwise = []
        metric_domain_mean = []
        mean_metric_sum = 0
        mean_metric_cnt = 0

        for sum_class, cnt in zip(sum_domain_class, count_domain):
            if cnt == 0:
                continue

            classwise = [v / cnt for v in sum_class]
            mean = sum(classwise) / len(classwise)

            metric_domain_classwise.append(classwise)
            metric_domain_mean.append(mean)
            mean_metric_sum += sum(classwise)
            mean_metric_cnt += len(classwise)

        mean_dice = mean_metric_sum / mean_metric_cnt
        return mean_dice, metric_domain_mean, metric_domain_classwise


def evaluate_volume_metrics(
    model,
    dataloader: DataLoader,
    cfg: dict,
    verbose: bool = False,
    eval: bool = True,
    save_dir: Path | None = None,
    collector_cls: type[Collector] = VolumewiseCollector,
) -> list[Collector]:
    if eval:
        model.eval()

    pred_fn = model

    current_volume = None
    num_volumes = 0

    if cfg["dataset"] == "fundus":

        # fundus is 2D dataset, treat each image as a volume
        def is_next_volume(path):  # type: ignore
            return True

    elif cfg["dataset"] == "mnms":
        # must be careful with this dataset
        # vendorB contains 2 centers, whose patient ids are overlapped
        # however, the volume ids are ascending for each center
        # and we visit the dataset center by center
        # so the volume can be correctly identified by the first 3 digits
        def is_next_volume(path: str):
            nonlocal current_volume
            nonlocal num_volumes
            # {patient_id:03d}{slice_id:03d}
            if current_volume != path[:3]:
                current_volume = path[:3]
                num_volumes += 1
                return True
            return False

    elif cfg["dataset"] == "prostate":

        def is_next_volume(path: str):
            nonlocal current_volume
            nonlocal num_volumes
            # {volume}_{slice}_{type}
            vol = path.split("_")[0]
            if current_volume != vol:
                current_volume = vol
                num_volumes += 1
                return True
            return False
    else:
        raise ValueError(f"Unknown dataset {cfg['dataset']}")

    ret = evaluate_with_pred_function_volume(pred_fn,
                                             dataloader,
                                             cfg,
                                             is_next_volume,
                                             save_dir,
                                             collector_cls=collector_cls)
    if verbose:
        print(num_volumes)
    return ret


def evaluate_slice_metrics(
    model,
    dataloader: DataLoader,
    cfg: dict,
    eval: bool = True,
    save_dir: Path | None = None,
    penalty: int | None = None,
):
    if eval:
        model.eval()

    pred_fn = model

    # treat each slice as a volume, i.e., slice-wise evaluation
    def is_next_volume(path):
        return True

    return evaluate_with_pred_function_volume(
        pred_fn,
        dataloader,
        cfg,
        is_next_volume,
        save_dir,
        collector_cls=VolumewiseCollector,
        dist_penalty=penalty)


@torch.no_grad()
def evaluate_with_pred_function_volume(
    pred_fn,
    dataloader: DataLoader,
    cfg: dict,
    is_next_volume: Callable[[str], bool],
    save_dir: Path | None = None,
    dist_penalty: int | None = None,
    collector_cls: type[Collector] = VolumewiseCollector,
) -> list[Collector]:
    # pred_fn: image tensor -> logits tensor

    convert = converter[cfg["dataset"]]

    n_domains = cfg["n_domains"]
    n_classes = cfg["n_classes"] - 1  # exclude background
    size = cfg["image_size"]

    metrics = [("dice", dc, {}), ("asd", asd, {}), ("hd", hd95, {})]
    results = [
        collector_cls(metric, n_classes, n_domains) for metric, *_ in metrics
    ]
    dist_metric = ["asd", "assd", "hd"]
    dist_penalty = dist_penalty or 2

    current_volume_pred = []
    current_volume_mask = []
    domain = -1
    num_samples = 0

    # save as png
    if save_dir is not None:
        input_dir = save_dir / "img"
        truth_dir = save_dir / "gt"
        pred_dir = save_dir / "pred"
        input_dir.mkdir(exist_ok=True)
        truth_dir.mkdir(exist_ok=True)
        pred_dir.mkdir(exist_ok=True)

    device = next(pred_fn.parameters()).device
    for img, mask, domain, path in dataloader:
        img, mask = img.to(device), mask.to(device)
        domain = domain.item()

        h, w = img.shape[-2:]
        # down
        if h != size or w != size:
            img = F.interpolate(img, (size, size),
                                mode="bilinear",
                                align_corners=False)
        # pred
        pred = pred_fn(img)
        # up
        if h != size or w != size:
            pred = F.interpolate(pred, (h, w),
                                 mode="bilinear",
                                 align_corners=False)

        if is_next_volume(path[0]) and len(current_volume_pred) > 0:
            # update current volume to dice_sum_class_domain
            vol_pred = torch.cat(current_volume_pred, dim=0)
            vol_mask = torch.cat(current_volume_mask, dim=0)
            result = np.zeros((len(metrics), n_classes))
            for cls in range(n_classes):
                p, m = pred_mask[cfg["dataset"]](vol_pred, vol_mask, cls)
                p = p.cpu().numpy().astype(np.bool8)
                m = m.cpu().numpy().astype(np.bool8)
                for i, (metric, fn, kwargs) in enumerate(metrics):
                    if metric not in dist_metric:
                        result[i, cls] = fn(p, m, **kwargs)
                    else:
                        if p.sum() == 0 and m.sum() == 0:
                            result[i, cls] = 0
                        elif p.sum() == 0 or m.sum() == 0:
                            result[i, cls] = dist_penalty
                        else:
                            result[i, cls] = fn(p, m, **kwargs)

            for i, _ in enumerate(metrics):
                results[i].update(result[i], domain)

            # clear current volume
            current_volume_pred = []
            current_volume_mask = []

        pred = pred.argmax(dim=1)
        mask = convert(mask)
        current_volume_pred.append(pred)
        current_volume_mask.append(mask)
        if save_dir is not None:
            im = to_image[cfg["dataset"]](img[0])


            # For cardiac, pred和gt保存为灰度图
            lb = to_label(mask[0], n_classes)
            pd = to_label(pred[0], n_classes)

            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im)
            im_pil = to_pil_image(im.cpu())
            im_pil.save(input_dir / f"{num_samples:04d}.png")

            lb_tensor = torch.from_numpy(lb).unsqueeze(0).to(torch.uint8)
            lb_pil = to_pil_image(lb_tensor)
            lb_pil.save(truth_dir / f"{num_samples:04d}.png")
            pd_tensor = torch.from_numpy(pd).unsqueeze(0).to(torch.uint8)
            pd_pil = to_pil_image(pd_tensor)
            pd_pil.save(pred_dir / f"{num_samples:04d}.png")

            # save as color
            if cfg["dataset"] == "prostate":
                im_kwargs = {"cmap": "gray"}
            else:
                im_kwargs = {}
            # save as png
            plt.imsave(input_dir / f"{num_samples:04d}.png", im, **im_kwargs)
            plt.imsave(truth_dir / f"{num_samples:04d}.png", lb)
            plt.imsave(pred_dir / f"{num_samples:04d}.png", pd)




            # ## For liver，pred和gt保存为二值图
            # # 将二值图数据转换为 PIL 图像并保存
            # # ## For liver，pred和gt保存为二值图
            # # # 将 lb 和 pd 转换为二值图（0 或 255）
            # lb = mask[0].cpu().numpy()  # 保存单通道图
            # pd = pred[0].cpu().numpy()
            # lb_binary = (lb > 0).astype(np.uint8) * 255
            # pd_binary = (pd > 0).astype(np.uint8) * 255
            #
            # lb_pil = Image.fromarray(lb_binary, mode='L')
            # lb_pil.save(truth_dir / f"{num_samples:04d}.png")
            # pd_pil = Image.fromarray(pd_binary, mode='L')
            # pd_pil.save(pred_dir / f"{num_samples:04d}.png")





        num_samples += 1

    # update last volume
    if len(current_volume_pred) > 0:
        vol_pred = torch.cat(current_volume_pred, dim=0)
        vol_mask = torch.cat(current_volume_mask, dim=0)
        result = np.zeros((len(metrics), n_classes))
        for cls in range(n_classes):
            p, m = pred_mask[cfg["dataset"]](vol_pred, vol_mask, cls)
            p = p.cpu().numpy()
            m = m.cpu().numpy()
            for i, (metric, fn, kwargs) in enumerate(metrics):
                if metric not in dist_metric:
                    result[i, cls] = fn(p, m, **kwargs)
                else:
                    if p.sum() == 0 and m.sum() == 0:
                        result[i, cls] = 0
                    elif p.sum() == 0 or m.sum() == 0:
                        result[i, cls] = dist_penalty
                    else:
                        result[i, cls] = fn(p, m, **kwargs)

        for i, _ in enumerate(metrics):
            results[i].update(result[i], domain)

        # clear current volume
        current_volume_pred = []
        current_volume_mask = []

    # get the final result
    return results
