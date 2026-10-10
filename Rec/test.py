import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import TestDataset
from model import OurNet
from options import build_test_parser


def load_checkpoint(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def main():
    args = build_test_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = OurNet(pretrained_backbone=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    loader = TestDataset(args.test_images, args.test_targets, args.image_size)
    elapsed = []
    with torch.inference_mode():
        for _ in range(len(loader)):
            image, gt, name, _ = loader.load_data()
            gt = np.asarray(gt, dtype=np.float32)
            image = image.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)

            prediction = F.interpolate(
                prediction, size=gt.shape, mode="bilinear", align_corners=False
            )
            prediction = prediction.sigmoid().cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (
                prediction.max() - prediction.min() + 1e-8
            )
            output_path = args.output_dir / name
            cv2.imwrite(str(output_path), prediction * 255)
            print(f"Saved: {output_path}")

    average = float(np.mean(elapsed))
    print(f"Average time: {average:.4f} s")
    print(f"Average FPS: {1 / average:.2f}")


if __name__ == "__main__":
    main()

