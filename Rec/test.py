# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
import cv2
from model.net import OurNet
from data import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--device', default='cuda', help='PyTorch device')
parser.add_argument('--test_path', type=Path, default=Path(__file__).resolve().parent / 'test_datasets', help='test dataset root')
parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parent / 'cpts' / 'RecNet_epoch_best.pth')
parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
opt = parser.parse_args()

torch.cuda.empty_cache()

device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

model = OurNet()
state_dict = torch.load(opt.checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_datasets = ['MR']
t_all = []
for dataset in test_datasets:
    save_path = opt.output_dir / dataset
    save_path.mkdir(parents=True, exist_ok=True)
    image_root = opt.test_path / dataset / 'Edge'
    gt_root = opt.test_path / dataset / 'GT'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        t1 = time.time()
        with torch.no_grad():
            res = model(image)
        t2 = time.time()
        t_all.append(t2 - t1)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        output_path = save_path / name
        print('save img to:', output_path)
        cv2.imwrite(str(output_path), res * 255)
    print('Test Done!')
    print('average time:{:.02f} s'.format(np.mean(t_all) / 1))
    print('average FPS :{:.02f} fps'.format(1 / np.mean(t_all)))

