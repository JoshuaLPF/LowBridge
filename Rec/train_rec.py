import os
import torch.nn.functional as F
import random
import numpy as np
from datetime import datetime
from model import OurNet
from torchvision.utils import make_grid
from data import get_loader, TestDataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
from options import build_train_parser
import torch, gc
import torch.nn as nn


opt = build_train_parser().parse_args()

gc.collect()
torch.cuda.empty_cache()

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

seed_torch()

image_root = opt.train_images
gt_root = opt.train_targets

test_image_root = opt.val_images
test_gt_root = opt.val_targets

save_path = opt.output_dir
save_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=save_path / 'RecNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("RecNet-Train_4_pairs")

model = OurNet(pretrained_backbone=opt.pretrained_backbone)


gpu_num = torch.cuda.device_count()
if gpu_num == 1:
    print("Use Single GPU -", opt.gpu_id)
elif gpu_num > 1:
    print("Use multiple GPUs -", opt.gpu_id)
    model = torch.nn.DataParallel(model)

model.cuda()

if opt.checkpoint is not None:
    state = torch.load(opt.checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if isinstance(model, torch.nn.DataParallel):
        state = {key if key.startswith("module.") else f"module.{key}": value
                 for key, value in state.items()}
    else:
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    print('load model from ', opt.checkpoint)

num_parms = 0
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


print('load data...')
train_loader = get_loader(image_root, gt_root, batchsize=opt.batch_size, trainsize=opt.image_size)
test_loader = TestDataset(test_image_root, test_gt_root, opt.image_size)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epochs:{};lr:{};batch_size:{};image_size:{};clip:{};decay_rate:{};checkpoint:{};output_dir:{};decay_epoch:{}'.format(
        opt.epochs, opt.lr, opt.batch_size, opt.image_size, opt.clip, opt.decay_rate, opt.checkpoint, save_path,
        opt.decay_epoch))

MSE_loss = nn.MSELoss()


step = 0
writer = SummaryWriter(str(save_path / 'summary'))
best_mae = 1
best_epoch = 0

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            pred = model(images)
            L1_loss = torch.mean(torch.abs(pred - gts))
            L2_loss = MSE_loss(pred, gts)
            loss = L1_loss + L2_loss
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} || L1_loss:{:4f}, L2_loss:{:4f}'.
                      format(datetime.now(), epoch, opt.epochs, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], L1_loss.data, L2_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, | L1_loss:{:4f}, L2_loss:{:4f}, mem_use:{:.0f}MB'.
                        format(epoch, opt.epochs, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], L1_loss.data, L2_loss.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = pred[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epochs, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path / f'RecNet_epoch_{epoch}.pth')
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / f'RecNet_epoch_{epoch + 1}.pth')
        print('save checkpoints successfully!')
        raise

def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path / 'RecNet_epoch_best.pth')
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path / 'RecNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epochs + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)
