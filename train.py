import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import random
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed=1
random.seed(seed) 
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)   
torch.manual_seed(seed)   
torch.cuda.manual_seed(seed)   
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.benchmark = False   
torch.backends.cudnn.deterministic = True
# make sure to run it successfully with warn_only=False
torch.use_deterministic_algorithms(True, warn_only=False)
import time
import torchvision
import pandas as pd
import json
import cv2
import csv
import albumentations
import math
import sys
from tqdm import tqdm
from mat_mish import MAT
import torchvision
from pytorch_metric_learning import losses
import scipy.io as scio
from torch_ema import ExponentialMovingAverage
from sam import disable_running_stats, enable_running_stats, SAM

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import pandas as pd

pd.options.mode.chained_assignment = None
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import types
from glob import glob
from torchvision import transforms
import albumentations as alb
from torch.optim import lr_scheduler
import math
from torch.optim.optimizer import Optimizer
import warnings
from vrm import mixup_data


warnings.filterwarnings('ignore')


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


def warm_restart(scheduler, T_mult=2):
    if scheduler.last_epoch == scheduler.T_max:
        scheduler.last_epoch = -1
        scheduler.T_max *= T_mult
    return scheduler


def random_mask(image, **params):

    mask_type = random.randint(0, 2)
    
    if mask_type == 0:
        return image
    elif mask_type == 1:
        w, h, c = image.shape
        image[w//2:, :, :] = 0
    elif mask_type == 2:
        w, h, c = image.shape
        if random.random() > 0.5:
            image[:, :h//2, :] = 0
        else:
            image[:, h//2:, :] = 0

    return image


class UBCDataset(Dataset):
    def __init__(
            self,
            image_path_list,
            image_label_list,
            is_train=True,
            transform=None
    ):
        self.transform = transform
        self.image_path_list = image_path_list
        self.image_label_list = image_label_list
        self.is_train = is_train

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]

        image = cv2.imread("/input/yiyao/SFAS/HySpeFAS_trainval/images/" + image_path.replace('.mat', '.png'))
        image = image / 255.0

        image_mat = scio.loadmat("/input/yiyao/SFAS/HySpeFAS_trainval/images/" + image_path)
        image_mat = image_mat['var']

        ori_image = np.concatenate([image, image_mat], 2)

        if self.transform is not None:
            augmented = self.transform(image=ori_image)
            aug_image = augmented["image"]
            image_1 = np.copy(aug_image).transpose(2, 0, 1)
        else:
            image_1 = np.copy(ori_image).transpose(2, 0, 1)

                
        label = self.image_label_list[index]
        label = torch.FloatTensor([label])


        return image_1, label


def generate_transforms(image_size):
    train_transform = albumentations.Compose([
        albumentations.RandomResizedCrop(image_size[0], image_size[1], scale=(0.8, 1.0), ratio=(0.8, 1.2), p=1.0),
        albumentations.Lambda(name='random_mask', image=random_mask, p=1.0),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Cutout(num_holes=2,
                              max_h_size=50,
                              max_w_size=50,
                              p=1.0,
                              fill_value=0),
        albumentations.Normalize(mean=tuple([0.3521539968746207, 0.6697889192590315, 0.5797934317311467, 0.22948576725358144, 0.2702134654956618, 0.33660533102942164, 0.3260432564557405, 0.28015848749983396, 0.2694844237391819, 0.39492272733839145, 0.5075355665243766, 0.5363099475815994, 0.5531093771455536, 0.5197733765494712, 0.4148619343494439, 0.34003734041080885, 0.29858960333512424, 0.27718474581700575, 0.29235125464137657, 0.3152905427666713, 0.3118694744237367, 0.3136628887682403, 0.3136298175713303, 0.33947353807853653, 0.3643556894874598, 0.3750838909790401, 0.3546842770229526, 0.34376014958151724, 0.3304321719695063, 0.30674894725687035, 0.31427482679502117, 0.3329501692819358, 0.24813784852259704]), std=tuple([0.06571835884173972, 0.13851548753413687, 0.11732161350574984, 0.09394411684512297, 0.11765776135163637, 0.1490966448243471, 0.1445041800217076, 0.12405607676694952, 0.11110454844621506, 0.16880290181403257, 0.23177597478921125, 0.2560494060129194, 0.26472611993447764, 0.251762624534547, 0.20659587327694154, 0.16688317344729187, 0.13876066339254498, 0.12013423642729014, 0.125264398875901, 0.1376299269013092, 0.13850527766543266, 0.1402076385926856, 0.13671944236591912, 0.14665628003430983, 0.15652842839867345, 0.15890061253454535, 0.15247351096318176, 0.15162494839329752, 0.1489521935394348, 0.1417529451872056, 0.14216491199264802, 0.14799438773936432, 0.09999813102027516]), max_pixel_value=1.0, p=1.0)
    ])

    val_transform = albumentations.Compose([
        albumentations.Resize(image_size[0], image_size[1]),
        albumentations.Normalize(mean=tuple([0.3521539968746207, 0.6697889192590315, 0.5797934317311467, 0.22948576725358144, 0.2702134654956618, 0.33660533102942164, 0.3260432564557405, 0.28015848749983396, 0.2694844237391819, 0.39492272733839145, 0.5075355665243766, 0.5363099475815994, 0.5531093771455536, 0.5197733765494712, 0.4148619343494439, 0.34003734041080885, 0.29858960333512424, 0.27718474581700575, 0.29235125464137657, 0.3152905427666713, 0.3118694744237367, 0.3136628887682403, 0.3136298175713303, 0.33947353807853653, 0.3643556894874598, 0.3750838909790401, 0.3546842770229526, 0.34376014958151724, 0.3304321719695063, 0.30674894725687035, 0.31427482679502117, 0.3329501692819358, 0.24813784852259704]), std=tuple([0.06571835884173972, 0.13851548753413687, 0.11732161350574984, 0.09394411684512297, 0.11765776135163637, 0.1490966448243471, 0.1445041800217076, 0.12405607676694952, 0.11110454844621506, 0.16880290181403257, 0.23177597478921125, 0.2560494060129194, 0.26472611993447764, 0.251762624534547, 0.20659587327694154, 0.16688317344729187, 0.13876066339254498, 0.12013423642729014, 0.125264398875901, 0.1376299269013092, 0.13850527766543266, 0.1402076385926856, 0.13671944236591912, 0.14665628003430983, 0.15652842839867345, 0.15890061253454535, 0.15247351096318176, 0.15162494839329752, 0.1489521935394348, 0.1417529451872056, 0.14216491199264802, 0.14799438773936432, 0.09999813102027516]), max_pixel_value=1.0, p=1.0)
    ])
    return train_transform, val_transform


def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def calculate(probs, labels, thr=0.5):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / labels.shape[0]
    return ACER, APCER, NPCER


def get_min_acer(probs, labels, thr, grid_density=100):
    probs = np.squeeze(probs, 1)
    labels = np.squeeze(labels, 1)
    acers = []
    apcers = []
    npcers = []
    thrs = []

    acer_05, apcer_05, npcer_05 = calculate(probs, labels, thr=thr)

    for i in range(grid_density + 1):
        thr = 0.0 + i * 1.0 / float(grid_density)
        ACER, APCER, NPCER = calculate(probs, labels, thr=thr)
        thrs.append(thr)
        apcers.append(APCER)
        npcers.append(NPCER)
        acers.append(ACER)
        
    acers = np.array(acers)
    apcers = np.array(apcers)
    npcers = np.array(npcers)
    thrs = np.array(thrs)

    acer_min = np.min(acers)
    acer_min_index = np.argmin(acers)
    thr_min = thrs[acer_min_index]
    apcer = apcers[acer_min_index]
    npcer = npcers[acer_min_index]

    return acer_min, thr_min, apcer, npcer, acer_05, apcer_05, npcer_05


def epochVal_test(path_val, model, dataLoader, loss_bce, val_batch_size, thr):
    model.eval()
    outGT = []
    outPRED = []
    valLoss = 0
    lossTrainNorm = 0

    for i, (input, target) in enumerate(dataLoader):

        if i == 0:
            ss_time = time.time()
        print(
            str(i) + "/" + str(int(len(path_val) / val_batch_size)) + "     " + str((time.time() - ss_time) / (i + 1)),
            end="\r",
        )

        outGT.append(target.detach().numpy())
        with torch.no_grad():
            with ema.average_parameters():
                varInput = torch.autograd.Variable(input.float().cuda())
                varTarget = torch.autograd.Variable(target.view(-1, 1).contiguous().cuda())
                logit, feature = model(varInput)
                outPRED.append(logit.sigmoid().cpu().detach().numpy())

                lossvalue = loss_bce(logit, varTarget)

        valLoss = valLoss + lossvalue.item()
        lossTrainNorm = lossTrainNorm + 1

    outPRED = np.concatenate(outPRED, 0)
    outGT = np.concatenate(outGT, 0)
    valLoss = valLoss / lossTrainNorm
    acer_min, thr_min, apcer, npcer, acer_05, apcer_05, npcer_05 = get_min_acer(outPRED, outGT, thr)

    return valLoss, outGT, outPRED, acer_min, thr_min, apcer, npcer, acer_05, apcer_05, npcer_05


class Config():
    image_size = (256, 256)
    train_batch_size = 30 * 4 * 2
    val_batch_size = 30 * 4 * 2
    workers = 16
    backbone = 'mat_b3_256'
    checkpoint_path = None
    snapshot_path_suffix = '_sfas_model_0'


if __name__ == "__main__":
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed=1
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)   
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True
    # make sure to run it successfully with warn_only=False
    torch.use_deterministic_algorithms(True, warn_only=False)

    args = Config()
    image_size = args.image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = args.workers

    backbone = args.backbone
    checkpoint_path = args.checkpoint_path
    print(backbone)

    snapshot_path = (
            "/input/yiyao/sfas_model_snapshot/"
            + backbone.replace("\n", "").replace("\r", "")
            + args.snapshot_path_suffix
    )

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    header = ["Epoch", "Learning rate", "Train Loss", "Val Loss", "acer_min", "thr_min", "apcer", "npcer"]
    if not os.path.isfile(snapshot_path + "/log.csv"):
        with open(snapshot_path + "/log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    df_train = pd.read_csv("train_all_balance.csv")
    df_val = pd.read_csv("val.csv")

    fake_df_train = df_train[df_train['label']==0]
    live_df_train = df_train[df_train['label']==1]
    print(len(fake_df_train), len(live_df_train))
    df_train = pd.concat([fake_df_train,  live_df_train, live_df_train, live_df_train, live_df_train, live_df_train, live_df_train, live_df_train, live_df_train, live_df_train, live_df_train])


    train_transform, val_transform = generate_transforms(image_size)

    path_train = df_train['image'].values.tolist()
    label_train = df_train['label'].values.tolist()

    path_val = df_val['image'].values.tolist()
    label_val = df_val['label'].values.tolist()

    train_dataset = UBCDataset(path_train, label_train, True, train_transform)
    val_dataset = UBCDataset(path_val, label_val, False, val_transform)

    def _init_fn(worker_id):
        random.seed(int(seed)+worker_id)
        np.random.seed(int(seed)+worker_id)

    g = torch.Generator()
    g.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False, worker_init_fn=_init_fn, generator=g
    )

    g = torch.Generator()
    g.manual_seed(0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False, worker_init_fn=_init_fn, generator=g
    )

    print("train dataset:", len(path_train), "  val dataset:", len(path_val))

    with open(snapshot_path + "/log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train dataset:", len(path_train), "  val dataset:", len(path_val)])
        writer.writerow(["train_batch_size:", train_batch_size, "val_batch_size:", val_batch_size])

    model = MAT()
    model = model.cuda()

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, adaptive=True, rho=0.5, lr=0.01, momentum=0.9, weight_decay=5e-3)
    scheduler = WarmRestart(optimizer, T_max=30, T_mult=1, eta_min=1e-4)

    loss_bce = nn.BCEWithLogitsLoss()
    loss_supcon = losses.CrossBatchMemory(losses.SupConLoss(temperature=0.07), embedding_size=1024, memory_size=1200)

    max_auc = 0
    min_loss = 1
    trMaxEpoch = 30

    for epochID in range(0, trMaxEpoch+1):

        start_time = time.time()
        model.train()

        trainLoss = 0
        lossTrainNorm = 0

        if epochID < 0:
            pass
        else:
            if epochID != 0:
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)

        for batchID, (input, target) in enumerate(train_loader):

            if batchID == 0:
                ss_time = time.time()
            print(
                str(batchID)
                + "/"
                + str(int(len(path_train) / train_batch_size))
                + "     "
                + str((time.time() - ss_time) / (batchID + 1)),
                end="\r",
            )

            varInput = torch.autograd.Variable(input.float()).cuda()
            varTarget = torch.autograd.Variable(target.view(-1, 1).contiguous().cuda())

            varInput, varTarget, _, _ = mixup_data(varInput, 
                                                    varTarget,
                                                    alpha=1.0,
                                                    device=varInput.device,
                                                    class_conditional=True,
                                                    num_classes=2)
            enable_running_stats(model)
            logit, feature = model(varInput)
            loss_ce = torchvision.ops.sigmoid_focal_loss(logit, varTarget, alpha=0.5).mean()
            losss_upcon = loss_supcon(feature, varTarget.view(-1, ))
            lossvalue =  loss_ce + 10.0 * losss_upcon
            lossvalue.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            logit, feature = model(varInput)
            loss_ce = torchvision.ops.sigmoid_focal_loss(logit, varTarget, alpha=0.5).mean()
            losss_upcon = loss_supcon(feature, varTarget.view(-1, ))
            lossvalue =  loss_ce + 10.0 * losss_upcon
            lossvalue.backward()
            optimizer.second_step(zero_grad=True)

            ema.update()


            trainLoss = trainLoss + lossvalue.item()
            lossTrainNorm = lossTrainNorm + 1

            del lossvalue


        trainLoss = trainLoss / lossTrainNorm

        valLoss, outGT, outPRED, acer_min, thr_min, apcer, npcer, acer_05, apcer_05, npcer_05 = epochVal_test(path_val, model, val_loader, loss_bce, val_batch_size, 0.5)

        epoch_time = time.time() - start_time

        with ema.average_parameters():
            torch.save(
                {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
                snapshot_path + "/model_epoch_" + str(epochID) + ".pth",
            )

        result = [
            epochID,
            round(optimizer.state_dict()["param_groups"][0]["lr"], 6),
            round(trainLoss, 5),
            round(valLoss, 5),
            round(acer_min, 5),
            round(thr_min, 5),
            round(apcer, 5),
            round(npcer, 5)
        ]
        print(result)

        with open(snapshot_path + "/log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result)

