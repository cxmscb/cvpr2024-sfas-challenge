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
import pandas as pd
from mat_mish import MAT
import math
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import scipy.io as scio
import cv2
import csv
import albumentations


image_size = (256, 256)

class UBCDataset_test(Dataset):
    def __init__(
            self,
            image_path_list,
            transform=None
    ):
        self.transform = transform
        self.image_path_list = image_path_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]

        image = cv2.imread("/input/yiyao/SFAS/HySpeFAS_test/images/" + image_path.replace('.mat', '.png'))
        image = image / 255.0

        image_mat = scio.loadmat("/input/yiyao/SFAS/HySpeFAS_test/images/" + image_path)
        image_mat = image_mat['var']

        ori_image = np.concatenate([image, image_mat], 2)

        if self.transform is not None:
            augmented = self.transform(image=ori_image)
            aug_image = augmented["image"]
            image = np.copy(aug_image).transpose(2, 0, 1)
        else:
            image = np.copy(ori_image).transpose(2, 0, 1)

        return image

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

    test_transform = albumentations.Compose([
            albumentations.Resize(image_size[0], image_size[1]),
            albumentations.Normalize(mean=tuple([0.3521539968746207, 0.6697889192590315, 0.5797934317311467, 0.22948576725358144, 0.2702134654956618, 0.33660533102942164, 0.3260432564557405, 0.28015848749983396, 0.2694844237391819, 0.39492272733839145, 0.5075355665243766, 0.5363099475815994, 0.5531093771455536, 0.5197733765494712, 0.4148619343494439, 0.34003734041080885, 0.29858960333512424, 0.27718474581700575, 0.29235125464137657, 0.3152905427666713, 0.3118694744237367, 0.3136628887682403, 0.3136298175713303, 0.33947353807853653, 0.3643556894874598, 0.3750838909790401, 0.3546842770229526, 0.34376014958151724, 0.3304321719695063, 0.30674894725687035, 0.31427482679502117, 0.3329501692819358, 0.24813784852259704]), std=tuple([0.06571835884173972, 0.13851548753413687, 0.11732161350574984, 0.09394411684512297, 0.11765776135163637, 0.1490966448243471, 0.1445041800217076, 0.12405607676694952, 0.11110454844621506, 0.16880290181403257, 0.23177597478921125, 0.2560494060129194, 0.26472611993447764, 0.251762624534547, 0.20659587327694154, 0.16688317344729187, 0.13876066339254498, 0.12013423642729014, 0.125264398875901, 0.1376299269013092, 0.13850527766543266, 0.1402076385926856, 0.13671944236591912, 0.14665628003430983, 0.15652842839867345, 0.15890061253454535, 0.15247351096318176, 0.15162494839329752, 0.1489521935394348, 0.1417529451872056, 0.14216491199264802, 0.14799438773936432, 0.09999813102027516]), max_pixel_value=1.0, p=1.0)
        ])


    df_test = pd.read_csv('test.csv')

    model = MAT()
    model = model.cuda()
    model_state = torch.load("/input/yiyao/sfas_model_snapshot/mat_b3_256_sfas_model_0/model_epoch_30.pth")["model_state_dict"]
    model.load_state_dict(model_state)
    model.eval()


    path_test = df_test['image'].values.tolist()
    print(len(path_test))
    test_dataset = UBCDataset_test(path_test, test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False
    )


    outPRED = []

    for i, (input) in enumerate(test_loader):
        print(i)
        with torch.no_grad():
            testInput = torch.autograd.Variable(input.float().cuda())
            logit, feature = model(testInput)
        pred = logit.sigmoid().cpu().detach().numpy()
        outPRED.append(pred)

    outPRED = np.concatenate(outPRED, 0)

    print(len(outPRED))

    thr = 0.23 # val thr
    with open("result.txt", 'w') as file:
        for index, pred in enumerate(outPRED):
            label = int(pred[0] >= thr)
            file.write(str(path_test[index])+" "+str(label) + '\n')