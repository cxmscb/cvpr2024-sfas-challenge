import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch

def mixup_data(x, 
               y, 
               alpha=1.0, 
               device='cpu',
               class_conditional=True,
               num_classes=10):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.shape[0]

    if not class_conditional:
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

    else:
        # Same class mixup with noise appended
        n = 2 # Number of images within the same class to mix
        mixed_x = x.detach().clone() 
        for i in range(num_classes):
            index = torch.nonzero(y == i).squeeze(-1).to(device)

            for j in range(n-1):
                index_shuffle = index[torch.randperm(index.size()[0])]
                x[index] = (lam * x[index] +(1-lam)* mixed_x[index_shuffle])

        mixed_x = x
        y_a, y_b = y, y
                
    return mixed_x, y_a, y_b, lam



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, 
               y, 
               alpha=1.0, 
               device='cpu',
               class_conditional=True,
               num_classes=10):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    assert class_conditional is True
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    batch_size = x.shape[0]

    if not class_conditional:
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

    else:
        # Same class mixup with noise appended
        n = 2 # Number of images within the same class to mix
        mixed_x = x.detach().clone() 
        for i in range(num_classes):
            index = torch.nonzero(y == i).squeeze(-1).to(device)

            for j in range(n-1):
                index_shuffle = index[torch.randperm(index.size()[0])]
                x[index][:, bbx1:bbx2, bby1:bby2] = mixed_x[index_shuffle][:, bbx1:bbx2, bby1:bby2]

        mixed_x = x
        y_a, y_b = y, y
                
    return mixed_x, y_a, y_b, lam