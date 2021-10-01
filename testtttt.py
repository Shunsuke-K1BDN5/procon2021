import os
from os.path import join
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from PIL import Image

import time

n_epoch = 25000
img_h = 4
img_w = 4
size_batch = 10
train_list = [[] for i in range(n_epoch)]

def get_img(path):
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    return img

def get_tensor(path):
    tensor = torch.load(path)
    tensor = np.array(tensor)
    tensor = tensor.astype(np.float32)
    return tensor

def make_batch(list_path_img):
    x_batch = []
    t_batch = []
    for path_img in list_path_img:
        img = get_tensor(path_img)
        x_batch.append(img)
        a = str(path_img).split('/')[-2]
        a = int(a.split("_")[-1])
        t_batch.append(a)
        #print(a)
    return torch.tensor(x_batch), torch.tensor(t_batch)


start = time.time()
def conveter(n):
    map(get_tensor,n)
    return torch.tensor(n)
test_list = [[f"E:/procon/data/4x4_v3/({k}, {m})_{k*img_w + m}/{a*size_batch + j}.pt" for k in range(img_h)  for m in range(img_w)  for j in range(size_batch) ] for a in range(n_epoch)]

test_list = map(conveter,test_list)

end = time.time() - start
print(end)
