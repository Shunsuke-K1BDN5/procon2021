# deta v3を使用するためのｆｉｌｅ　基本はv1と変わらない
#gpuを使用可能に下
# ライブラリーのインストール

import os
from os.path import join
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#NNモデル
class MyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn = nn.Conv2d(3, 16, 1, 1)
        self.fc1 = nn.Linear(16*16*16,120)# Input Layer to Intermediate modules
        self.fc2 = nn.Linear(120,256) #Intermediate modules to Output Layer
        self.fc3 = nn.Linear(256,1200) #Intermediate modules to Output Layer
        self.fc4 = nn.Linear(1200,1000) #Intermediate modules to Output Layer
        self.fc5 = nn.Linear(1000,256) #Intermediate modules to Output Layer
        self.bn0 = nn.BatchNorm2d(num_features=16) 

    def forward(self, x):#順伝播 Forward propagation
        print(x)
        x = self.bn0(self.cn(x))
        x = x.view(-1,16*16*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x

#画像を取得 return 画像のテンソル, 画像のラベル番号
def get_img(path):
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    return img

#モデルに画像を入力し学習させる return なし



def make_batch(list_path_img):
    x_batch = []
    t_batch = []
    for path_img in list_path_img:
        img = get_img(path_img)
        img = np.array(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        x_batch.append(img)
        a = str(path_img).split('/')[-2]
        a = int(a.split("_")[-1])
        t_batch.append(a)
        print(a)
    return torch.tensor(x_batch), torch.tensor(t_batch)


mymodel = MyNet2().to(device)
#test_img = ["D:/procon_2021/data/16x16_v2/(0, 1)_0/0.jpg"]
#img_bat, ans_bat = make_batch(test_img)
#print(img_bat.shape)
opt = optim.SGD(mymodel.parameters(), lr=0.01, momentum=0.9)

print("a")

# ロスと精度を保存するリスト（訓練用・テスト用）
list_loss_train = []
list_loss_test = []
list_acc_train = []
list_acc_test = []



size_batch = 1

img_w = 16
img_h = 16

# 学習データの学習回数
n_epoch = 150

for i in range(n_epoch):
    sum_loss = 0.
    sum_acc = 0.
    train_list = []
    for j in range(size_batch):
        #縦横のフォルダからbatch size 分画像を持ってくる
        for k in range(img_h):
            for m in range(img_w):
                train_list.append(f"D:/procon_2021/data/16x16_v3/({k}, {m})_{k*img_w + m}/{i*size_batch + j}.jpg")
                print(f"D:/procon_2021/data/16x16_v3/({k}, {m})_{k*img_w + m}/{i*size_batch + j}.jpg")
    
    x_batch, t_batch = make_batch(train_list)
    x_batch = x_batch.to(device)
    t_batch = t_batch.to(device)
    y = mymodel(x_batch)
    loss = F.cross_entropy(y, t_batch)

    # 逆伝播
    opt.zero_grad()
    loss.backward()

    # パラメータ更新
    opt.step()

    # ロスと精度を蓄積
    sum_loss += loss.item()
    sum_acc += (y.max(1)[1] == t_batch).sum().item()
    print(sum_loss)
    print(sum_acc)

    mean_loss = sum_loss / len(x_batch)
    mean_acc = sum_acc / len(x_batch)
    list_loss_train.append(mean_loss)
    list_acc_train.append(mean_acc)
    print("- mean loss:", mean_loss)
    print("- mean accuracy:", mean_acc)    

print("fin")




