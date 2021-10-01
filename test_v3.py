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
        self.cn1 = nn.Conv2d(16,32,1,1)
        self.cn2 = nn.Conv2d(32,64,1,1)
        #self.cn3 = nn.Conv2d(256,256,1,1)
        self.fc1 = nn.Linear(64*16*16,256)# Input Layer to Intermediate modules
        self.bn0 = nn.BatchNorm2d(num_features=16) 
        self.bn1 = nn.BatchNorm2d(num_features=32)   # c1用のバッチ正則化
        self.bn2 = nn.BatchNorm2d(num_features=64)

    def __call__(self, x):#順伝播 Forward propagation
        print(x)
        x = self.bn0(self.cn(x))
        x = self.bn1(self.cn1(x))
        x = self.bn2(self.cn2(x))
        #x = self.bn3(self.cn3(x))
        x = x.view(-1, 64*16*16)
        y = self.fc1(x)
        return y

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
        #print(a)
    return torch.tensor(x_batch), torch.tensor(t_batch)


mymodel = MyNet2().to(device)
#test_img = ["D:/procon_2021/data/16x16_v2/(0, 1)_0/0.jpg"]
#img_bat, ans_bat = make_batch(test_img)
#print(img_bat.shape)
opt = optim.Adam(mymodel.parameters(), lr=0.001)

print("a")
print(torch.cuda.get_device_name())

plot_x = []
# ロスと精度を保存するリスト（訓練用・テスト用）
list_loss_train = []
list_loss_test = []
list_acc_train = []
list_acc_test = []



size_batch = 1

img_w = 16
img_h = 16

# 学習データの学習回数
n_epoch = 20000

for i in range(n_epoch):
    sum_loss = 0.
    sum_acc = 0.
    train_list = []
    for j in range(size_batch):
        #縦横のフォルダからbatch size 分画像を持ってくる
        for k in range(img_h):
            for m in range(img_w):
                train_list.append(f"D:/procon_2021/data/16x16_v10/({k}, {m})_{k*img_w + m}/{i*size_batch + j}.jpg")
                #print(f"D:/procon_2021/data/16x16_v3/({k}, {m})_{k*img_w + m}/{i*size_batch + j}.jpg")
    
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

    mean_loss = sum_loss / len(x_batch)
    mean_acc = sum_acc / len(x_batch)
    list_loss_train.append(mean_loss)
    list_acc_train.append(mean_acc)
    plot_x.append(i)
    print(f"- now {i}")
    print("- mean loss:", mean_loss)
    print("- mean accuracy:", mean_acc)    

print("fin")

print(list_acc_train)
print(f"max acc {max(list_acc_train)}")
print(f"-mean acc {sum(list_acc_train)/len(list_acc_train)}")
plt.plot(plot_x,list_acc_train)



