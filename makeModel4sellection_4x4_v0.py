# cnnよりも全結合層をできるだけつなげた
#gpuを使用可能に下
#lr を可変にしてる
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
        self.fc1 = nn.Linear(4*4*3,48)# Input Layer to Intermediate modules
        self.fc2 = nn.Linear(48,48) #Intermediate modules to Output Layer
        self.fc3 = nn.Linear(48,24) #Intermediate modules to Output Layer
        self.fc4 = nn.Linear(24,24) #Intermediate modules to Output Layer
        self.fc5 = nn.Linear(24,16) #Intermediate modules to Output Layer
        self.fc6 = nn.Linear(16,16)

    def forward(self, x):#順伝播 Forward propagation
        x = x.view(-1,16*16*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        y = self.fc6(x)
        #y = nn.Softmax(self.fcp(x))
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
opt = optim.Adam(mymodel.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

print("a")
print(torch.cuda.get_device_name())

plot_x = []
# ロスと精度を保存するリスト（訓練用・テスト用）
list_loss_train = []
list_loss_test = []
list_acc_train = []
list_acc_test = []



size_batch = 1

img_w = 4
img_h = 4

# 学習データの学習回数
n_epoch = 40000

load_batch = 1000

train_list = []
x_batch = [[] for i in range(load_batch)]
t_batch = [[] for i in range(load_batch)]
v = 0
def loader(v):
    for time in range(load_batch):
        for j in range(size_batch):
            #縦横のフォルダからbatch size 分画像を持ってくる
            for k in range(img_h):
                for m in range(img_w):
                    train_list.append(f"E:/procon/data/4x4_v0/({k}, {m})_{k*img_w + m}/{v*load_batch+time*size_batch + j}.jpg")
                    #print(f"D:/procon_2021/data/16x16_v3/({k}, {m})_{k*img_w + m}/{i*size_batch + j}.jpg")
        x_batch[time], t_batch[time] = make_batch(train_list)
        train_list.clear()
        print(f"loading {time + v*load_batch}")
    return v + 1

def reset():
    new_model = MyNet2()
    return new_model

v = loader(v)
reset_i = False
print("learning start")

i = 0
while i < n_epoch:
    if i%load_batch == 0 and i != 0:
        #1000回やって正答率5%以下ならやり直し

        print("1000time")
        if max(list_acc_train) < 0.03:
            temp_model = mymodel
            mymodel = reset()
            mymodel.to(device)
            i = 0
            print("reset")
            print(temp_model == mymodel)
        else:
            print(f"nice {max(list_acc_train)}")
            v = loader(v) 
    
    sum_loss = 0.
    sum_acc = 0.
    opt.zero_grad()
    x = x_batch[i%load_batch].to(device)
    t = t_batch[i%load_batch].to(device)
    y = mymodel(x)
    loss = F.cross_entropy(y, t)

    # 逆伝播
    
    loss.backward()

    

    # ロスと精度を蓄積
    sum_loss += loss.item()
    sum_acc += (y.max(1)[1] == t).sum().item()

    # パラメータ更新
    opt.step()
    #scheduler.step()

    mean_loss = sum_loss / len(x)
    mean_acc = sum_acc / len(x)
    list_loss_train.append(mean_loss)
    list_acc_train.append(mean_acc)
    plot_x.append(i)
    print(f"- now {i}")
    print("- mean loss:", mean_loss)
    print("- mean accuracy:", mean_acc) 
    i += 1
      

print("fin v5 both steped")

print(list_acc_train)
print(f"max acc {max(list_acc_train)}")
print(f"-mean acc {sum(list_acc_train)/len(list_acc_train)}")
plt.plot(plot_x,list_acc_train)



