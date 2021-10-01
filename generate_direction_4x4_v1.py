# 上下左右どちらに動いてるかのモデルの学習用がぞう

import random
from typing import Tuple
#import numpy as np
import numpy as np
from PIL import Image

import torch


class puzzle():

    def __init__(self,hight,width,numOfselect,numOfmove,type = "zero2end"):
        self.hight = hight
        self.width = width
        self.s_time = numOfselect
        self.m_time = numOfmove
        self.fowerd_rec = dict()
        self.backwerd_rec = dict()
        self.pazzle = [[0] * self.width for i in range(self.hight)]
        self.ndarray = np.zeros((self.hight,self.width,4))
        self.ans = [[0] * self.width for i in range(self.hight)]
        self.ndarray_ans = np.zeros((self.hight,self.width,4))
        self.place_dimension = [[0] * self.width for i in range(self.hight)]
        self.cartesian_coodinates()
        self.zero2end()
        #下は答えを１６baisiteru 
        self.ndarray_ans= np.array(self.ans)
        # 下はplace_dimension をpazzle に追加してる
        self.pazzle = np.array(self.pazzle) + np.array(self.place_dimension)
        self.pazzle = self.pazzle.tolist()
        #print_2d(self.pazzle)
    
    
    #実際にパズルを作成するmain
    def create(self):
        print("aa")
        for i in range(self.s_time):
            target = self.select_cell()
            #print(f"we select {target}")
            move_time_of_target = random.randint(1,self.m_time)
            #move_time_of_target = 2
            intial_target = target
            self.fowerd_rec[intial_target] = ""
            self.pazzle[target[0]][target[1]][2] = 256
            self.make_pic(target,4)
            self.pazzle[target[0]][target[1]][2] = 0
            for j in range(move_time_of_target):
                target, direction = self.move_cell(target,intial_target)
                self.make_pic(target, direction)
                #print(f"move {target} direction  {direction}")
            self.make_backword(target,intial_target)
        return self.pazzle

    
    def list2ndarray(self):
        self.ndarray = np.array(self.pazzle)
        return self.ndarray, self.ndarray_ans


    # マス選択、移動
    def select_cell(self):
        return (random.randrange(0,self.hight),random.randrange(0,self.width))
    
    def move_cell(self,target,i_target):
        moveList = ["D","L","U","R"]

        way = random.randrange(1,5)
        have_record =  len(self.fowerd_rec[i_target])
        if have_record:
            if  moveList[way-1] == self.fowerd_rec[i_target][-1]:
                while(moveList[way-1] == self.fowerd_rec[i_target][-1]):
                    way = random.randrange(1,5)
        
        direction = way-1
        self.pazzle[target[0]][target[1]][2] = 0
        if way == 1 : # 上
            self.pazzle[target[0]-1][target[1]], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]-1][target[1]]
            #print(f"swap {self.pazzle[target[0]-1][target[1]]} {self.pazzle[target[0]][target[1]] }")
            self.fowerd_rec[i_target] += "U"
            self.pazzle[target[0]-1][target[1]][3] = 255
            if target[0]-1 < 0:
                return (self.hight-1,target[1]), direction
            else:
                return (target[0]-1,target[1]), direction
        elif way == 2: #右
            if target[1] + 1 < self.width:
                self.pazzle[target[0]][target[1]+1], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]][target[1]+1]
                #print(f"swap {self.pazzle[target[0]][target[1]+1]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "R"
                self.pazzle[target[0]-1][target[1]][3] = 255
                return (target[0],target[1]+1), direction
            else:
                self.pazzle[target[0]][0], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]][0]
                #print(f"swap {self.pazzle[target[0]][-1]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "R"
                self.pazzle[target[0]-1][target[1]][3] = 255
                return (target[0],0), direction
        elif way == 3: #下
            if target[0] + 1< self.hight:
                self.pazzle[target[0]+1][target[1]], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]+1][target[1]]
                #print(f"swap {self.pazzle[target[0]+1][target[1]]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "D"
                self.pazzle[target[0]-1][target[1]][3] = 255
                return (target[0] + 1,target[1]), direction
            else:
                self.pazzle[0][target[1]], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[0][target[1]]
                #print(f"swap {self.pazzle[-1][target[1]]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "D"
                self.pazzle[target[0]-1][target[1]][3] = 255
                return (0,target[1]), direction
        elif way == 4: #左
            self.pazzle[target[0]][target[1]-1], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]][target[1]-1]
            #print(f"swap {self.pazzle[target[0]][target[1]-1]} {self.pazzle[target[0]][target[1]] }")
            self.fowerd_rec[i_target] += "L"
            self.pazzle[target[0]-1][target[1]][3] = 255
            if target[1]-1 < 0:
                return(target[0],self.width-1), direction
            else:
                return (target[0] ,target[1]-1), direction
        
        
    def make_backword(self,target,i_target):
        target = list(target)
        if target[0] < 0:
            #print(f"bbb {target}")
            target[0] = self.hight+target[0]
            #print(f"aaa {target}")
        if target[1] < 0:
            #print(f"bbb {target}")
            target[1] = self.width+target[1]
            #print(f"aaa {target}")
        target = tuple(target)
        self.backwerd_rec[target] = ""
        for i in self.fowerd_rec[i_target]:
            if i == "U":
                self.backwerd_rec[target] += "D"
            elif i == "D":
                self.backwerd_rec[target] += "U"
            elif i == "R":
                self.backwerd_rec[target] += "L"
            elif i == "L":
                self.backwerd_rec[target] += "R"
    

    #数字の初期化バリエーションs
    def zero2end(self):
        #このバージョンのために変更元は他のバージョンを参照
        for i in range(self.hight):
            for j in range(self.width):
                self.place_dimension[i][j] = [0,0,i* self.width + j,0]
    
    def cartesian_coodinates(self):
        for i in range(self.hight):
            for j in range(self.width):
                self.pazzle[i][j] = [i,j,0,0]
                self.ans[i][j] = [i,j,0,0]
            
    def deffrence(self):
        p ,a = pazzle.list2ndarray()
        return p -a 

    def make_pic(self,target,direction,type = "nomal"):
        folder = target
        label = file_num[direction]
        #Image.fromarray(deffrences.astype(np.uint8)).save(f"E:/procon/data/4x4_LearnDirection_v0/{direction}/{label}.jpg")
        #print(deffrences)
        torch.save(self.pazzle,f"E:/procon/data/4x4_LearnDirection_v8/{direction}/{label}.pt")
        file_num[direction] += 1
        #print_2d(deffrences)
    
        
def print_2d(my_list):
    for a in my_list:
        print(*a) 

# pazzle = puzzle(4, 4, 1, 3)
# print_2d(pazzle.create())
# print(pazzle.fowerd_rec)
# print(pazzle.backwerd_rec)
file_num = [0,0,0,0,0]
trial_num = 600000

for i in range(trial_num):
    select = random.randint(2,16)
    #select = 3
    pazzle = puzzle(4, 4, select, 16 ,"cartesian_coordinates")
    #pazzle = puzzle(2, 2, select, 2 ,"cartesian_coordinates")
    pazzle.create()
    # print(pazzle.fowerd_rec)
    # print(pazzle.backwerd_rec)

    # print("pazzle")
    # print_2d(pazzle.pazzle)

    # p, a = pazzle.list2ndarray()
    # print("sabun")
    # deffrence = p - a
    # print("diff")
    # print(deffrence)
    # print(f"p \n {p}")
    # print(f"a \n {a[1]}")
    # print(f"b \n {pazzle.ans[1]}")
    # #print(pazzle.ndarray)
    # print(pazzle.ans)
    print(i)

