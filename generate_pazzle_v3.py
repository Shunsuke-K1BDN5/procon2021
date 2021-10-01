import random
from typing import Tuple
#import numpy as np
import numpy as np
from PIL import Image




class puzzle():

    def __init__(self,hight,width,numOfselect,numOfmove,type = "zero2end",arrange = False):
        self.hight = hight
        self.width = width
        self.s_time = numOfselect
        self.m_time = numOfmove
        self.pazzle = [[0] * self.width for i in range(self.hight)]
        self.fowerd_rec = dict()
        self.backwerd_rec = dict()
        self.ndarray = np.zeros((self.hight,self.width,3))
        self.arrange = arrange
        #差分を作成するための回答配列
        self.ans = [[0] * self.width for i in range(self.hight)]
        if type == "zero2end":
            self.zero2end()
        elif type == "cartesian_coordinates":
            self.cartesian_coodinates()
    
    
    #実際にパズルを作成するmain
    def create(self):
        for i in range(self.s_time):
            target = self.select_cell()
            #print(f"we select {target}")
            #move_time_of_target = random.randint(1,self.m_time)
            move_time_of_target = 2
            intial_target = target
            self.fowerd_rec[intial_target] = ""
            for j in range(move_time_of_target):
                #print(f"move {target}")
                print("before")
                print_2d(self.pazzle)
                target = self.move_cell(target,intial_target)
                print("after")
                print_2d(self.pazzle)
                print(f"target = {target}")
                self.make_pic(target)
            self.make_backword(target,intial_target)
        
        return self.pazzle

    
    def list2ndarray(self,input_array):
        Zeros = np.zeros((self.hight,self.width,3))
        for i in range(self.hight):
            for j in range(self.width):
                for k in range(3):
                    self.ndarray[i][j][k] = input_array[i][j][k]*16
        return Zeros


    # マス選択、移動
    def select_cell(self):
        return (random.randrange(0,self.hight),random.randrange(0,self.width))
    
    def move_cell(self,target,i_target):
        way = random.randrange(1,5)
        if way == 1: # 上
            self.pazzle[target[0]-1][target[1]], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]-1][target[1]]
            #print(f"swap {self.pazzle[target[0]-1][target[1]]} {self.pazzle[target[0]][target[1]] }")
            self.fowerd_rec[i_target] += "U"
            if target[0]-1 < 0:
                return (self.hight-1,target[1])
            else:
                return (target[0]-1,target[1])
        elif way == 2: #右
            if target[1] + 1 < self.width:
                self.pazzle[target[0]][target[1]+1], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]][target[1]+1]
                #print(f"swap {self.pazzle[target[0]][target[1]+1]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "R"
                return (target[0],target[1]+1)
            else:
                self.pazzle[target[0]][-1], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]][-1]
                #print(f"swap {self.pazzle[target[0]][-1]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "R"
                return (target[0],0)
        elif way == 3: #下
            if target[0] + 1< self.hight:
                self.pazzle[target[0]+1][target[1]], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]+1][target[1]]
                #print(f"swap {self.pazzle[target[0]+1][target[1]]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "D"
                return (target[0] + 1,target[1])
            else:
                self.pazzle[-1][target[1]], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[-1][target[1]]
                #print(f"swap {self.pazzle[-1][target[1]]} {self.pazzle[target[0]][target[1]] }")
                self.fowerd_rec[i_target] += "D"
                return (0,target[1])
        elif way == 4: #左
            self.pazzle[target[0]][target[1]-1], self.pazzle[target[0]][target[1]]  = self.pazzle[target[0]][target[1]] , self.pazzle[target[0]][target[1]-1]
            #print(f"swap {self.pazzle[target[0]][target[1]-1]} {self.pazzle[target[0]][target[1]] }")
            self.fowerd_rec[i_target] += "L"
            if target[1]-1 < 0:
                return(target[0],self.width-1)
            else:
                return (target[0] ,target[1]-1)
        
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
        for i in range(self.hight):
            for j in range(self.width):
                self.pazzle[i][j] = i* self.width + j
    
    def cartesian_coodinates(self):
        for i in range(self.hight):
            for j in range(self.width):
                self.pazzle[i][j] = [i,j,0]
                self.ans[i][j] = [i,j,0]

    def deffrence(self,ans,pazzle):
        a = np.array(ans) & np.array(pazzle)
        defren = a^pazzle
        print_2d(defren.tolist())
        return defren
    
    def make_pic(self,target):
        if self.arrange:
            deffrence_pazzle = self.deffrence(self.ans,self.pazzle).tolist()
        formated = pazzle.list2ndarray(deffrence_pazzle)
        folder = target
        label = file_num[folder[0]][folder[1]]
        f_num = folder[0]*16 + folder[1]
        Image.fromarray(formated.astype(np.uint8)).save(f"D:/procon_2021/data/16x16_test/{folder}_{f_num}/{label}.jpg")
        file_num[folder[0]][folder[1]] += 1
    
        
def print_2d(my_list):
    for a in my_list:
        print(*a) 

# pazzle = puzzle(4, 4, 1, 3)
# print_2d(pazzle.create())
# print(pazzle.fowerd_rec)
# print(pazzle.backwerd_rec)
file_num = [[0] * 16 for i in range(16)]

for i in range(3):
    select = random.randint(2,128)
    pazzle = puzzle(16, 16, 1, 2,"cartesian_coordinates",True)
    pazzle.create()
    # print(pazzle.fowerd_rec)
    # print(pazzle.backwerd_rec)
    # pazzle.list2ndarray()
    # folder = list(pazzle.backwerd_rec.keys())[-1]
    # label = file_num[folder[0]][folder[1]]
    # f_num = folder[0]*16 + folder[1]
    # Image.fromarray(pazzle.ndarray.astype(np.uint8)).save(f"D:/procon_2021/data/16x16_test/{folder}_{f_num}/{label}.jpg")
    # file_num[folder[0]][folder[1]] += 1
    #print(pazzle.ndarray)
    print(i)
