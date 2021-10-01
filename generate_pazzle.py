#選択および交換AIを作成するためのランダムなパズル作成プログラム

import random

def generate_puzzle(hight,width):

    puzzle = [[0]*16 for i in range(16)] # 0で埋められている2次元配列を作る
    
    
    choices = list(range(width*hight))
    print(choices)

    for h in range(hight):
        for w in range(width):
            number = choices[random.randint(0,len(choices)-1)]
            puzzle[h][w] = number
            choices.remove(number)
    
    print(choices)
    return puzzle


my_puzzle = generate_puzzle(10,10)

for i in range(16):
    print(my_puzzle[i])