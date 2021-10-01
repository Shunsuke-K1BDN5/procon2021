import glob
import os
 

for k in range(16):
    for j in range(16):
        # 拡張子.txtのファイルを取得する
        path = f'D:/procon_2021/data/16x16/({k}, {j})/*.jpg'
        i = 1
        
        # txtファイルを取得する
        flist = glob.glob(path)
        print(f"{k},{j}")
        
        # ファイル名を一括で変更する
        for file in flist:
            if not os.path.exists(f"D:/procon_2021/data/16x16/({k}, {j})/" + str(i) + '.jpg'):
                os.rename(file,f"D:/procon_2021/data/16x16/({k}, {j})/" + str(i) + '.jpg')
            i +=1
        
