import os

mom_path = "4x4_v3"
main_path = f"E:/procon/data/{mom_path}"
os.mkdir(main_path)
for i in range(4):
    for j in range(4):
        num = i*4 + j
        dir_path = f"{main_path}/({i}, {j})_{num}"
        os.mkdir(dir_path)