# 读取p开头的csv文件，并水平合并

import glob
import os
import pandas as pd


# input_path = os.getcwd()  # 获取当前文件夹地址
input_path = r'D:\Documents\5G项目\2021-12-30\chip1-1'  # 手动输入目标文件夹地址
print(input_path)
file_counter = 0
feature_name = 'org'
all_files = glob.glob(os.path.join(input_path, f'{feature_name}*.csv'))
all_data_frames = []
for file in all_files:
    data_frame = pd.read_csv(file, index_col=False, header=0, sep=',')
    data_frame.columns = list(data_frame.columns)[:-1]+[os.path.basename(file)[:-4]]  # 最后一列以文件名命名
    all_data_frames.append(data_frame)
    file_counter += 1
data_frame_concat = pd.concat(all_data_frames, axis=1)  # axis=1-平行拼接
output_path = os.path.join(input_path, f'combine_{feature_name}_{str(file_counter)}_files.csv')
data_frame_concat.to_csv(output_path, index=False)
