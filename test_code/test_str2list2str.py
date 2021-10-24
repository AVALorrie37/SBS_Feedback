"""
测试string与list与numpy互换
todo:模块功能*3
1.能够自由调节幅值比例(输入输出框1，按钮1)
2.能够自由调节梳齿间隔(输入输出框2，按钮1)
3.能够通过反馈，用增益补偿器件不理想带来的波动(去底噪+300点平滑)(输入输出框1，按钮2)
"""
import sys
import json
import multi_Lorenz_2_triangle as mlt
import numpy as np

bandwidth = 90  # MHz
comb_df = 3.6  # MHz
type_filter = 'square'  # type_filter='square','triangle'
N_pump = int(bandwidth / comb_df)

amp_seq = mlt.initial_amp_seq(N_pump, type_filter)
amp_seq_list = amp_seq.tolist()   # numpy转list
str_amp_seq = str(amp_seq_list)   # list转string
print(str_amp_seq)
new_amp_seq = json.loads(str_amp_seq)  # string转list
print('new:', new_amp_seq)
# values = list(map(float, str_amp_seq.split(",")))


n_dots=int(5)
left_list = np.hstack((np.ones(n_dots) / n_dots, np.zeros(n_dots+1)))
print(left_list)