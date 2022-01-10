# 读取org/smt/onf开头的csv文件，处理成降噪、减基线的布里渊开关增益并水平合并输出

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths,peak_prominences


def peak_analysis(freq, gain_smoothed, CF):
    # 函数功能：峰值分析，具体包括通过开关增益计算主峰频率、增益、半高全宽FWHM,基线
    # 输入：测量频率(单位GHz)，开关增益
    # 输出：主峰BFS(CF:泵浦中心频率),峰值，FWHM,基线(常数)
    freq = np.array(freq, dtype='float64')
    f_resolution = float(freq[1] - freq[0])  # 频率分辨率(GHz)
    # peaks, _ = find_peaks(gain_on_off, width=500, rel_height=0.1)  # 寻峰
    gain_smoothed = np.array(gain_smoothed)
    max_peak = np.max(gain_smoothed)
    peaks, _ = find_peaks(gain_smoothed, height=[max_peak - 1, max_peak])  # 寻峰

    prominences = peak_prominences(gain_smoothed, peaks)[0]  # 计算峰高
    idx_main_peak = prominences.argmax()  # 找主峰
    BFS = CF - freq[peaks[idx_main_peak]]  # 求BFS(单位GHz)
    main_peak_gain = prominences[idx_main_peak]  # 主峰峰值
    baseline = max(gain_smoothed[peaks]) - main_peak_gain  # 求基线

    results_half = peak_widths(gain_smoothed, peaks, rel_height=0.5)  # tuple{0：宽度;1：高度;2:xmin;3:xmax}
    FWHM_main_peak = results_half[0][idx_main_peak] * 1e3 * f_resolution  # 主峰半高全宽(单位MHz)

    return BFS, main_peak_gain, FWHM_main_peak, baseline


plt.style.use('seaborn-whitegrid')

# input_path = os.getcwd()  # 获取当前文件夹地址
input_path = r'D:\Documents\5G项目\2021-12-30\chip3-2-1226'  # 手动输入目标文件夹地址
print(input_path)
file_counter = 0
feature_names = ['org', 'avg', 'onf', 'smt']

all_files = glob.glob(os.path.join(input_path, f'???_C*.csv'))
all_data_frames = []
for file in all_files:
    data_frame = pd.read_csv(file, index_col=False, header=0, sep=',')
    filename_str = os.path.splitext(file)[0]  # 文件名(不含后缀)
    filename_str = os.path.basename(filename_str)  # 文件名(不含路径和后缀)
    filename_str_split = filename_str.split('_')
    CF_str = filename_str_split[1]
    CF = float(CF_str[2:-1])
    print('CF',CF)
    feature_name = filename_str_split[0]
    if feature_name == 'org' or 'avg':  # 原始数据需去噪、减基线
        data_frame['amp_list'] = savgol_filter(data_frame['amp_list'], 301, 3)  # 300点平滑
        BFS, main_peak_gain, FWHM_main_peak, baseline = peak_analysis(data_frame['freq_list'], data_frame['amp_list'], CF)
        data_frame['amp_list'] = data_frame['amp_list'] - baseline
    elif feature_name == 'onf':
        pass
    elif feature_name == 'smt':  # 平滑数据减基线
        BFS, main_peak_gain, FWHM_main_peak, baseline = peak_analysis(data_frame['freq_list'], data_frame['amp_list'],
                                                                      CF)
        data_frame['amp_list'] = data_frame['amp_list'] - baseline

    data_frame['freq_list'] = CF-data_frame['freq_list']/1E9  # 横坐标转换为BFS(GHz)
    data_frame.columns = list(data_frame.columns)[:-1]+[filename_str]  # 最后一列以文件名命名
    plt.plot(data_frame['freq_list'],data_frame[filename_str])
    all_data_frames.append(data_frame)
    file_counter += 1
data_frame_concat = pd.concat(all_data_frames, axis=1)  # axis=1-平行拼接

plt.xlabel('BFS(GHz)')
plt.ylabel('On-Off Gain(dB)')
plt.show()

output_path = os.path.join(input_path, f'solved_{str(file_counter)}_files.csv')
data_frame_concat.to_csv(output_path, index=False)
