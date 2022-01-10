import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths,peak_prominences
import matplotlib.pyplot as plt

import numpy as np
import multi_Lorenz_2_triangle as mlt
'''对开关增益做峰值分析，同时求基线'''

if __name__ == '__main__':
    # 单频泵浦
    # csvframe2 = pd.read_csv('D:\\Documents\\项目\\2021-10-8实验结果(材料0930)\\1.不同pump功率_调PC_记录双峰功率比最大值\\1.5um-7cm-chip1\\pump25.8\\10-2.31.csv')
    # csvframe3 = pd.read_csv('D:\\Documents\\项目\\2021-10-8实验结果(材料0930)\\1.不同pump功率_调PC_记录双峰功率比最大值\\1.5um-7cm-chip1\\BJ.csv')
    # 多音泵浦
    # csvframe2 = pd.read_csv('D:\\Documents\\项目\\2021-9-22 扫频验证\\300MHz正常.csv')
    # csvframe3 = pd.read_csv('D:\\Documents\\项目\\2021-9-22 扫频验证\\300MHz-BJ.csv')

    #
    plt.style.use('seaborn-whitegrid')
    csvframe2 = pd.read_csv('D:\\Documents\\5G项目\\2021-10-25\\pump20.5dbm--7.24dbm.csv',index_col=False, header=0, sep=',')
    csvframe3 = pd.read_csv('D:\\Documents\\5G项目\\2021-10-25\\BJ.csv',index_col=False, header=0, sep=',')

    freq = csvframe2['x0000'] / (10 ** 9)  # GHz
    freq = np.array(freq,dtype="float64")
    delta_freq = float(freq[1]-freq[0])
    csvframe2_3_mag = csvframe2['y0000'] - csvframe3['y0000']

    csvframe2_3_mag = savgol_filter(csvframe2_3_mag, 301, 3)

    # peaks, _ = find_peaks(csvframe2_3_mag, width=1000, rel_height=0.1)

    max_mag = np.max(csvframe2_3_mag)
    peaks, _ = find_peaks(csvframe2_3_mag, height=[max_mag, max_mag])

    prominences = np.array(peak_prominences(csvframe2_3_mag, peaks))[0]
    idx_main_peak = prominences.argmax()
    BFS = 15 - freq[peaks[idx_main_peak]]
    results_half = peak_widths(csvframe2_3_mag, peaks, rel_height=0.5)  # tuple{0：宽度;1：高度;2:xmin;3:xmax}
    results_full = peak_widths(csvframe2_3_mag, peaks, rel_height=1)
    FWHM_main_peak = results_half[0][idx_main_peak]*1e3*delta_freq  # MHz
    baseline = max(csvframe2_3_mag[peaks])- prominences[idx_main_peak]
    csvframe2_3_mag.tolist()
    gain_offset = csvframe2_3_mag - baseline

    plt.plot(freq, csvframe2_3_mag,  label='gain_on_off')
    plt.plot(freq, gain_offset, label='gain_offset')
    plt.xlabel("Freq(GHz)")
    plt.ylabel("S21(DB)")

    plt.plot(freq[peaks], csvframe2_3_mag[peaks], "x")
    plt.hlines(results_half[1],freq[0]+results_half[2]*delta_freq,freq[0]+results_half[3]*delta_freq, color="C2")
    plt.hlines(results_full[1],freq[0]+results_full[2]*delta_freq,freq[0]+results_full[3]*delta_freq, color="C3")

    plt.legend()
    plt.show()


    # freq_seq = np.linspace(-5, 5, 10)+7.27
    # f_index = mlt.search_index(freq_seq - BFS, freq)  # 搜索时减去BFS
    #
    # ratio = 0.5  # 加窗点数/FWHM对应点数
    # n_dots = int(ratio * (f_index[1] - f_index[0]))  # 半区间取点个数
    # sample_array = np.array([gain_offset[i - n_dots:i + n_dots+1] for i in f_index])
    # left_list = np.hstack((np.ones(n_dots) / n_dots, np.zeros(n_dots+1)))
    # left_measure_sam = np.dot(sample_array, left_list)