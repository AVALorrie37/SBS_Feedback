'''
    功能：离线处理实验数据，验证反馈算法
    PS:部分函数粘贴自其他模块，后续考虑是否合并调用
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multi_Lorenz_2_triangle as mlt
import SBS_DSP as sd
import location_by_correlation as loc
from math import pi
import time


def awgn_filter(x, window_size):
    # 功能: 两次加窗平滑滤掉x上噪声
    length = x.size - window_size
    y = x
    for i in range(length):
        y[i] = np.sum(x[i:i + window_size]) / window_size
    z = y
    for i in np.invert(range(length)):
        z[i + window_size] = np.sum(y[i:i + window_size]) / window_size
    return z


def lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omega；omega_B-布里渊增益最大点（BFS）；gamma_B-布里渊线宽
    # 输出：Lorenz型的增益因子g_B * g_0 * L_eff/A_eff
    # g_0 = 5 * 10 ** (-11)  # 代入石英光纤典型参量值，单位m/W
    # alpha = 0.19  # 光纤损耗，单位db/km
    # L_eff = 10.2 ** 3 * (1 - np.exp(-alpha * 10)) / alpha  # 代入光纤长度10.2km
    # MFD = 10.3 * 10 ** (-6)  # G652D模场直径：10.4+-0.8 um of 1550nm
    # A_eff = pi * MFD ** 2 / 4  # 此处近似修正因子k=1
    # gain_max = g_0 * L_eff / A_eff / 2  # lorenz峰值
    gain_max = 10
    gamma_b22 = (gamma_B / 2) ** 2
    gain_lorenz = gain_max * gamma_b22 / ((omega - omega_B) ** 2 + gamma_b22)

    # VNA_amp_db_max = 10 / np.log(10) * (gain_max*0.05/2 + np.log(np.sqrt(5*10**(-7))))
    # print('VNA_amp_db_max=', VNA_amp_db_max)
    return gain_lorenz


def corre_filter(measure_brian, gamma_B):
    # 功能：洛伦兹互相关去噪，修改线宽gamma_B影响分辨率
    x = np.linspace(-3 * 10 ** 3, 3 * 10 ** 3, 1000)  # 扫频范围，单位MHz
    ref_brian = mlt.lorenz(x, 0, gamma_B)
    # plt.plot(x, ref_brian)
    corr = np.correlate(measure_brian, ref_brian, 'same')
    index_max = measure_brian.argmax()
    corr_refine = corr / corr.max() * np.mean(measure_brian[index_max-5:index_max+5])
    return corr_refine

if __name__ == '__main__':
    '''计算开关增益'''
    csvframe2 = pd.read_csv('D:\\Documents\\项目\\210121\\FIBER\\15BW60_10_2.csv', skiprows=6, nrows=20000)
    csvframe3 = pd.read_csv('D:\\Documents\\项目\\210121\\FIBER\\15BW100_10BeiJing.csv', skiprows=6, nrows=20000)

    # plt.plot(csvframe['Freq(Hz)'], csvframe['S21(DB)'])
    # plt.xlabel("Freq(Hz)")
    # plt.ylabel("S21(DB)")
    f_resolution = (csvframe2['Freq(Hz)'][1] - csvframe2['Freq(Hz)'][0])/1e6
    print('f_resolution =', f_resolution, 'MHz')

    # -------------画开关幅频/相频--------------
    select = 0  # 幅频-0；相频-1
    if select == 0:
        if 'S21(MAG)' in csvframe2:
            plt.figure(1)  # 画开关幅频（MAG)
            csvframe2_3_mag = csvframe2['S21(MAG)'] / csvframe3['S21(MAG)']
            csvframe2_3_mag = np.maximum(csvframe2_3_mag, np.ones(csvframe2_3_mag.size) * 0.00001)
            csvframe2_3_mag = 10 * np.log(csvframe2_3_mag) / np.log(10)

            # csvframe2_3_mag = awgn_filter(np.array(csvframe2_3_mag), 50)
            # csvframe2_3_mag = awgn_filter(csvframe2_3_mag, 80)
            plt.plot(csvframe2['Freq(Hz)'] / (10 ** 9), csvframe2_3_mag)
            plt.xlabel("Freq(GHz)")
            plt.ylabel("S21(DB)")
        elif 'S21(DB)' in csvframe2:
            plt.figure(1)  # 画开关幅频（DB)
            csvframe2_3_mag = csvframe2['S21(DB)'] - csvframe3['S21(DB)']
            # csvframe2_3_mag = csvframe2['S21(DB)']

            # 对实验数据滤波；目前有两种方式：平滑或互相关去噪
            t0 = time.time()
            csvframe2_3_mag_fil1 = awgn_filter(np.array(csvframe2_3_mag), 40)
            t1 = time.time()
            csvframe2_3_mag_fil2 = corre_filter(csvframe2_3_mag, gamma_B=30/f_resolution)
            t2 = time.time()
            print('time1=', t1 - t0, 's')
            print('time2=', t2 - t1, 's')

            plt.plot(csvframe2['Freq(Hz)'] / (10 ** 9), csvframe2_3_mag)
            # plt.plot(csvframe2['Freq(Hz)'] / (10 ** 9), csvframe2_3_mag_fil1)
            plt.plot(csvframe2['Freq(Hz)'] / (10 ** 9), csvframe2_3_mag_fil2)
            plt.xlabel("Freq(GHz)")
            plt.ylabel("S21(DB)")
    elif select == 1:
        plt.figure(2)  # 画开关相频（DEG)
        csvframe2_3_deg = np.mod(csvframe2['S21(DEG)'] - csvframe3['S21(DEG)'] + 180, 360) - 180
        # csvframe2_3_deg = csvframe2['S21(DEG)'] - csvframe3['S21(DEG)']
        # csvframe2_3_deg = awgn_filter(np.array(csvframe2_3_deg), 30)
        plt.plot(csvframe2['Freq(Hz)'] / (10 ** 9), csvframe2_3_deg)
        plt.xlabel("Freq(GHz)")
        plt.ylabel("S21(DEG)")

        ## 标记最大点和最小点
        deg_max = np.max(csvframe2_3_deg)
        p1 = csvframe2['Freq(Hz)'][np.argmax(csvframe2_3_deg)] / (10 ** 9)
        plt.text(p1, deg_max, '   ' + str((float('%.2f' % p1), float('%.2f' % deg_max))), ha='left', va='top', fontsize=15)

        deg_min = np.min(csvframe2_3_deg)
        p2 = csvframe2['Freq(Hz)'][np.argmin(csvframe2_3_deg)] / (10 ** 9)
        plt.text(p2, deg_min, str((float('%.2f' % p2), float('%.2f' % deg_min))) + '    ', ha='right', va='bottom',
                 fontsize=15)

        plt.scatter([p1, p2], [deg_max, deg_min], s=35, marker='*', color='red')

    # plt.xlim(3.12, 3.1205)

    '''离线设计与实验数据获取'''
    AWG_framerate = 64 * 10 ** 9  # AWG采样率
    Df = 1 * 10 ** 6
    N_AWG = int(AWG_framerate / Df)
    t_AWG = N_AWG * (1 / AWG_framerate)
    ts = np.linspace(0, t_AWG, N_AWG, endpoint=False)
    f_list, amp_list, phase_list = sd.square_filter(center_F=15 * 10 ** 9, bandwidth=60 * 10 ** 6, df=10 * 10 ** 6)
    # f_list, amp_list, phase_list = sd.triangle_filter(center_F=15 * 10 ** 9, bandwidth=200 * 10 ** 6, df=10 * 10 ** 6)
    f_list = np.array(f_list)
    print('f_list =', f_list)
    amp_list = np.array(amp_list)
    print('amp_list =', amp_list)
    freq_measure = np.array(csvframe2['Freq(Hz)'])
    amp_measure = csvframe2_3_mag

    bfs = loc.bfs_correct(f_list/1e6, freq_measure/1e6, amp_measure, 30)  # 单位MHz
    bfs = 10.833e3
    print('BFS =', bfs)

    freq_design_seq = f_list - bfs*1e6
    print('freq_design_seq =', freq_design_seq)
    amp_design_seq = amp_list
    print('amp_design_seq =', amp_design_seq)
    f_index = mlt.search_index(freq_design_seq, freq_measure)

    '''离线反馈'''
    N_iteration = 7
    for _ in range(N_iteration):
        amp_measure = corre_filter(amp_measure, gamma_B=30/f_resolution)
        expected_amp_sam = mlt.expected_gain2(f_index, amp_measure, 'square')
        amp_measure_sam = np.array([amp_measure[i] for i in f_index])  # 最接近频梳频率的采样点增益
        # print('amp_measure_sam =', amp_measure_sam)
        print('expected_amp_sam =', expected_amp_sam)
        amp_design_seq = np.sqrt(expected_amp_sam / amp_measure_sam) * amp_design_seq  # （3-7）-->边界收敛不一致

        alpha = np.mean(amp_design_seq)
        amp_design_seq[0] = np.sqrt(alpha * expected_amp_sam[0] / amp_measure_sam[0] * amp_design_seq[0])
        amp_design_seq[-1] = np.sqrt(alpha * expected_amp_sam[-1] / amp_measure_sam[-1] * amp_design_seq[-1])
        amp_design_seq[1:-1] = np.sqrt(expected_amp_sam[1:-1] / amp_measure_sam[1:-1]) * amp_design_seq[1:-1]

        print('amp_design_seq =', amp_design_seq)
        # ys = sd.synthesize1(amp_design_seq_new, f_list, ts, phase_list)
        # wavefile = (ys - min(ys)) / (max(ys) - min(ys)) - 0.5

        nml_amp_seq = mlt.normalize_amp_seq(amp_design_seq, f_list/1e6, phase_list)

        measure_brian = 0.025 * mlt.conv_lorenz(freq_measure/1e6, nml_amp_seq, f_list/1e6, gamma_b=26.7, BFS=10833).real  # 单位MHz
        index_max = measure_brian.argmax()
        measure_brian = measure_brian / measure_brian.max() * abs(np.mean(csvframe2_3_mag[index_max - 5:index_max + 5]))
        # measure_brian = measure_brian / measure_brian.max() * abs(np.mean(amp_measure[index_max - 5:index_max + 5]))*0.147 -43.7

        amp_measure = mlt.awgn(measure_brian, snr=30)  # snr = 53

    plt.plot(freq_measure/1e9, amp_measure, label='迭代' + str(N_iteration) + '次幅值', color='r')
    # plt.legend()
    plt.show()