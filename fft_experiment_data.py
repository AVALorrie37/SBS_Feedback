import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SBS_DSP as sd
from scipy.fftpack import fft

'''对实验数据进行fft'''

# # 解决字体显示问题
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串


def get_fft_amp(TD_N, fs=80e3):
    # FFT求单边频谱及对应幅值，TD_N为时域采样点，fs为采样频率(MHz)
    N_fft = TD_N.size

    fft_y = fft(TD_N)[range(int(N_fft / 2))]
    abs_y = np.abs(fft_y)  # 取模
    normalize_y = abs_y / N_fft * 2  # 归一化

    half_hz = np.arange(int(N_fft / 2)) / N_fft * fs

    plt.xlim(14e3, 14.4e3)
    plt.plot(half_hz, normalize_y, 'b')  # 以频率为横坐标
    plt.title('单边频谱(归一化)', fontsize=9, color='blue')
    plt.xlabel("F（GHz）")
    # plt.ylabel("A归一化")
    plt.subplot(2, 1, 2)
    plt.plot(range(result_data.size), result_data, 'b')  # 以频率为横坐标
    amp_seq=None
    return amp_seq

# # 参数设置
N_fft = 64000    # 时域采样个数
N_AWG = 159960
fs = 64 * 10 ** 9  # AWG采样频率
fs_GHZ = fs / 10 ** 9
# design_pump = np.loadtxt('D:\\desktop\\AWG_cos_Square10.5GHz100.0MHz.txt')
design_pump_f = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\C4--0--00000.csv', skiprows=2, nrows=159960)
design_pump = design_pump_f["Ampl"]
# design_pump = pd.read_table('D:\\Documents\\项目\\1013设计数据\\非等间隔\\AWG_cos_square16G150MHz.txt', sep='\n', header=None, nrows=N_AWG)
# result_data = pd.read_csv('D:\\Documents\\项目\\1013实验数据\\1013YAP\\SQUARE16G120MSMOOTHING.csv', skiprows=6, nrows=20000)
result_data_f = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\F1--0--00000.csv', skiprows=2, nrows=666)
result_data = result_data_f["Ampl"]

# design_pump = np.round(design_pump,1)  # 加入量化误差
# print(design_pump)

fft_N = np.arange(N_fft)
TD_N = [design_pump[i % N_AWG] for i in fft_N]  # 循环补数据
# TD_N = [design_pump.values[i % N_AWG] for i in fft_N]  # 循环补数据

# copy自师兄代码，检查AWG归一化前后数据
no_normal = True
if no_normal:
    AWG_framerate = 64 * 10 ** 9  # AWG采样率
    Df = 1 * 10 ** 6
    N_AWG = int(AWG_framerate / Df)
    t_AWG = N_AWG * (1 / AWG_framerate)
    center_F = 14 * 10 ** 9
    bandwidth = 200 * 10 ** 6
    df = 15 * 10 ** 6
    f_list, amp_list, phase_list = sd.square_filter(center_F, bandwidth, df)
    # f_list, amp_list, phase_list = SBS_DSP.triangle_filter(center_F, bandwidth, df)

    # amp_list=FBFunction(amp_list)
    # f_list=FList(f_list)

    ts = np.linspace(0, t_AWG, N_AWG, endpoint=False)
    ys = sd.synthesize1(amp_list, f_list, ts, phase_list)
    print(f_list)
    wavefile = (ys - min(ys)) / (max(ys) - min(ys)) - 0.5
    TD_N = wavefile

# 去掉直流分量
TD_Nm=np.mean(TD_N)
TD_N=TD_N-TD_Nm

# TD_N = np.zeros((N_fft,1), dtype = float)
# TD_N[range(int(N_AWG))] = design_pump.values[range(int(N_AWG))]  #AWG采样，其余补零
fft_y = fft(TD_N)

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度

normalization_y = abs_y / N_fft * 2           #归一化处理（双边频谱）
half_fft_N = fft_N[range(int(N_fft / 2))]      #取一半区间
normalization_half_y = normalization_y[range(int(N_fft / 2))]
# normalization_half_y = 10*np.log(normalization_half_y)
#由于对称性，只取一半区间（单边频谱）
plt.subplot(2, 1, 1)
# plt.plot(half_fft_N,normalization_half_y,'b')  #以点为横坐标
half_hz = half_fft_N / N_fft * fs_GHZ
# p_half_hz = half_hz[range(15000,17000)]
# plt.xlim(15.925,16.1)
# plt.xlim(11.2, 11.5)
plt.xlim(13.8, 14.2)
plt.plot(half_hz, normalization_half_y, 'b')  #以频率为横坐标
plt.title('单边频谱(归一化)', fontsize=9, color='blue')
plt.xlabel("F（GHz）")
# plt.ylabel("A归一化")
plt.subplot(2, 1, 2)
plt.plot(range(result_data.size), result_data, 'b')  #以频率为横坐标
# plt.xlabel("采样间隔")
plt.show()

# mine
# plt.plot(design_pump)
# plt.plot(result_data['Freq(Hz)'], result_data['S12(DB)'])
plt.show()
