import numpy as np
import cmath
import matplotlib.pyplot as plt
from math import pi
import SBS_DSP as WT

# # 解决字体显示问题
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串


def initial_f_seq(len_seq, central_freq, df):
    f_seq = np.arange(-len_seq//2+1, len_seq//2+1) * df + central_freq
    return f_seq

def complex_lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omega；omega_B-布里渊增益最大点（BFS）；gamma_B-布里渊线宽
    # 输出：Lorenz型的增益因子g_B * g_0 * L_eff/A_eff
    g_0 = 4 * 10 ** (-11)  # 代入石英光纤典型参量值，单位m/W
    alpha = 0.22  # 光纤损耗，单位db/km
    L_eff = 10**3 * (1-np.exp(-alpha*10))/alpha  # 代入光纤长度10km
    MFD = 10.4 * 10 **(-6) # G652D模场直径：10.4+-0.8 um of 1550nm
    A_eff = pi * MFD**2 / 4  # 此处近似修正因子k=1
    gain_max = g_0 * L_eff/A_eff  # lorenz峰值
    gain_lorenz = gain_max * gamma_B/2/(gamma_B/2-(omega-omega_B)*1j)
    return gain_lorenz


def conv_lorenz(x, amp_seq, f_seq, gamma_b):
    # f_seq---每个泵浦所在频率点（=df+BFS）
    total_brian = np.zeros(x.size).astype('complex128')
    for i in range(f_seq.size):
        total_brian += complex_lorenz(x, f_seq[i], gamma_B)*amp_seq[i]
    total_brian = 10 / np.log(10) * total_brian
    return total_brian


def gain_lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omega；omega_B-布里渊增益最大点（BFS）；gamma_B-布里渊线宽
    # 输出：Lorenz型的增益因子g_B * g_0 * L_eff/A_eff
    g_0 = 3 * 10 ** (-11)  # 代入石英光纤典型参量值，单位m/W
    alpha = 0.22  # 光纤损耗，单位db/km
    L_eff = 10**3 * (1-np.exp(-alpha*10))/alpha  # 代入光纤长度10km
    MFD = 10.4 * 10 **(-6) # G652D模场直径：10.4+-0.8 um of 1550nm
    A_eff = pi * MFD**2 / 4  # 此处近似修正因子k=1
    gain_max = g_0 * L_eff/A_eff  # lorenz峰值
    # gain_max = 10000
    gamma_b22 = (gamma_B / 2) ** 2
    gain_lorenz = gain_max/2 * gamma_b22 / ((omega-omega_B)**2 + gamma_b22)
    return gain_lorenz


def phase_lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omega；omega_B-布里渊增益最大点（BFS）；gamma_B-布里渊线宽
    # 输出：Lorenz型的增益因子g_B * g_0 * L_eff/A_eff
    g_0 = 3 * 10 ** (-11)  # 代入石英光纤典型参量值，单位m/W
    alpha = 0.22  # 光纤损耗，单位db/km
    L_eff = 10**3 * (1-np.exp(-alpha*10))/alpha  # 代入光纤长度10km
    MFD = 10.4 * 10 **(-6) # G652D模场直径：10.4+-0.8 um of 1550nm
    A_eff = pi * MFD**2 / 4  # 此处近似修正因子k=1
    gain_max = g_0 * L_eff/A_eff  # lorenz峰值
    # gain_max = 10000
    phase_lorenz = gain_max * gamma_B*(omega-omega_B) / (4*(omega-omega_B)**2 + gamma_B**2)
    return phase_lorenz

if __name__ == '__main__':
    N_pump = 1  # 梳齿个数；对称：奇数
    df = 3  # MHz
    gamma_B = 30  # 布里渊线宽，单位MHz
    central_freq = 0  # 10 * 10 ** 3  # 滤波器中心频率

    '''宽谱公式所测卷积增益谱'''
    f_seq = initial_f_seq(N_pump, central_freq, df)
    # f_measure = np.linspace(9.95 * 10 ** 3, 10.05 * 10 ** 3, 10000)  # 扫频范围，单位MHz
    f_measure = np.linspace(-200, 200, 10000)  # 扫频范围，单位MHz
    real_lorenz0 = gain_lorenz(f_measure, f_seq[0], gamma_B)
    imag_lorenz0 = phase_lorenz(f_measure, f_seq[0], gamma_B)
    real_lorenz = complex_lorenz(f_measure, f_seq[0], gamma_B).real
    imag_lorenz = complex_lorenz(f_measure, f_seq[0], gamma_B).imag
    plt.subplot(2, 2, 1)
    plt.plot(f_measure, real_lorenz0, label="单频增益")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(f_measure, real_lorenz, label="宽谱单增益")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(f_measure, imag_lorenz0, label="单频相位")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(f_measure, imag_lorenz, label="宽谱单相位")

    # gain_lorenz1 = gain_lorenz(f_measure, f_seq[0], gamma_B)
    # phase_lorenz1 = phase_lorenz(f_measure, f_seq[0], gamma_B)
    # plt.subplot(1, 2, 1)
    # plt.plot(f_measure, gain_lorenz1, label="单频幅值")
    # plt.subplot(1, 2, 2)
    # plt.plot(f_measure, phase_lorenz1, label="单频相位")

    # real_measure_brian = conv_lorenz(f_measure, amp_seq, f_seq, gamma_B).real
    # imag_measure_brian = conv_lorenz(f_measure, amp_seq, f_seq, gamma_B).imag
    # # plt.plot(f_measure, real_measure_brian, label='宽谱卷积' + type_filter)
    # plt.plot(f_measure, imag_measure_brian, label='宽谱相位')

    # plt.xlim(9950, 10050)

    plt.legend()
    plt.show()