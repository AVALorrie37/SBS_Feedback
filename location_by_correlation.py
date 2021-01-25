''' 互相关定位算法, 抗噪声能力较强
    大部分文件复制来源于multi_lorenz_2_triangle.py，后续可整合到一起
    目前已实现单洛伦兹定位，分辨率取决于采样点数
    单洛伦兹定位修改为用滤波取最值的方法，进一步提高精度，但只适用于反馈前只有单峰的情况
    多洛伦兹多BFS暂时不行，考虑换用L-M算法或POWELL算法
'''
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import time

# # 解决字体显示问题
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串


def lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omega；omega_B-布里渊增益最大点（BFS）；gamma_B-布里渊线宽
    # 输出：Lorenz型的增益因子g_B * g_0 * L_eff/A_eff
    g_0 = 5 * 10 ** (-11)  # 代入石英光纤典型参量值，单位m/W
    alpha = 0.19  # 光纤损耗，单位db/km
    L_eff = 10.2 ** 3 * (1 - np.exp(-alpha * 10)) / alpha  # 代入光纤长度10.2km
    MFD = 10.3 * 10 ** (-6)  # G652D模场直径：10.4+-0.8 um of 1550nm
    A_eff = pi * MFD ** 2 / 4  # 此处近似修正因子k=1
    gain_max = g_0 * L_eff / A_eff / 2  # lorenz峰值
    # gain_max = 10000
    gamma_b22 = (gamma_B / 2) ** 2
    gain_lorenz = gain_max * gamma_b22 / ((omega - omega_B) ** 2 + gamma_b22)

    # VNA_amp_db_max = 10 / np.log(10) * (gain_max*0.05/2 + np.log(np.sqrt(5*10**(-7))))
    # print('VNA_amp_db_max=', VNA_amp_db_max)
    return gain_lorenz


def initial_amp_seq(len_seq, type_filter):
    if type_filter == 'square':
        amp_seq0 = np.ones(len_seq) / len_seq
    elif type_filter == 'triangle':
        amp_seq1 = np.linspace(0, 1, len_seq // 2)
        amp_seq2 = np.linspace(1, 0, len_seq // 2)
        if len_seq % 2 == 1:
            amp_seq2 = np.insert(amp_seq2, 0, 1 + amp_seq1[1])
        amp_seq0 = np.hstack((amp_seq1, amp_seq2)) / len_seq * 2
    else:
        print('非法字符，请检查type_filter')
        amp_seq0 = None
    return amp_seq0


def initial_f_seq(len_seq, central_freq, df):
    f_seq = np.arange(-len_seq // 2 + 1, len_seq // 2 + 1) * df + central_freq
    return f_seq


def add_lorenz(x, amp_seq, f_seq, BFS_seq, gamma_b):
    total_brian = np.zeros(x.size)
    for i in range(f_seq.size):
        total_brian += (amp_seq[i] ** 2) * lorenz(x, (f_seq[i] - BFS_seq[i]), gamma_b)
    total_brian = 10 / np.log(10) * total_brian
    return total_brian


def search_index(f_seq, f_measure):
    # 功能：找到f_seq在f_measure中最接近位置(<0.51*f_resolution)的索引f_index
    # PS : 当前默认每个点都能找到，如果范围不对应可能会出现隐藏bug
    f_index = np.zeros(f_seq.size, dtype=int)
    index_bef = 0
    f_resolution = (f_measure[-1] - f_measure[0]) / f_measure.size
    for i in range(f_seq.size):
        for j in range(index_bef, f_measure.size):
            if abs(f_measure[j] - f_seq[i]) < 0.51 * f_resolution:
                f_index[i] = j
                index_bef = j
                break
    return f_index


def corre_filter(measure_brian, gamma_B):
    # 功能：洛伦兹互相关去噪，修改线宽gamma_B影响分辨率
    x = np.linspace(-3 * 10 ** 3, 3 * 10 ** 3, 1000)  # 扫频范围，单位MHz
    ref_brian = lorenz(x, 0, gamma_B)
    # plt.plot(x, ref_brian)
    corr = np.correlate(measure_brian, ref_brian, 'same')
    index_max = measure_brian.argmax()
    corr_refine = corr / corr.max() * np.mean(measure_brian[index_max-5:index_max+5])
    return corr_refine


def bfs_correct(f_seq, f_measure, measure_brian, gamma_B=15):
    # 功能：洛伦兹互相关滤波取最值得到较准确整体BFS(单位：MHz)，精度取决于f_measure
    # 前提：观察窗口内有SBS增益且只有一个最大值)
    f_resolution = f_measure[1]-f_measure[0]
    print('f_resolution',f_resolution)
    amp_measure = corre_filter(measure_brian, gamma_B / f_resolution)
    cfp=np.median(f_seq)
    cf=f_measure[amp_measure.argmax()]
    print('cfp:', cfp, 'cf:', cf)
    bfs = cfp - cf
    return bfs


def bfs_correct_1(f_seq, f_measure, measure_brian, amp_seq, rough_bfs,gamma_B=15):
    # 功能：根据大致BFS，洛伦兹互相关得到修正后的更准确整体BFS(单位：MHz)(精度不够，舍弃)
    # PS:线宽gamma_B默认为15，后续看是否要动态更改
    bfs_seq = np.ones(f_seq.size)*rough_bfs
    ref_brian = add_lorenz(f_measure, amp_seq, f_seq, bfs_seq, gamma_B)  # 单位MHz
    corr = np.correlate(measure_brian, ref_brian, "full")
    N_shift = corr.argmax() - ref_brian.size + 1  # 偏移位数
    bfs = bfs_seq[0]-(f_measure[1]-f_measure[0])*N_shift
    return bfs


def bfs_correct_2(f_seq, f_measure, measure_brian, amp_seq, rough_bfs):
    # 功能：根据大致BFS，洛伦兹互相关得到修正后的更准确BFS序列(单位：MHz)(未完成，暂时舍弃)
    # PS:线宽默认为15，后续看是否要动态更改
    bfs_seq = np.ones(f_seq.size)*rough_bfs
    gamma_B = 1  # 布里渊线宽，单位MHz
    index_sort_amp = (-amp_seq).argsort()
    for _ in range(1):
        for i in index_sort_amp:
            ref_brian = add_lorenz(f_measure, np.array([amp_seq[i]]), np.array([f_seq[i]]), bfs_seq, gamma_B)  # 单位MHz
            # ref_brian = add_lorenz(f_measure, amp_seq, f_seq, bfs_seq, gamma_B)  # 单位MHz
            corr = np.correlate(measure_brian, ref_brian, "full")
            plt.subplot(212)
            plt.plot(corr)
            plt.xlim(15000, 24000)  # 横坐标范围
            N_shift = corr.argmax() - ref_brian.size + 1  # 偏移位数
            bfs_seq[i] = bfs_seq[i]-(f_measure[1]-f_measure[0])*N_shift
    return bfs_seq


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return x + np.random.randn(len(x)) * np.sqrt(npower)


def awgn_filter(x, window_size):
    length = x.size - window_size
    y = x
    for i in range(length):
        y[i] = np.sum(x[i:i + window_size]) / window_size
    z = y
    for i in np.invert(range(length)):
        z[i + window_size] = np.sum(y[i:i + window_size]) / window_size
    return z


if __name__ == '__main__':
    '''参数设置'''
    N_pump = 2  # 梳齿个数；对称：奇数
    df = 15  # MHz
    gamma_B = 15  # 布里渊线宽，单位MHz
    central_freq = 0  # 泵浦中心频率（MHz）
    BFS = 6  # 布里渊频移（MHz），同时观察梳齿与增益关系时置零
    # BFS = 5#*np.random.randn(1)  # 布里渊频移（MHz），同时观察梳齿与增益关系时置零
    # BFS_seq = 5*np.random.randn(N_pump)
    BFS_seq = 5*np.ones(N_pump)
    print('BFS_seq =', BFS_seq)
    type_filter = 'square'  # type_filter='square','triangle'
    N_iteration = 50  # 迭代次数
    iteration_type = 1  # 迭代方式，1-2+3，2-线性，3-根号,4-边界参考旁边
    alpha = 1 / N_pump  # 迭代系数--和平均梳齿幅值相同
    snr = 23  # 倍数，非db

    '''初始化频梳幅值与频率'''
    amp_seq = initial_amp_seq(N_pump, type_filter)
    # amp_seq[0] = 0.5
    # amp_seq[1] = 0.9
    f_seq = initial_f_seq(N_pump, central_freq, df)

    print('f_seq:', f_seq)


    '''测量增益谱并作图与泵浦比较'''
    f_measure = np.linspace(-0.1 * 10 ** 3, 0.1 * 10 ** 3, 20000)  # 扫频范围，单位MHz
    # f_measure = np.linspace(3.1 * 10 ** 3, 4.0 * 10 ** 3, 20000)  # 扫频范围，单位MHz
    # plt.xlim(15000, 25000)  # 横坐标范围
    # plt.xlim(-50, 50)  # 横坐标范围

    measure_brian = add_lorenz(f_measure, amp_seq, f_seq, BFS_seq, gamma_B)  # 单位MHz
    measure_brian = awgn(measure_brian, snr)  # 加噪声
    # plt.subplot(211)
    plt.plot(f_measure, measure_brian, label='反馈前' + type_filter)  # 画总增益谱
    # plt.bar(f_seq, amp_seq / amp_seq.max() * measure_brian.max() / 2, label='反馈前泵浦', width=1.1, color="k")  # 画频移后泵浦梳齿
    plt.xlim(-50, 50)  # 横坐标范围

    # 计算校正后BFS
    bfs = bfs_correct(f_seq, f_measure, measure_brian, 5)
    # bfs = bfs_correct1(f_seq, f_measure, measure_brian,amp_seq, 0)
    print('BFS =', bfs)
    bfs_seq = np.ones(f_seq.size)*bfs
    measure_brian = add_lorenz(f_measure, amp_seq, f_seq, bfs_seq, gamma_B)  # 单位MHz
    plt.plot(f_measure, measure_brian, label='校正后' + type_filter)  # 画总增益谱

    plt.title('梳齿数:' + str(N_pump) + '；间隔:' + str(df) + 'MHz；线宽:' + str(gamma_B) + 'MHz')
    plt.legend()

    plt.show()
