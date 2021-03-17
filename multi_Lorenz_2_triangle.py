''' 用洛伦兹画三角增益谱，现已扩展到矩形滤波器+离线反馈算法
    变量：amp_seq，df_seq
    输出：三角增益谱
    进度：
    1.初始化增益赋值为常数，等间隔
        画出增益谱
        画出梳齿
    2.计算绝对值误差
        3db带宽范围：fmax-fmin+FWHM（半峰全宽）
        插值采样（待补充）

    3.加入反馈（目前是简易版）
        画图同时观察梳齿与对应增益时，令BFS=0，并修改扫频范围（注意扫频点数不要太少，否则影响反馈效果

    4.泵浦归一化(未完成但影响不大)
'''
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import SBS_DSP as sd
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


def complex_lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omega；omega_B-布里渊增益最大点（BFS）；gamma_B-布里渊线宽
    # 输出：Lorenz型的增益因子g_B * g_0 * L_eff/A_eff
    g_0 = 5 * 10 ** (-11)  # 代入石英光纤典型参量值，单位m/W
    alpha = 0.19  # 光纤损耗，单位db/km
    L_eff = 10.2 ** 3 * (1 - np.exp(-alpha * 10)) / alpha  # 代入光纤长度10km
    MFD = 10.3 * 10 ** (-6)  # G.652D模场直径：10.4+-0.8 um of 1550nm
    A_eff = pi * MFD ** 2 / 4  # 此处近似修正因子k=1
    gain_max = g_0 * L_eff / A_eff / 2  # lorenz峰值
    gain_lorenz = gain_max * gamma_B / 2 / (gamma_B / 2 - (omega - omega_B) * 1j)
    return gain_lorenz


def initial_amp_seq(len_seq, type_filter):
    if type_filter == 'square':
        amp_seq0 = np.ones(len_seq)
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


# def initial_df_seq(N, central_freq, df):
#     df_seq = np.ones(N+1) * df
#     return df_seq


def initial_f_seq(len_seq, central_freq, df):
    f_seq = np.arange(-len_seq // 2 + 1, len_seq // 2 + 1) * df + central_freq
    return f_seq


def normalize_amp_seq(amp_seq, f_seq, phase_list):
    # f_seq单位MHz
    ts = np.linspace(0, 1.e-6, 64000, endpoint=False)
    ys = sd.synthesize1(amp_seq, f_seq*1e6, ts, phase_list)
    amp_seq = amp_seq / (max(ys) - min(ys))
    return amp_seq


def add_lorenz(x, amp_seq, f_seq, gamma_b, BFS):
    total_brian = np.zeros(x.size)
    for i in range(f_seq.size):
        total_brian += (amp_seq[i] ** 2) * lorenz(x, f_seq[i] - BFS, gamma_b)
    total_brian = 10 / np.log(10) * total_brian
    return total_brian


def conv_lorenz(x, amp_seq, f_seq, gamma_b, BFS):
    # f_seq---每个泵浦所在频率点（=df+BFS）
    total_brian = np.zeros(x.size).astype('complex128')
    for i in range(f_seq.size):
        total_brian += complex_lorenz(x, f_seq[i] - BFS, gamma_b) * (amp_seq[i] ** 2)
    total_brian = 10 / np.log(10) * total_brian
    return total_brian


def search_index(f_seq, f_measure):
    # 功能：找到f_seq在f_measure中最接近位置(<0.51*f_resolution)的索引f_index
    # PS : 当前默认每个点都能找到，如果范围不对应可能会出现隐藏bug
    f_index = np.zeros(f_seq.size, dtype=int)
    index_bef = 0
    f_resolution = (f_measure[1] - f_measure[0])
    for i in range(f_seq.size):
        for j in range(index_bef, f_measure.size):
            if abs(f_measure[j] - f_seq[i]) < 0.51 * f_resolution:
                f_index[i] = j
                index_bef = j
                break
    return f_index


def bfs_correct(f_seq, f_measure, measure_brian, bfs):
    # 功能：根据大致BFS，洛伦兹互相关得到修正后的更准确BFS
    bfs_seq = np.ones(f_seq.size)*bfs
    amp_seq = np.ones(f_seq.size)
    gamma_B = 15  # 布里渊线宽，单位MHz
    ref_brian = add_lorenz(f_measure, np.array([amp_seq[0]]), np.array([f_seq[0]]), gamma_B, bfs_seq[0])  # 单位MHz
    f_index = search_index(f_seq, f_measure)
    corr = np.correlate(measure_brian, ref_brian, "full")
    plt.plot(corr)
    plt.show()
    N_shift = corr.argmax() - measure_brian.size+1  # 偏移个数
    bfs = f_measure[N_shift] - f_measure[f_index[0]]
    return bfs


def expected_gain(brian_measure_sam, type_filter):
    # 3db带宽范围：fmax - fmin + FWHM（半峰全宽）
    # 简化版：只算泵浦对应位置
    len_seq = brian_measure_sam.size
    if type_filter == 'square':
        expected_gain_sam = np.ones(len_seq) * np.mean(brian_measure_sam)
    elif type_filter == 'triangle':
        mb_min = np.min(measure_brian)
        mb_max = np.max(measure_brian)
        if len_seq % 2 == 0:
            expected_seq1 = np.linspace(mb_min, mb_max, len_seq // 2)
            expected_seq2 = np.linspace(mb_max, mb_min, len_seq // 2)
        else:
            expected_seq1 = np.linspace(mb_min, mb_max, len_seq // 2 + 1)
            expected_seq2 = np.linspace(mb_max, mb_min, len_seq // 2 + 1)
            expected_seq2 = np.delete(expected_seq2, 0)
        expected_gain_sam = np.hstack((expected_seq1, expected_seq2))
    else:
        print('非法字符，请检查type_filter')
        expected_gain_sam = None
    return expected_gain_sam


def expected_gain2(f_index, measure_brian, type_filter):
    # 3db带宽范围：fmax - fmin + FWHM（半峰全宽）
    # 2版：均值取fmax - fmin范围内，最后算泵浦对应位置
    len_seq = f_index.size
    brian_measure_sam = np.array([measure_brian[i] for i in f_index])  # 最接近频梳频率的采样点增益

    if type_filter == 'square':
        # expected_gain_sam = np.ones(len_seq) * mean_measure_brian
        expected_gain_sam = np.ones(len_seq) * np.mean(measure_brian[f_index[0]:f_index[-1]])
    elif type_filter == 'triangle':
        mb_min = max(np.min(measure_brian), 0)
        mb_max = np.max(measure_brian)
        if len_seq % 2 == 0:
            expected_seq1 = np.linspace(mb_min, mb_max, len_seq // 2)
            expected_seq2 = np.linspace(mb_max, mb_min, len_seq // 2)
        else:
            expected_seq1 = np.linspace(mb_min, mb_max, len_seq // 2 + 1)
            expected_seq2 = np.linspace(mb_max, mb_min, len_seq // 2 + 1)
            expected_seq2 = np.delete(expected_seq2, 0)
        expected_gain_sam = np.hstack((expected_seq1, expected_seq2))
    else:
        print('非法字符，请检查type_filter')
        expected_gain_sam = None
    return expected_gain_sam


def error(expected_gain_sam, brian_measure_sam):
    # 简化版：只算泵浦对应位置
    error_brian_seq = expected_gain_sam - brian_measure_sam
    error_brian = np.mean(error_brian_seq ** 2 / error_brian_seq.size)
    return error_brian


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / x.size
    npower = xpower / snr
    return x + (np.random.randn(x.size)-0.5) * np.sqrt(npower)


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
    N_pump = 10  # 梳齿个数；对称：奇数
    df = 10  # MHz
    gamma_B = 30  # 布里渊线宽，单位MHz
    central_freq = 3.3 * 10 ** 3  # 泵浦中心频率（MHz）
    BFS = 0  # 布里渊频移（MHz），同时观察梳齿与增益关系时置零
    # BFS = 5#*np.random.randn(1)  # 布里渊频移（MHz），同时观察梳齿与增益关系时置零
    type_filter = 'square'  # type_filter='square','triangle'
    N_iteration = 5  # 迭代次数
    iteration_type = 1  # 迭代方式，1-2+3，2-线性，3-根号,4-边界参考旁边
    alpha = 1  # 迭代系数--和平均梳齿幅值相同
    snr = 23  # 倍数，非db

    '''初始化频梳幅值与频率'''
    amp_seq = initial_amp_seq(N_pump, type_filter)
    f_seq = initial_f_seq(N_pump, central_freq, df)

    # 改边界间隔
    # f_seq[0] = f_seq[0]-df/4*3
    # f_seq[-1] = f_seq[-1]+df/4*3
    # f_seq[-1] = f_seq[-2] + gamma_B/2
    # f_seq[0] = f_seq[1] - gamma_B/2
    print('f_seq:', f_seq)

    # amp_seq[6] = 1.7
    # amp_seq[4] = 0.0032
    # amp_seq[-5] = 0.9
    # amp_seq[-1] = 0.2
    '''归一化泵浦梳齿（未完成）'''
    # amp_seq_sum = np.sum(amp_seq)
    # amp_seq = amp_seq / amp_seq_sum
    print('amp_seq:', amp_seq)

    '''加入随机相位并归一化梳齿幅值'''
    phase_list = [sd.randen_phase() for i in range(N_pump)]  # 随机相位
    phase_list = np.zeros(N_pump)
    # phase_list[3]=np.pi
    nml_amp_seq = normalize_amp_seq(amp_seq, f_seq, phase_list)
    print('nml_amp_seq:', nml_amp_seq)

    '''测量增益谱并作图与泵浦比较'''
    f_measure = np.linspace(3.1 * 10 ** 3, 3.5 * 10 ** 3, 20000)  # 扫频范围，单位MHz
    # f_measure = np.linspace(3.1 * 10 ** 3, 4.0 * 10 ** 3, 20000)  # 扫频范围，单位MHz
    # plt.xlim(3250, 3350)  # 横坐标范围

    measure_brian = add_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz
    measure_brian = awgn(measure_brian, snr)  # 加噪声
    plt.plot(f_measure, measure_brian, label='反馈前' + type_filter)  # 画总增益谱
    plt.bar(f_seq, amp_seq / amp_seq.max() * measure_brian.max() / 2, label='反馈前泵浦', width=1.1, color="k")  # 画频移后泵浦梳齿

    # # 计算校正后BFS
    # bfs = bfs_correct(f_seq, f_measure, measure_brian, 0)
    # measure_brian = add_lorenz(f_measure, amp_seq, f_seq, gamma_B, BFS-bfs)  # 单位MHz
    # plt.plot(f_measure, measure_brian, label='校正后' + type_filter)  # 画总增益谱

    '''宽谱公式所测卷积增益谱'''
    draw = True
    # draw = False
    if draw:
        # real_lorenz = complex_lorenz(f_measure, f_seq[0], gamma_B).real
        # imag_lorenz = complex_lorenz(f_measure, f_seq[0], gamma_B).imag
        # arg_lorenz = np.arctan(imag_lorenz/real_lorenz)
        # plt.plot(f_measure, real_lorenz, label='宽谱' )  # 画增益谱
        # plt.plot(f_measure, arg_lorenz, label='相位' )  # 画相位
        measure_brian = conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)
        measure_brian = awgn(measure_brian, snr)
        measure_brian = awgn_filter(measure_brian, 80)  # 滤波
        plt.plot(f_measure, measure_brian.real, label='宽谱卷积' + type_filter)
        # plt.plot(f_measure, measure_brian.imag, label='宽谱相位' + type_filter)

    '''迭代反馈'''
    wanna_feedback = True
    # wanna_feedback = False
    if wanna_feedback:
        f_index = search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
        f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率

        # for _ in range(N_iteration):
        #     brian_measure_sam = np.array([measure_brian[i] for i in f_index])  # 最接近频梳频率的采样点增益
        #
        #     # expected_gain_sam = expected_gain(brian_measure_sam, type_filter)
        #     expected_gain_sam = expected_gain2(f_index, measure_brian, type_filter)
        #
        #     error_brian_seq = error(expected_gain_sam, brian_measure_sam)
        #     print('error_brian_seq:', error_brian_seq)
        #
        #     # 更新amp_seq，目前有三种方式
        #     # 方式1
        #     # amp_seq[0] = np.sqrt(alpha * expected_gain_sam[0] / brian_measure_sam[0] * amp_seq[0])
        #     # amp_seq[-1] = np.sqrt(alpha * expected_gain_sam[-1] / brian_measure_sam[-1] * amp_seq[-1])
        #     # amp_seq[1:-1] = np.sqrt(expected_gain_sam[1:-1] / brian_measure_sam[1:-1]) * amp_seq[1:-1]
        #     # 方式2
        #     amp_seq = np.sqrt(expected_gain_sam / brian_measure_sam) * amp_seq  # （3-7）-->边界收敛不一致
        #     # 方式3
        #     # amp_seq = np.sqrt(alpha * expected_gain_sam / brian_measure_sam * amp_seq)  # （3-8）修正-->需解决放大（归一化？）
        #
        #     # amp_seq = (amp_seq - min(amp_seq)) / (max(amp_seq)-min(amp_seq))
        #     measure_brian = add_lorenz(f_measure, amp_seq, f_seq, gamma_B)  # 单位MHz
        #
        #
        # plt.plot(f_measure, measure_brian, label='迭代'+str(N_iteration)+'次', color='g')  # 画总增益谱
        # plt.bar(f_seq, amp_seq / amp_seq.max() * brian_measure_sam.max()/2, label='反馈后泵浦', width=1.1, color="red")  # 画频移后泵浦梳齿
        #
        # plt.title('梳齿数:'+str(N_pump)+'；间隔:'+str(df)+'MHz；线宽:'+str(gamma_B)+'MHz')

        '''宽谱迭代'''
        for _ in range(N_iteration):
            brian_measure_sam = np.array([measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益

            # expected_gain_sam = expected_gain(brian_measure_sam, type_filter)
            expected_gain_sam = expected_gain2(f_index, measure_brian.real, type_filter)

            # # 计算函数耗时
            # start_time = time.time()
            # end_time = time.time()
            # print('time=', end_time - start_time, 's')

            # error_brian_seq = error(expected_gain_sam, brian_measure_sam)
            # print('error_brian_seq:', error_brian_seq)

            # 更新amp_seq，目前有三种方式
            alpha = np.mean(amp_seq)
            print('alpha： ', alpha)
            if iteration_type == 1:  # 方式1：2+3
                amp_seq[0] = np.sqrt(alpha * expected_gain_sam[0] / brian_measure_sam[0] * amp_seq[0])
                amp_seq[-1] = np.sqrt(alpha * expected_gain_sam[-1] / brian_measure_sam[-1] * amp_seq[-1])
                amp_seq[1:-1] = np.sqrt(expected_gain_sam[1:-1] / brian_measure_sam[1:-1]) * amp_seq[1:-1]
            elif iteration_type == 2:  # 方式2：线性
                amp_seq = np.sqrt(expected_gain_sam / brian_measure_sam) * amp_seq  # （3-7）-->边界收敛不一致
            elif iteration_type == 3:  # 方式3：加根号
                amp_seq = np.sqrt(alpha * expected_gain_sam / brian_measure_sam * amp_seq)  # （3-8）修正
            elif iteration_type == 4:  # 方式4：边界参考旁边
                amp_seq[1:-1] = np.sqrt(expected_gain_sam[1:-1] / brian_measure_sam[1:-1]) * amp_seq[1:-1]
                amp_seq[-1] = amp_seq[-2]
                amp_seq[0] = amp_seq[1]

            print('amp_seq:', amp_seq)

            nml_amp_seq = normalize_amp_seq(amp_seq, f_seq, phase_list)
            measure_brian = conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz
            # measure_brian = awgn(measure_brian, snr)  # 加噪
            # measure_brian = awgn_filter(measure_brian, 80)  # 滤波

    # -------------反馈后边界修正-------------------
    i_wanna_change = True
    i_wanna_change = False
    if i_wanna_change:
        # amp_seq[-1] = amp_seq[-2]
        # amp_seq[0] = amp_seq[1]
        # amp_seq[-1] = (2 * amp_seq[-2] + amp_seq[-1])/3
        # amp_seq[0] = (2 * amp_seq[1] + amp_seq[0]) / 3
        # f_seq[-1] = f_seq[-1] + df / 4 * 3
        f_seq[-1] = f_seq[-2] + gamma_B/2
        f_seq[0] = f_seq[1] - gamma_B/2
        # amp_seq[0] = np.sqrt(amp_seq[1]*amp_seq[0])
        measure_brian = conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz
        # measure_brian = awgn(measure_brian, snr)
        # measure_brian = awgn_filter(measure_brian, 80)  # 滤波

    # ------------画图-----------------------------
    if wanna_feedback:
        plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值', color='r')
        # plt.plot(f_measure, measure_brian.imag, label='迭代' + str(N_iteration) + '次相位', color='g')

        plt.bar(f_seq, amp_seq / amp_seq.max() * brian_measure_sam.max() / 2, label='反馈后泵浦', width=1.1,
                color="red")  # 画频移后泵浦梳齿

    plt.title('梳齿数:' + str(N_pump) + '；间隔:' + str(df) + 'MHz；线宽:' + str(gamma_B) + 'MHz')

    plt.legend()
    plt.show()
