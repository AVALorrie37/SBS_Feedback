'''
测试并用曲率作为衡量平整度标准
TODO:1.计算泵浦对应处曲率并作图
     2.自动准确判断平台带边界
'''

import multi_Lorenz_2_triangle as mlt
import numpy as np
import matplotlib.pyplot as plt
from ref_code import ref_PJcurvature as cc

if __name__ == '__main__':
    ''' [1] input initial settings (set requirements of filter) '''
    bandwidth = 30  # MHz
    comb_df = 3  # MHz  小梳齿间隔
    extend_df = 20  # MHz 大带宽间隔
    N_iteration = 4  # 迭代次数
    pt_cur = 10  # 取通带内需观察曲率的点的个数
    iteration_type = 1  # 迭代方式，[1]-2+3，[2]-线性，[3]-根号,[4]-边界参考旁边 (默认选[1])
    gamma_B = 15  # MHz，布里渊线宽(通过单梳测量得到，可以只存一次）
    type_filter = 'square'  # type_filter='square','triangle'

    ''' [2] check and preprocess '''
    assert bandwidth % comb_df == 0
    N_pump = int(bandwidth / comb_df)
    central_freq = 0  # 因为只要确定形状，故此处中心频率采用相对值，设置为0
    BFS = 0  # 因为只要确定形状，故不考虑布里渊频移，设置为0

    ''' [2-1] 初始化频梳幅值，频率与相位'''
    amp_seq = mlt.initial_amp_seq(N_pump, type_filter)
    f_seq = mlt.initial_f_seq(N_pump, central_freq, comb_df)
    phase_list = np.zeros(N_pump)  # 先不考虑随机相位问题
    nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)  # 归一化后泵浦

    ''' [2-2] 计算增益谱 '''
    f_measure_width = bandwidth
    f_measure = np.linspace(central_freq - f_measure_width, central_freq + f_measure_width, 100000)  # 设置扫频范围与点数，单位MHz
    measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)
    measure_max = measure_brian.real.max()
    normal_measure_brian = measure_brian.real / measure_max*f_measure_width

    '''[2-3] 计算泵浦对应处最大曲率，判断平台带边界(未完成) '''
    # f_index_cur = mlt.search_index(f_seq - BFS, f_measure)  # 找到泵浦梳齿对应单布里渊增益中心位置索引
    #
    # ka = []
    # no = []
    # po = []
    # for f_idx in f_index_cur:
    #     x = f_measure[f_idx - 1:f_idx + 2]
    #     print(x)
    #     y = normal_measure_brian[f_idx - 1:f_idx + 2]
    #     kappa, norm = cc.PJcurvature(x, y)
    #     ka.append(kappa)
    #     no.append(norm)
    #     po.append([x[1], y[1]])
    #
    # po = np.array(po)
    # no = np.array(no)
    # ka = np.array(ka)
    #
    # idx_max_curs = np.argmax(ka)  # 获取平台带内最大曲率索引值

    ''' [2-4] 初始化每次迭代时中间平坦带的曲率最大值，并计算迭代前结果'''
    max_curs = np.zeros(N_iteration+1)  # 每次迭代时中间平坦带的曲率最大值
    f_seq_cur = np.linspace(f_seq[3], f_seq[-4], pt_cur)  # 取通带内需观察曲率的点
    f_index_cur = mlt.search_index(f_seq_cur - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引

    ka = []
    no = []
    po = []
    for f_idx in f_index_cur:
        x = f_measure[f_idx - 1:f_idx + 2]
        print(x)
        y = normal_measure_brian[f_idx - 1:f_idx + 2]
        kappa, norm = cc.PJcurvature(x, y)
        ka.append(kappa)
        no.append(norm)
        po.append([x[1], y[1]])

    po = np.array(po)
    no = np.array(no)
    ka = np.array(ka)

    max_curs[0] = max(ka[1:-2])  # 存未迭代时，平台带内最大曲率

    ''' [3] 迭代反馈 '''
    f_index = mlt.search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
    f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率
    for i_iter in range(N_iteration):
        brian_measure_sam = np.array([measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
        expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)

        # 更新amp_seq，目前有4种方式
        amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)
        nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
        measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz

        normal_measure_brian = measure_brian.real / measure_max*f_measure_width

        ''' [4] 计算泵浦对应处归一化响应的曲率并画图'''
        ka = []
        no = []
        po = []
        for f_idx in f_index_cur:
            x = f_measure[f_idx-1:f_idx+2]
            # print(x)
            y = normal_measure_brian[f_idx-1:f_idx+2]
            kappa, norm = cc.PJcurvature(x, y)
            ka.append(kappa)
            no.append(norm)
            po.append([x[1], y[1]])

        po = np.array(po)
        no = np.array(no)
        ka = np.array(ka)

        max_curs[i_iter+1] = max(ka[1:-2])  # 存当前迭代时，平台带内最大曲率

    ''' [5] 输出参数与画图看效果 '''
    print(amp_seq)

    ''' [5-1] 最后一次响应 '''
    plt.plot(f_measure, normal_measure_brian, label='迭代' + str(N_iteration) + '次幅值')
    plt.legend()

    ''' [5-2] 最后一次响应对应曲率 '''
    # plt.figure(figsize=(40, 2), dpi=120)
    plt.plot(po[:, 0], po[:, 1])
    plt.quiver(po[:, 0], po[:, 1], ka * no[:, 0], ka * no[:, 1])
    plt.axis('equal')

    ''' [5-3] 平坦带采样处最大曲率变化情况 '''
    # x = np.arange(0, N_iteration+1, 1).astype(dtype=np.str)
    # plt.plot(x, max_curs, marker="o")
    # plt.xlabel('iteration')
    # plt.ylabel('max curvature')

    plt.show()