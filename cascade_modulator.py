'''TODO:1.仿真级联调制
   TODO:2.试图反馈
'''

import multi_Lorenz_2_triangle as mlt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    ''' [1] input initial settings (set requirements of filter) '''
    bandwidth = 30  # MHz
    comb_df = 3.6  # MHz  小梳齿间隔
    extend_df = 20  # MHz 大带宽间隔
    num_copy = 1  # 级联调制器之后带宽扩展倍数
    amp_copy = np.ones(num_copy)/num_copy  # 级联调制器之后每个扩展分量增益
    # amp_copy = mlt.multi_change(amp_copy, [0, -1], [0.045, 0.045])  # 修改个别点
    N_iteration = 4  # 迭代次数
    iteration_type = 1  # 迭代方式，[1]-2+3，[2]-线性，[3]-根号,[4]-边界参考旁边 (默认选[1])
    gamma_B = 9  # MHz，布里渊线宽(通过单梳测量得到，可以只存一次）
    type_filter = 'square'  # type_filter='square','triangle'

    ''' [2] check and preprocess '''
    # assert bandwidth % comb_df == 0
    N_pump = int(bandwidth / comb_df)
    central_freq = 0  # 因为只要确定形状，故此处中心频率采用相对值，设置为0
    BFS = 0  # 因为只要确定形状，故不考虑布里渊频移，设置为0
    cf_list = np.linspace(-(num_copy-1)*extend_df/2, (num_copy-1)*extend_df/2, num_copy)
    print(cf_list)

    ''' [2-1] 初始化频梳幅值，频率与相位'''
    amp_seq = mlt.initial_amp_seq(N_pump, type_filter)
    f_seq = mlt.initial_f_seq(N_pump, central_freq, comb_df)
    phase_list = np.zeros(N_pump)  # 先不考虑随机相位问题
    nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)  # 归一化后泵浦

    ''' [2-2] 计算增益谱 '''
    f_measure_width = bandwidth + (num_copy-1)*extend_df
    f_measure = np.linspace(central_freq - f_measure_width, central_freq + f_measure_width, 80000)  # 设置扫频范围与点数，单位MHz
    sgl_measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)
    measure_brian = np.zeros(len(f_measure), dtype='complex128')
    for i in range(num_copy):
        measure_brian += amp_copy[i]*mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS-cf_list[i])

    ''' [3] 迭代反馈 '''
    f_index = mlt.search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
    f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率
    for _ in range(N_iteration):
        brian_measure_sam = np.array([sgl_measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
        expected_gain_sam = mlt.expected_gain2(f_index, sgl_measure_brian.real, type_filter)

        # 更新amp_seq，目前有4种方式
        amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)
        nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
        sgl_measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)
    nml_amp_seq[0] = nml_amp_seq[2]*1.5
    nml_amp_seq[-1] = nml_amp_seq[-3]*1.5
    print("nml_amp_seq = ", nml_amp_seq)

    for i in range(num_copy):
        measure_brian += amp_copy[i] * mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS - cf_list[i])

    ''' [4] 输出参数与画图看效果 '''
    print(amp_seq)
    plt.bar(f_seq, amp_seq / amp_seq.max() * measure_brian.real.max() / 2, width=1.1,
            color="red")  # 画泵浦梳齿
    plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值')
    # plt.legend()
    plt.show()


    '''[频率，增益]写入csv '''
    measure_brian_csv = pd.DataFrame({'Freq': f_measure, 'amp':  measure_brian.real})
    pump_seq_csv = pd.DataFrame({'泵浦中心频率偏移量': f_seq, '泵浦功率': amp_seq})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    measure_brian_csv.to_csv("D:\\Documents\\项目\\仿真数据\\measure_brian.csv", index=False, sep=',')
    pump_seq_csv.to_csv("D:\\Documents\\项目\\仿真数据\\pump_seq.csv", index=False, sep=',')