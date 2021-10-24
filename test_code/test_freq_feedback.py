'''
TODO: 测试频率反馈
'''
import multi_Lorenz_2_triangle as mlt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def df_feedback(freq_design_seq, freq, gain_offset, BFS, FWHM):
    # 功能：通过左右区间积分，在自然线宽范围内微调梳齿频率间隔（待验证）
    # 输入：梳齿频率freq_design_seq(Hz), 开关增益的频率freq(Hz)和校准基线后的开关响应gain_offset(dB)
    # BFS  # 读取单频测量所得BFS，单位GHz
    # FWHM  # 读取单频测量所得FWHM，单位MHz

    # 边缘梳齿频率不变，只改中间
    freq_design_seq_sam = freq_design_seq[1:-1]
    new_freq_design = freq_design_seq

    # print('freq_design_seq_sam =', freq_design_seq_sam + BFS*1e9)
    f_index = mlt.search_index(freq_design_seq_sam + BFS*1e9, freq)  # 搜索时减去BFS,freq_design_seq和freq单位相同(Hz)
    # print('find freq =', freq[f_index])

    ratio = 0.4  # 加窗点数/FWHM对应点数
    n_dots = int(ratio * (f_index[1] - f_index[0]))  # 半区间取点个数
    # print('n_dots =', n_dots)
    sample_array = np.array([gain_offset[i - n_dots:i + n_dots+1] for i in f_index])
    # print('sample_array size =', sample_array.shape)
    left_list = np.hstack((np.ones(n_dots) / n_dots, np.zeros(n_dots+1)))
    left_measure_sam = np.dot(sample_array, left_list)
    right_list = np.hstack((np.zeros(n_dots + 1), np.ones(n_dots) / n_dots))
    right_measure_sam = np.dot(sample_array, right_list)
    temp = (0.5-left_measure_sam/(left_measure_sam+right_measure_sam))
    offset_f = temp*FWHM
    print('temp ratio', temp)
    print("Offset_f", offset_f)
    new_freq_design[1:-1] = freq_design_seq_sam - offset_f*1e3
    return new_freq_design

if __name__ == '__main__':
    ''' [1] input initial settings (set requirements of filter) '''
    bandwidth = 30  # MHz
    comb_df = 5  # MHz
    iteration_type = 1  # 迭代方式，[1]-2+3，[2]-线性，[3]-根号,[4]-边界参考旁边 (默认选[1])
    gamma_B = 9  # MHz，布里渊线宽(通过单梳测量得到，可以只存一次）
    type_filter = 'square'  # type_filter='square','triangle'
    sample_type = 1  # [1]-只采对应点；[2]-采附近均值;[3]-采附近r个自然线宽内均值;[4]-采附近r个自然线宽内加权值
    N_iteration = 2  # 迭代次数

    ''' [2] check and preprocess '''
    # assert bandwidth % comb_df == 0
    N_pump = int(bandwidth / comb_df)
    central_freq = 0  # 因为只要确定形状，故此处中心频率采用相对值，设置为0
    BFS = 1  # 因为只要确定形状，故不考虑布里渊频移(MHz)，设置为0

    ''' [2-1] 初始化频梳幅值，频率与相位'''
    amp_seq = mlt.initial_amp_seq(N_pump, type_filter)
    f_seq = mlt.initial_f_seq(N_pump, central_freq, comb_df)
    phase_list = np.zeros(N_pump)  # 先不考虑随机相位问题
    nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)  # 归一化后泵浦

    ''' [2-2] 计算增益谱 '''
    f_measure = np.linspace(central_freq - bandwidth, central_freq + bandwidth, 90000)  # 设置扫频范围与点数，单位MHz
    # print('f_resolution =', f_measure[1]-f_measure[0], 'MHz')
    measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)

    ''' [3] 迭代反馈幅值 '''
    f_index = mlt.search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
    f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率
    for _ in range(N_iteration):
        brian_measure_sam = np.array([measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
        expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)

        # 更新amp_seq，目前有4种方式
        amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)
        nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
        measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz

    # print('gamma-B：', mlt.gmmb_correct(f_measure, measure_brian.real))  # 3db带宽？


    ''' [4] 迭代反馈频率 '''
    freq_feedback = True
    if freq_feedback:
        for _ in range(10):
            # 更新freq_seq
            f_seq = df_feedback(f_seq*1e6, f_measure*1e6, measure_brian.real, BFS/1e3, gamma_B)/1e6  # f_seq单位MHz
            measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz
            print('new f_seq =', f_seq)

    # ''' [3] 迭代反馈幅值 '''
    # f_index = mlt.search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
    # f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率
    # for _ in range(5):
    #     brian_measure_sam = np.array([measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
    #     expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)
    #
    #     # 更新amp_seq，目前有4种方式
    #     amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)
    #     nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
    #     measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz



    ''' [5] 输出参数与画图看效果 '''
    print(amp_seq)
    plt.bar(f_seq, amp_seq / amp_seq.max(), label='反馈后泵浦', width=0.8,
            color="gray")  # 画频移后泵浦梳齿

    expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)
    measure_max = measure_brian.real.max()
    normal_measure_brian = measure_brian.real / measure_max
    normal_expected_gain = np.ones(len(measure_brian)) * expected_gain_sam[0] / measure_max

    # plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值')
    plt.plot(f_measure, normal_measure_brian, label='迭代' + str(N_iteration) + '次幅值')
    plt.plot(f_measure, normal_expected_gain, label='期望响应')

    plt.legend()
    plt.show()

    '''[频率，增益]写入csv '''
    measure_brian_csv = pd.DataFrame({'Freq': f_measure, 'amp':  measure_brian.real})
    pump_seq_csv = pd.DataFrame({'泵浦中心频率偏移量': f_seq, '泵浦功率': amp_seq})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    measure_brian_csv.to_csv("D:\\Documents\\项目\\仿真数据\\measure_brian.csv", index=False, sep=',')
    pump_seq_csv.to_csv("D:\\Documents\\项目\\仿真数据\\pump_seq.csv", index=False, sep=',')


