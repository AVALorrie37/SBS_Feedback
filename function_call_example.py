'''
TODO: show how to call functions in this project to design pump
'''
import multi_Lorenz_2_triangle as mlt
import numpy as np
import matplotlib.pyplot as plt
import SBS_DSP as sd

if __name__ == '__main__':
    mode_type = "manual_mode"  # "manual_mode" or "auto_mode"
    if mode_type == "manual_mode":
        # manual mode (don't know how many times of loops)
        ''' [1] input initial settings (set requirements of filter) '''
        bandwidth = 20  # MHz
        comb_df = 4  # MHz
        N_iteration = 1  # 迭代次数
        iteration_type = 1  # 迭代方式，[1]-2+3，[2]-线性，[3]-根号,[4]-边界参考旁边 (默认选[1])
        gamma_B = 10  # MHz，布里渊线宽(通过单梳测量得到，可以只存一次）
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
        f_measure = np.linspace(central_freq - bandwidth, central_freq + bandwidth, 20000)  # 设置扫频范围与点数，单位MHz
        measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)

        ''' [3] 迭代反馈 '''
        f_index = mlt.search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
        f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率
        for _ in range(N_iteration):
            brian_measure_sam = np.array([measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
            expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)

            # 更新amp_seq，目前有4种方式
            amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)
            nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
            measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz

    elif mode_type == "auto_mode":
        # TODO: auto mode (automatically stop loop when meets need)
        ''' [1] input initial settings (set requirements of filter) '''
        bandwidth = 100  # MHz
        comb_df = 10  # MHz
        iteration_type = 1  # 迭代方式，[1]-2+3，[2]-线性，[3]-根号,[4]-边界参考旁边 (默认选[1])
        gamma_B = 38  # MHz，布里渊线宽(通过单梳测量得到，可以只存一次）
        type_filter = 'square'  # type_filter='square','triangle'

        ''' [2] check and preprocess '''
        assert bandwidth % comb_df == 0
        N_pump = int(bandwidth / comb_df)+1
        central_freq = 0  # 因为只要确定形状，故此处中心频率采用相对值，设置为0
        BFS = 0  # 因为只要确定形状，故不考虑布里渊频移，设置为0

        ''' [2-1] 初始化频梳幅值，频率与相位'''
        amp_seq = mlt.initial_amp_seq(N_pump, type_filter)
        f_seq = mlt.initial_f_seq(N_pump, central_freq, comb_df)
        phase_list = np.zeros(N_pump)  # 先不考虑随机相位问题
        # phase_list = [sd.randen_phase() for i in range(N_pump)]  # 随机相位
        nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)  # 归一化后泵浦

        ''' [2-2] 计算增益谱 '''
        window_width = max([bandwidth, gamma_B])
        f_measure = np.linspace(central_freq - window_width, central_freq + window_width, 40000)  # 设置扫频范围与点数，单位MHz
        measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)
        # print('gamma-B：', mlt.gmmb_correct(f_measure, measure_brian.real))  # 3db带宽？

        ''' [3] 迭代反馈,当平整度增加时停止迭代 '''
        N_iteration = 0  # 迭代次数统计
        f_index = mlt.search_index(f_seq - BFS, f_measure)  # 找到梳齿对应单布里渊增益中心位置索引
        f_measure_sam = [f_measure[i] for i in f_index]  # 最接近频梳对应的单布里渊增益中心的采样点频率
        brian_measure_sam = np.array([measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
        expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)
        measure_max = measure_brian.real.max()
        normal_measure_brian = measure_brian.real / measure_max
        mean_normal_expected = np.mean(expected_gain_sam) / measure_max
        bias = abs(normal_measure_brian[f_index[0]:f_index[-1]] - mean_normal_expected)
        flatness = max(bias) - min(bias)

        while N_iteration < 50:
            # 更新amp_seq，目前有4种方式
            new_amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)
            nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
            new_measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz

            # 更新后采样计算平均偏差值
            brian_measure_sam = np.array([new_measure_brian.real[i] for i in f_index])  # 最接近频梳频率的采样点增益
            expected_gain_sam = mlt.expected_gain2(f_index, new_measure_brian.real, type_filter)

            measure_max = np.max(new_measure_brian.real)
            normal_measure_brian = new_measure_brian.real / measure_max
            mean_normal_expected = np.mean(expected_gain_sam) / measure_max
            bias = abs(normal_measure_brian[f_index[0]:f_index[-1]] - mean_normal_expected)
            new_flatness = max(bias) - min(bias)
            if new_flatness >= flatness:
                break

            if N_iteration % 10 == 0:
                plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值')

            amp_seq = new_amp_seq
            flatness = new_flatness
            measure_brian = new_measure_brian
            N_iteration += 1

    # print('gamma-B：', mlt.gmmb_correct(f_measure, measure_brian.real))  # 3db带宽？

    ''' [4] 输出参数与画图看效果 '''
    print(amp_seq)
    plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值')
    plt.legend()
    plt.show()


