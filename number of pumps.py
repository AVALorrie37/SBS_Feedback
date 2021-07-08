'''
观察并量化相同带宽时最优梳齿个数
'''
import multi_Lorenz_2_triangle as mlt
import numpy as np
import matplotlib.pyplot as plt
import SBS_DSP as sd
import ref_PJcurvature as cc

if __name__ == '__main__':
    ''' [1] input initial settings (set requirements of filter) '''
    Bandwidth = 80  # MHz
    iteration_type = 1  # 迭代方式，[1]-2+3，[2]-线性，[3]-根号,[4]-边界参考旁边 (默认选[1])
    gamma_B = 15  # MHz，布里渊线宽(通过单梳测量得到，可以只存一次）
    type_filter = 'square'  # type_filter='square','triangle'
    sample_type = 4  # [1]-只采对应点；[2]-采附近均值;[3]-采附近r个自然线宽内均值;[4]-采附近r个自然线宽内加权值

    max_curs = []
    pt_cur = 30  # 取通带内需观察曲率的点的个数
    max_N_pump = 20  # 最大允许泵浦个数
    for N_pump in range(Bandwidth // gamma_B+2, max_N_pump):
        ''' [2] 计算特定梳齿数时最接近设计要求的带宽和间隔 '''
        comb_df = round(Bandwidth / N_pump, 2)  # MHz
        bandwidth = comb_df * (N_pump-1)
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
        brian_measure_sam = mlt.measure_sampling(measure_brian.real, f_index, sample_type)
        expected_gain = np.ones(len(measure_brian)) * np.mean(measure_brian.real[f_index[0]:f_index[-1]])
        expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)
        measure_max = measure_brian.real.max()
        normal_measure_brian = measure_brian.real / measure_max
        # normal_expected_gain = np.ones(len(measure_brian)) * expected_gain_sam[0] / measure_max
        # normal_expected_gain_sam = np.ones(len(brian_measure_sam))
        normal_brian_measure_sam = brian_measure_sam / measure_max
        mean_normal_expected = np.mean(expected_gain_sam) / measure_max
        band_pass_res = normal_measure_brian[f_index[0]:f_index[-1]]  # 通带响应
        flatness = max(band_pass_res) - min(band_pass_res)
        print(flatness)

        while N_iteration < 10:
            if N_iteration % 2 == 0:  # 画图
                # plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值')  # 原始响应
                plt.plot(f_measure, normal_measure_brian, label='迭代' + str(N_iteration) + '次幅值')  # 最大值归一化
            # 更新amp_seq，目前有4种方式
            new_amp_seq = mlt.change_amp_seq(amp_seq, expected_gain_sam, brian_measure_sam, iteration_type)  # 更新泵浦
            # new_amp_seq = mlt.change_amp_seq(amp_seq, normal_expected_gain_sam, normal_brian_measure_sam, iteration_type)  # 用归一化的来更新泵浦
            nml_amp_seq = mlt.normalize_amp_seq(amp_seq, f_seq, phase_list)
            new_measure_brian = mlt.conv_lorenz(f_measure, nml_amp_seq, f_seq, gamma_B, BFS)  # 单位MHz

            # 更新后采样计算平均偏差值
            brian_measure_sam = mlt.measure_sampling(new_measure_brian.real, f_index, sample_type)
            expected_gain_sam = mlt.expected_gain2(f_index, new_measure_brian.real, type_filter)

            measure_max = np.max(new_measure_brian.real)
            normal_measure_brian = new_measure_brian.real / measure_max
            normal_brian_measure_sam = brian_measure_sam / measure_max
            mean_normal_expected = np.mean(expected_gain_sam) / measure_max
            bias = abs(normal_measure_brian[f_index[0]:f_index[-1]] - mean_normal_expected)
            band_pass_res = normal_measure_brian[f_index[0]:f_index[-1]]  # 通带响应
            new_flatness = max(band_pass_res) - min(band_pass_res)
            print(new_flatness)

            # 迭代停止条件：
            if new_flatness >= flatness:
                break

            amp_seq = new_amp_seq
            flatness = new_flatness
            measure_brian = new_measure_brian
            N_iteration += 1

        ''' [4] 初始化每次迭代时中间平坦带的曲率最大值，并计算迭代前结果'''
        f_seq_cur = np.linspace(f_seq[1], f_seq[-2], pt_cur)  # 取通带内需观察曲率的点
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

        max_curs.append(max(ka[1:-2]))  # 存迭代完后，平台带内最大曲率

    # print('gamma-B：', mlt.gmmb_correct(f_measure, measure_brian.real))  # 3db带宽？

    ''' [5] 输出参数与画图看效果 '''
    ''' [5-1] 泵浦对应位梳齿搬移 '''
    print(amp_seq)
    plt.bar(f_seq, amp_seq / amp_seq.max(), label='反馈后泵浦', width=0.8,
            color="gray")  # 画频移后泵浦梳齿

    ''' [5-2] 部分迭代与最后一次响应 '''
    expected_gain_sam = mlt.expected_gain2(f_index, measure_brian.real, type_filter)
    measure_max = measure_brian.real.max()
    normal_measure_brian = measure_brian.real / measure_max
    normal_expected_gain = np.ones(len(measure_brian)) * expected_gain_sam[0] / measure_max

    # plt.plot(f_measure, measure_brian.real, label='迭代' + str(N_iteration) + '次幅值')
    plt.plot(f_measure, normal_measure_brian, label='迭代' + str(N_iteration) + '次幅值')
    plt.plot(f_measure, normal_expected_gain, label='期望响应')

    plt.legend()

    ''' [5-3] 最后一次响应对应曲率 '''
    # plt.figure(figsize=(40, 2), dpi=120)
    plt.plot(po[:, 0], po[:, 1])
    plt.quiver(po[:, 0], po[:, 1], ka * no[:, 0], ka * no[:, 1],color='#D2691E')
    # plt.axis('equal')

    ''' [5-4] 平坦带采样处最大曲率变化情况 '''
    # plt.figure()
    # x = np.arange(Bandwidth // gamma_B+2, max_N_pump, 1).astype(dtype=np.str)
    # plt.plot(x, max_curs, marker="o")
    # plt.xlabel('N_pump')
    # plt.ylabel('max curvature')

    plt.show()


