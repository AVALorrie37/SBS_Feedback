import math
import numpy as np
import random
import scipy
# import numpy.fft as fft
import os
import shutil
# import multi_Lorenz_2_triangle as FB

from scipy.fftpack import fft, ifft

import matplotlib.pyplot as plt

# from thinkdsp import CosSignal, SumSignal
PI2 = math.pi * 2


def randen_phase():
    # list = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    list = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    a = random.choice(list)
    return a


def triangle_wave(start, zhouqi, midu, xdecimals, ydecimals):
    '''
    :param start: the fist value of the wave
    :param end:  the end value of the wave
    :param zhouqi:  the zhouqi range of the wave
    :param midu:  every zhouqi, there are how many points in this zhouqi
    :return: the x array and the y array
    '''

    xout = []
    yout = []
    x = np.around(np.arange(start, start + zhouqi, midu), decimals=xdecimals)
    # y = np.where(x<start+0.5, x-start, 0)
    y = np.around(np.where(x >= start + zhouqi / 2, start + zhouqi - x, x - start), decimals=ydecimals)

    return x, y


def square_wave(start, zhouqi, midu, xdecimals, ydecimals):
    '''
    :param start: the fist value of the wave
    :param end:  the end value of the wave
    :param zhouqi:  the zhouqi range of the wave
    :param midu:  every zhouqi, there are how many points in this zhouqi
    :return: the x array and the y array
    '''
    xout = []
    yout = []
    x = np.around(np.arange(start, start + zhouqi, midu), decimals=xdecimals)
    # y = np.where(x<start+0.5, x-start, 0)
    y = np.around(np.where(x >= start + zhouqi / 2, 1, 0), decimals=ydecimals)

    return x, y


def square_filter(center_F, bandwidth, df):
    # start_F = center_F - bandwidth / 2
    # # end_F=center_F+bandwidth/2
    dots = int(round(bandwidth / df))
    # # a = np.random.randint(0, df / (10 ** 6) + 1)
    # f_list = []
    # amp_list = np.empty(dots)
    # phase_list = []
    # i = 0
    # while i < dots:
    #     # f_list.append(start_F + i * df + a * 10 ** 6)
    #     f_list.append(start_F + i * df)
    #     amp_list[i] = 0.003
    #     phase_list.append(randen_phase())
    #     i = i + 1
    amp_list = np.ones(dots)
    f_list = np.arange(-dots // 2 + 1, dots // 2 + 1) * df + center_F
    phase_list = np.random.randint(low=0, high=8, size=dots) * (np.pi / 4)

    return f_list, amp_list, phase_list


def triangle_filter(center_F, bandwidth, df):
    dots = int(round(bandwidth / df))
    # start_F = center_F - bandwidth / 2
    # end_F = center_F + (center_F - start_F)
    # # a = np.random.randint(0, df / (10 ** 6) + 1)
    # f_list = []
    # amp_list = np.empty(dots)
    # phase_list = []
    # i = 0
    # j = int(dots / 2 + 1)
    # while i < dots:
    #     # f_list.append(start_F + i * df + a * 10 ** 6)
    #     f_list.append(start_F + i * df)
    #     phase_list.append(randen_phase())
    #     if i < j:
    #         amp_list[i] = 0.003*i * df * 2 / (center_F - start_F)
    #     else:
    #         amp_list[i] = -1*0.003*(end_F - start_F - i * df) * 2 / (start_F - center_F)
    #
    #     i = i + 1
    phase_list = np.random.randint(low=0, high=8, size=dots) * (np.pi / 4)
    f_list = np.arange(-dots // 2 + 1, dots // 2 + 1) * df + center_F
    amp_list1 = np.linspace(0, 1, dots // 2)
    amp_list2 = np.linspace(1, 0, dots // 2)
    if dots % 2 == 1:
        amp_list2 = np.insert(amp_list2, 0, 1 + amp_list1[1])
    amp_list = np.hstack((amp_list1, amp_list2))

    return f_list, amp_list, phase_list

def Guass_filter(center_F, bandwidth, df):
    dots=int(round(bandwidth / df))
    phase_list = np.random.randint(low=0, high=8, size=2*dots) * (np.pi / 4)
    f_list=np.linspace(center_F-2*bandwidth,center_F+2*bandwidth,2*dots)
    amp_list=np.exp(-1*((f_list-center_F)**2)/(2*(bandwidth**2)))/(np.sqrt(2*np.pi)*bandwidth)

    return f_list,amp_list,phase_list

def Band_stop_filter(center_F, bandwidth, df, signal_BW):
    start_F = float(center_F) - signal_BW / 2
    dots = int(signal_BW / df + 1)
    dot = int(bandwidth / df + 1)
    mindot = int((dots - dot) / 2 + 1)
    maxdot = mindot + dot
    f_list = []
    amp_list = np.empty(dots)
    phase_list = []
    i = 0
    while i < dots:
        f_list.append(start_F + i * df)
        phase_list.append(randen_phase())
        if mindot <= i < maxdot:
            amp_list[i] = 0.003
        else:
            amp_list[i] = 0.1
        i = i + 1

    return f_list, amp_list, phase_list


class Signal:
    """Represents a time-varying signal."""

    def __add__(self, other):
        """Adds two signals.

        other: Signal

        returns: Signal
        """
        if other == 0:
            return self
        return SumSignal(self, other)

    __radd__ = __add__

    @property
    def period(self):
        """Period of the signal in seconds (property).

        Since this is used primarily for purposes of plotting,
        the default behavior is to return a value, 0.1 seconds,
        that is reasonable for many signals.

        returns: float seconds
        """
        return 0.1


class SumSignal(Signal):
    """Represents the sum of signals."""

    def __init__(self, *args):
        """Initializes the sum.

        args: tuple of signals
        """
        self.signals = args

    @property
    def period(self):
        """Period of the signal in seconds.

        Note: this is not correct; it's mostly a placekeeper.

        But it is correct for a harmonic sequence where all
        component frequencies are multiples of the fundamental.

        returns: float seconds
        """
        return max(sig.period for sig in self.signals)

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        return sum(sig.evaluate(ts) for sig in self.signals)


class Sinusoid(Signal):
    """Represents a sinusoidal signal."""

    def __init__(self, freq=440, amp=1.0, offset=0, func=np.sin):
        """Initializes a sinusoidal signal.

        freq: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        offset: float phase offset in radians
        func: function that maps phase to amplitude
        """
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return 1.0 / self.freq

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys


def CosSignal(freq, amp, offset):
    """Makes a cosine Sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians

    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=np.cos)


def synthesize1(amps, fs, ts, offset):
    components = [CosSignal(freq, amp, offset)
                  for amp, freq, offset in zip(amps, fs, offset)]
    signal = SumSignal(*components)

    ys = signal.evaluate(ts)
    return ys


def get_fft(ys, Fs):
    L = len(ys)
    FFT_y = abs(fft(ys)) / L
    Fre = np.arange(int(L / 2)) * Fs / L
    FFT_y = FFT_y[range(int(L / 2))]
    return FFT_y, Fre


def get_mifile(ys):
    headfile = '''DEPTH =16384 ;
WIDTH = 14;
ADDRESS_RADIX = UNS;
DATA_RADIX = UNS;
CONTENT BEGIN
'''
    mif = open('FPGA_cos_square.mif', 'w')
    # mif = open('FPGA_cos_triangle.mif', 'w')
    mif.writelines(headfile)

    i = 0
    # ys=ys+abs(min(ys))
    ys = ys * 2 ** 13
    # ys = (ys - min(ys)) / (max(ys) - min(ys))*2**14-1
    # ys=(ys/max(abs(ys)))*2**14-1

    # test
    fs, hz = get_fft(ys, N_FPGA)
    plt.figure()
    plt.plot(hz, fs)
    plt.show()

    for item in ys:
        item = round(abs(item))
        mif.writelines(str(i))  # wirtelines（）只能输入字符串类型
        mif.writelines(' : ')
        mif.writelines(str(item))
        mif.writelines(';')
        mif.writelines('\n')
        i += 1
    mif.write('END')
    mif.close()

    return mif


def get_awgfile(ys, center_F, bandwidth, df,shape):
    shape_list=['Square','Triangle','Band_stop','Guass']
    center = str(center_F / (10 ** 9))
    ys = (ys - min(ys)) / (max(ys) - min(ys)) - 0.5
    # ys=1.0/(1+np.exp(-ys))
    # mean = np.average(ys)
    # sigma=np.std(ys)
    # ys=(ys-mean)/sigma
    txt = 'AWG_'+shape_list[shape] + 'CW='+center + 'GHz' +'BW='+ str(bandwidth / (10 ** 6)) + 'MHz' + 'Df='+str(df / (10 ** 6)) + 'MHz' + '.txt'
    # txt='AWG_cos_square.txt'
    with open(txt, 'w') as f:
        for item in ys:
            f.write(str(item))
            f.write('\n')

    return txt


def lorenz(omega, omega_B, gamma_B):
    # 输入：频率-omege；Omega_B布里渊增益最大点（BFS）；gamma_B布里渊线宽
    # 输出：Lorenz型的增益因子g_B*g_0*L_eff/A_eff
    # omega_sbs=10.7**9
    g_0 = 4 * 10 ** (-11)  # 代入石英光纤典型常量值，单位m/W
    alpha = 0.22  # 光纤损耗，单位dB/km
    L_eff = 10 ** 3 * (1 - np.exp(-alpha * 10)) / alpha
    MFD = 10.4 * 10 ** (-6)  # G652D模场直径：10.4+-0.8um  1550nm
    A_eff = np.pi * MFD ** 2 / 4  # 此处近似修正因子k=1
    gain_max = g_0 * L_eff / A_eff  # lorenz峰值
    gamma_b22 = (gamma_B / 2) ** 2
    gain_lorenz = gain_max * gamma_b22 / ((omega - omega_B) ** 2 + gamma_b22)
    return gain_lorenz


def add_lorenz(x, amp_seq, f_seq, gamma_b):
    total_brian = np.zeros(len(x))
    for i in range(len(f_seq)):
        total_brian += (amp_seq[i] ** 2) * lorenz(x, f_seq[i] - 9.7e9, gamma_b)
    total_brian = 10 / np.log(10) * total_brian
    return total_brian


if __name__ == '__main__':
    FPGA_framerate = 2.5 * 10 ** 9  # FPGA采样率
    AWG_framerate = 64 * 10 ** 9  # AWG采样率

    Df = 1 * 10 ** 6
    # Df=1.25*10**6
    # Df=1.526*10**5        #频率分辨率
    FM_AWG = AWG_framerate / 2.56  # AWG最高分析频率
    FM_FPGA = FPGA_framerate / 2.56  # FPGA最高分析频率
    N_AWG = int(AWG_framerate / Df)
    N_FPGA = int(FPGA_framerate / Df)
    # N_FPGA=2**14
    t_AWG = N_AWG * (1 / AWG_framerate)
    t_FPGA = N_FPGA * (1 / FPGA_framerate)
    # offset = randen_phase()
    # amps = np.array([2, 2, 2, 2])
    #  fs = [650,670,690, 710, 730, 750]
    # ts = np.linspace(0, 0.00004, 256000, endpoint=True)
    # ys = synthesize1(amps, fs, ts, offset)

    # N=2**14    #采样率不变，采样个数为2的幂次方，方便后期FFT，之后判断时间间隔，生成波形数据
    # ts0=(0.00004/((FPGA_framerate/2)*(N_AWG-1))
    # amp_list=np.array([2,2,2,2,2,2])
    # f_list=[650,670,690,710,730,750]

    # f_list, amp_list, phase_list = square_filter(center_F=14 * 10 ** 9, bandwidth=200 * 10 ** 6, df=15 * 10 ** 6)
    # f_list, amp_list, phase_list=triangle_filter(center_F=20*10**9,bandwidth=200*10**6,df=15*10**6)
    # f_list, amp_list, phase_list=Band_stop_filter(center_F=10*10**9, bandwidth=3*10**9, signal_BW=5*10**9, df=10**7)
    # 以MHz为单位，频梳间隔为15~20MHz
    # ts = np.linspace(0,t_FPGA,N_FPGA,endpoint=False)
    center_F = 14 * 10 ** 9
    bandwidth = 600 * 10 ** 6
    df = 2 * 10 ** 6
    shape=3
    CF_gap=0
    BW_gap=30*10**6
    df_gap=0
    for i in range(0, 1):
        if shape==0:
            f_list, amp_list, phase_list = square_filter(center_F, bandwidth, df)
        elif shape==1:
            f_list, amp_list, phase_list = triangle_filter(center_F, bandwidth, df)
        elif shape==2:
            f_list, amp_list, phase_list = Band_stop_filter(center_F, bandwidth, df, signal_BW=1 * 10 ** 9)
        elif shape==3:
            f_list, amp_list, phase_list = Guass_filter(center_F, bandwidth, df)
        else:
            amp_list = []
            f_list = []
            phase_list = []

        ts = np.linspace(0, t_AWG, N_AWG, endpoint=False)
        ys = synthesize1(amp_list, f_list, ts, phase_list)
        print(f_list)
        txt = get_awgfile(ys, center_F, bandwidth, df,shape)
        center_F = center_F + CF_gap
        bandwidth=bandwidth+BW_gap
        df = df + df_gap
# 验证设计波形AWG.np.array格式

    # f_list,amp_list,phase_list=square_filter(center_F,bandwidth,df)
    # ts=np.linspace(0,t_AWG,N_AWG,endpoint=False)
    # ys=synthesize1(amp_list,f_list,ts,phase_list)
    # wavefile = (ys - min(ys)) / (max(ys) - min(ys)) - 0.5
    # ys = np.ones(len(wavefile)) * wavefile
    # print(ys)


    # amp_list()

    # ts = np.linspace(0, t_AWG, N_AWG, endpoint=False)

    # ys = synthesize1(amp_list, f_list, ts, phase_list)
    # print(f_list)
    # ys = (ys - min(ys)) / (max(ys) - min(ys)) * 2 ** 14

    # mif=get_mifile(ys)
    # txt = get_awgfile(ys)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

    plt.figure()
    plt.subplot(211)
    plt.plot(ts, ys, 'b')
    plt.xlabel("t（毫秒）")
    plt.ylabel("S(t)幅值")
    plt.title("叠加信号图")
    # plt.show()

    # fs,hz=get_fft(ys,N_FPGA)
    fs, hz = get_fft(ys, N_AWG)
    # angle_fs = np.angle(np.abs(np.abs(fft(ys))/N_FPGA))
    # angle_hz=np.arange(len(ys))
    plt.subplot(212)
    plt.plot(hz, fs, 'g')
    # plt.subplot(313)
    # plt.plot(angle_fs,angle_fs,'p')
    plt.xlabel("F（MHz）")
    plt.ylabel("A归一化")
    plt.title("PUMP频梳")
    plt.savefig('triangle.png')
    plt.show()

    # print(len(ys))
    # ys=np.ones(len(ys))*ys
    shape_list = ['Square', 'Triangle', 'Band_stop', 'Guass']
    ys = np.loadtxt('AWG_'+shape_list[shape] + 'CW='+str((center_F-CF_gap)/10**9) + 'GHz' +'BW='+ str((bandwidth-BW_gap) / (10 ** 6)) + 'MHz' + 'Df='+str((df-df_gap) / (10 ** 6)) + 'MHz' + '.txt'
    )
    # fft_ys = np.abs(np.abs(fft(ys)) / N_AWG)
    # fft_ys = scipy.fft.fft(ys)
    # fft_ys1 = abs(fft_ys)
    # fft_ys2 = abs(fft_ys1 / N_AWG)
    # fft_ys3 = fft_ys2[range(int(N_AWG / 2))]
    # Fs = np.arange(len(ys))
    FFT_y, Fre=get_fft(ys,AWG_framerate)
    #
    #
    # angle_ys = np.angle(fft_ys)
    #
    plt.figure()
    # plt.subplot(311)
    # plt.plot(ts, ys)
    # plt.subplot(312)
    # plt.xlim(14500, 15500)
    # plt.plot(Fs, fft_ys2)
    plt.plot(Fre,FFT_y)
    # plt.subplot(313)
    # plt.plot(Fs, angle_ys)
    plt.xlim(1.3e10,1.52e10)
    plt.show()
