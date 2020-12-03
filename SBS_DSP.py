import math
import numpy as np
import random
import scipy
# import numpy.fft as fft
import os
import shutil

from scipy.fftpack import fft, ifft

import matplotlib.pyplot as plt

# from thinkdsp import CosSignal, SumSignal
PI2 = math.pi * 2


def randen_phase():
    # list = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    list = [0,np.pi / 2,np.pi, 3 * np.pi / 2]
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
    start_F = center_F - bandwidth / 2
    # end_F=center_F+bandwidth/2
    dots = int(bandwidth / df + 1)
    # a = np.random.randint(0, df / (10 ** 6) + 1)
    f_list = []
    amp_list = np.empty(dots)
    phase_list = []
    i = 0
    while i < dots:
        # f_list.append(start_F + i * df + a * 10 ** 6)
        f_list.append(start_F + i * df)
        amp_list[i] = 2
        phase_list.append(randen_phase())
        i = i + 1

    return f_list, amp_list, phase_list


def triangle_filter(center_F, bandwidth, df):
    dots = int(bandwidth / df + 1)
    start_F = center_F - bandwidth / 2
    end_F = center_F + (center_F - start_F)
    # a = np.random.randint(0, df / (10 ** 6) + 1)
    f_list = []
    amp_list = np.empty(dots)
    phase_list = []
    i = 0
    j = int(dots / 2 + 1)
    while i < dots:
        # f_list.append(start_F + i * df + a * 10 ** 6)
        f_list.append(start_F + i * df)
        phase_list.append(randen_phase())
        if i < j:
            amp_list[i] = i * df * 2 / (center_F - start_F)
            # continue
        else:
            amp_list[i] = (end_F - start_F - i * df) * 2 / (start_F - center_F)
            # continue

        i = i + 1

    return f_list, amp_list, phase_list


def Band_stop_filter(center_F, bandwidth, signal_BW, df):
    start_F = center_F - signal_BW / 2
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
            amp_list[i] = 0.2
        else:
            amp_list[i] = 2
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


def get_fft(ys, N):
    p = abs(fft(ys))
    p1 = p / N
    p2 = p1[range(int(N / 2))]
    # p2=p1
    hz = np.arange(len(ys))
    hz1 = hz
    hz2 = hz1[range(int(N / 2))]
    # hz2=np.fft.fftfreq(int(N),1)

    return p2, hz2


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


def get_awgfile(ys):

    # ys = (ys - min(ys)) / (max(ys) - min(ys))-0.5
    # ys=1.0/(1+np.exp(-ys))
    mean=np.average(ys)
    sigma=np.std(ys)
    ys=(ys-mean)/sigma
    # txt = 'AWG_cos_triangle.txt'
    txt='AWG_cos_square.txt'
    with open(txt, 'w') as f:
        for item in ys:
            f.write(str(item))
            f.write('\n')

    return txt


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

    f_list, amp_list, phase_list = square_filter(center_F=15 * 10 ** 9, bandwidth=400 * 10 ** 6, df=15 * 10 ** 6)
    # f_list, amp_list, phase_list=triangle_filter(center_F=20*10**9,bandwidth=200*10**6,df=15*10**6)
    # f_list, amp_list, phase_list=Band_stop_filter(center_F=10*10**9, bandwidth=3*10**9, signal_BW=5*10**9, df=10**7)
    # 以MHz为单位，频梳间隔为15~20MHz
    # ts = np.linspace(0,t_FPGA,N_FPGA,endpoint=False)
    ts = np.linspace(0, t_AWG, N_AWG, endpoint=False)

    ys = synthesize1(amp_list, f_list, ts, phase_list)
    print(f_list)
    # ys = (ys - min(ys)) / (max(ys) - min(ys)) * 2 ** 14

    # mif=get_mifile(ys)
    txt = get_awgfile(ys)

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
    ys=np.loadtxt('AWG_cos_square.txt')
    fft_ys=np.abs(np.abs(fft(ys))/N_AWG)
    fft_ys=scipy.fft.fft(ys)
    fft_ys1=abs(fft_ys)
    fft_ys2=abs(fft_ys1/N_AWG)
    fft_ys3=fft_ys2[range(int(N_AWG/2))]
    Fs=np.arange(len(ys))
    #
    #
    # angle_ys = np.angle(fft_ys)
    #
    plt.figure()
    # plt.subplot(311)
    # plt.plot(ts, ys)
    # plt.subplot(312)
    plt.plot(Fs, fft_ys2)
    # plt.subplot(313)
    # plt.plot(Fs, angle_ys)
    plt.show()
