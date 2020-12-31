# 产生AWGN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import rv_continuous
import random as rd
from scipy.fftpack import fft, ifft


def awn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.rand(len(x)) * np.sqrt(npower)


if __name__ == '__main__':
    t = np.arange(0, 1000000) * 0.1
    x = np.sin(t)
    n = awn(x, 6)
    xn = x + n  # 增加了6dBz信噪比噪声的信号
    plt.subplot(211)
    plt.hist(n, bins=100, density=True)
    plt.subplot(212)
    plt.psd(n)
    plt.show()
