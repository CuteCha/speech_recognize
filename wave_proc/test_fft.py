#! /usr/bin/python3
# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def test1():
    '''
    离散傅里叶变换
    一维时序信号y，它由2V的直流分量(0Hz)，和振幅为3V，频率为50Hz的交流信号，以及振幅为1.5V，频率为75Hz的交流信号组成：
    y = 2 + 3*np.cos(2*np.pi*50*t) + 1.5*np.cos(2*np.pi*75*t)
    然后我们采用256Hz的采样频率，总共采样256个点。
    '''
    fs = 256  # 采样频率， 要大于信号频率的两倍
    t = np.arange(0, 1, 1.0 / fs)  # 1秒采样fs个点
    N = len(t)
    freq = np.arange(N)  # 频率counter

    # x = 2 + 3 * cos(2 * pi * 50 * t) + 1.5 * cos(2 * pi * 75 * t)  # 离散化后的x[n]
    x = 2 + 3 * np.cos(2 * np.pi * 10 * t) + 1.5 * np.cos(2 * np.pi * 15 * t)  # 离散化后的x[n]

    X = np.fft.fft(x)  # 离散傅里叶变换

    '''
    根据STFT公式原理，实现的STFT计算，做了/N的标准化
    '''
    X2 = np.zeros(N, dtype=np.complex)  # X[n]
    for k in range(0, N):  # 0,1,2,...,N-1
        for n in range(0, N):  # 0,1,2,...,N-1
            # X[k] = X[k] + x[n] * np.exp(-2j * pi * k * n / N)
            X2[k] = X2[k] + (1 / N) * x[n] * np.exp(-2j * np.pi * k * n / N)

    fig, ax = plt.subplots(5, 1, figsize=(12, 12))

    # 绘制原始时域图像
    ax[0].plot(t, x, label='原始时域信号')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(freq, abs(X), 'r', label='调用np.fft库计算结果')
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()

    ax[2].plot(freq, abs(X2), 'r', label='根据STFT计算结果')
    ax[2].set_xlabel('Freq (Hz)')
    ax[2].set_ylabel('Amplitude')
    ax[2].legend()

    X_norm = X / (N / 2)  # 换算成实际的振幅
    X_norm[0] = X_norm[0] / 2
    ax[3].plot(freq, abs(X_norm), 'r', label='转换为原始信号振幅')
    ax[3].set_xlabel('Freq (Hz)')
    ax[3].set_ylabel('Amplitude')
    ax[3].set_yticks(np.arange(0, 3))
    ax[3].legend()

    freq_half = freq[range(int(N / 2))]  # 前一半频率
    X_half = X_norm[range(int(N / 2))]

    ax[4].plot(freq_half, abs(X_half), 'b', label='前N/2个频率')
    ax[4].set_xlabel('Freq (Hz)')
    ax[4].set_ylabel('Amplitude')
    ax[4].set_yticks(np.arange(0, 3))
    ax[4].legend()

    plt.show()


if __name__ == "__main__":
    test1()
