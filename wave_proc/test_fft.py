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


def test2():
    sampling_rate = 8096  # 采样率
    fft_size = 1024  # FFT长度
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    x = np.sin(2 * np.pi * 156.25 * t) + 2 * np.sin(2 * np.pi * 234.375 * t) + 3 * np.sin(2 * np.pi * 200 * t)
    xs = x[:fft_size]

    xf = np.fft.rfft(xs) / fft_size  # 返回fft_size/2+1 个频率

    freqs = np.linspace(0, sampling_rate // 2, fft_size // 2 + 1)  # 表示频率
    xfp = np.abs(xf) * 2  # 代表信号的幅值，即振幅

    plt.figure(num='original', figsize=(15, 6))
    plt.plot(x[:fft_size])

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t[:fft_size], xs)
    plt.xlabel("t(s)")
    plt.title("156.25Hz and 234.375Hz waveform and spectrum")

    plt.subplot(212)
    plt.plot(freqs, xfp)
    plt.xlabel("freq(Hz)")
    plt.ylabel("amplitude")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def test3():
    N = 1024
    t = np.linspace(0, 2 * np.pi, N)
    x = 0.3 * np.cos(t) + 0.5 * np.cos(2 * t + np.pi / 4) + 0.8 * np.cos(3 * t - np.pi / 3)
    xf = np.fft.fft(x) / N
    freq = np.arange(N)

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, x)
    plt.xlabel("t(s)")
    plt.title("waveform and spectrum")

    plt.subplot(212)
    plt.plot(freq, np.abs(xf) * 2)
    plt.xlabel("freq(Hz)")
    plt.ylabel("amplitude")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def test4():
    Fs = 64
    f_o = [5, 15, 20]
    t = np.arange(0, 10, 1.0 / Fs)
    x = np.sum([np.sin(2 * np.pi * f1 * t) for f1 in f_o], axis=0)
    N = len(t)

    X = np.fft.fft(x)
    f = np.arange(N) * Fs * 1.0 / N

    f_shift = f - Fs / 2
    X_shift = np.fft.fftshift(X)  # 调整0频位置

    N_p = N // 2
    f_p = f_shift[N_p:]
    X_p = (np.abs(X_shift)[N_p:]) * 2

    x_r = np.fft.ifft(X)

    plt.figure(figsize=(8, 20))
    plt.subplot(511)
    plt.plot(t, x)
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Signal")

    plt.subplot(512)
    plt.plot(f, np.abs(X))
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("spectrum")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(513)
    plt.plot(f_shift, np.abs(X_shift))
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("spectrum")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(514)
    plt.plot(f_p, X_p)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("spectrum")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(515)
    plt.plot(t, np.real(x_r))
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("reconstruct Signal")

    plt.show()


def test5():
    F = 30
    Fs = 256
    Ts = 1 / Fs

    tc = np.arange(0, 5 / F, 1e-4)
    xc = np.cos(2 * np.pi * F * tc)
    td = np.arange(0, 5 / F, Ts)
    xd = np.cos(2 * np.pi * F * td)
    N = len(td)

    xr = np.sum([xd[i] * np.sinc(tc / Ts - i) for i in range(N)], axis=0)
    err = np.sqrt((xr - xc) ** 2)

    plt.figure(figsize=(8, 20))
    plt.subplot(311)
    plt.plot(tc, xc, label='Original')
    plt.plot(tc, xr, label='Reconstruct')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal(Original vs Reconstruct)")
    plt.legend()

    plt.subplot(312)
    plt.plot(td, xd, 'r--o')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Sampling Signal")

    plt.subplot(313)
    plt.plot(tc, err, 'm--o')
    plt.xlabel("t(s)")
    plt.ylabel("Err")
    plt.title("Err signal")

    plt.show()


def test6():
    def rect(t, T):
        Th = T / 2.0
        return np.where((t > -Th) & (t < Th), 1.0, 0.0)

    def rect2(t, tao, T):
        N = len(t)
        tao_h = tao / 2.0
        res = np.where((t > -tao_h) & (t < tao_h), 1.0, 0.0)
        k = 1
        while k * T < t[-1]:
            res += np.where((t > k * T - tao_h) & (t < k * T + tao_h), 1.0, 0.0)
            res += np.where((t > -k * T - tao_h) & (t < -k * T + tao_h), 1.0, 0.0)
            k += 1

        return res

    tc = np.arange(-2 * np.pi, 2 * np.pi, 1e-4)
    xc = np.cos(tc) + 1.0

    sc = rect2(tc, 0.5, np.pi / 4.0)
    xs = sc * xc

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(tc, xc, 'b--', label='Origin')
    plt.plot(tc, xs, 'r-', label='RectSam')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.legend(loc='lower left')

    plt.subplot(212)
    # plt.plot(tc, xc, 'r--', label='Original')
    plt.plot(tc, xs, 'b-', label='Reconstruct')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal(Original vs Reconstruct)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test6()
