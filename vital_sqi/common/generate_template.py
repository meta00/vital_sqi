"""Generating templates of ECG and PPG complexes"""
import numpy as np

from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import scipy
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp


def squeeze_template(s, width):
    """handy

    Parameters
    ----------
    s :
        param width:
    width :


    Returns
    -------


    """
    s = np.array(s)
    total_len = len(s)
    span_unit = 2
    out_res = []
    for i in range(int(width)):
        if i == 0:
            centroid = (total_len / width) * i
        else:
            centroid = (total_len / width) * i
        left_point = int(centroid) - span_unit
        right_point = int(centroid + span_unit)
        if left_point < 0:
            left_point = 0
        if right_point > len(s):
            left_point = len(s)
        out_res.append(np.mean(s[left_point:right_point]))
    return np.array(out_res)

def ppg_dual_double_frequency_template(width):
    """
    EXPOSE
    Generate a PPG template by using 2 sine waveforms.
    The first waveform double the second waveform frequency
    :param width: the sample size of the generated waveform
    :return: a 1-D numpy array of PPG waveform
    having diastolic peak at the low position
    """
    t = np.linspace(0, 1, width, False)  # 1 second
    sig = np.sin(2 * np.pi * 2 * t - np.pi / 2) + \
        np.sin(2 * np.pi * 1 * t - np.pi / 6)
    sig_scale = MinMaxScaler().fit_transform(np.array(sig).reshape(-1, 1))
    return sig_scale.reshape(-1)


def skew_func(x, e=0, w=1, a=0):
    """
    handy
    :param x: input sequence of time points
    :param e: location
    :param w: scale
    :param a: the order
    :return: a 1-D numpy array of a skewness distribution
    """
    t = (x - e) / w
    omega = (1 + erf((a * t) / np.sqrt(2))) / 2
    gaussian_dist = 1 / (np.sqrt(2 * np.pi)) * np.exp(-(t ** 2) / 2)
    return 2 / w * gaussian_dist * omega


def ppg_absolute_dual_skewness_template(width, e_1=1,
                                        w_1=2.5, e_2=3,
                                        w_2=3, a=4):
    """
    EXPOSE
    Generate a PPG template by using 2 skewness distribution.
    :param width: the sample size of the generated waveform
    :param e_1: the epsilon location of the first skew distribution
    :param w_1: the scale of the first skew distribution
    :param e_2: the epsilon location of the second skew distribution
    :param w_2: the scale of the second skew distribution
    :param a: the order
    :return: a 1-D numpy array of PPG waveform
    having diastolic peak at the high position
    """
    x = np.linspace(0, 11, width, False)
    p_1 = skew_func(x, e_1, w_1, a)
    p_2 = skew_func(x, e_2, w_2, a)
    p_ = np.max([p_1, p_2], axis=0)
    sig_scale = MinMaxScaler().fit_transform(np.array(p_).reshape(-1, 1))
    return sig_scale.reshape(-1)


def ppg_nonlinear_dynamic_system_template(width):
    """
    EXPOSE
    :param width:
    :return:
    """
    x1 = 0.15
    x2 = 0.15
    u = 0.5
    beta = 1
    gamma1 = -0.25
    gamma2 = 0.25
    x1_list = [x1]
    x2_list = [x2]

    dt = 0.1
    for t in np.arange(1, 100, dt):
        y1 = 0.5 * (np.abs(x1 + 1) - np.abs(x1 - 1))
        y2 = 0.5 * (np.abs(x2 + 1) - np.abs(x2 - 1))
        dx1 = -x1 + (1 + u) * y1 - beta * y2 + gamma1
        dx2 = -x2 + (1 + u) * y2 + beta * y1 + gamma2

        x1 = x1 + dx1 * dt
        x2 = x2 + dx2 * dt

        x1_list.append(x1)
        x2_list.append(x2)

    local_minima = argrelextrema(np.array(x2_list), np.less)[0]
    s = np.array(x2_list[local_minima[-2]:local_minima[-1] + 1])

    rescale_signal = squeeze_template(s, width)

    window = signal.windows.cosine(len(rescale_signal), 0.5)
    signal_data_tapered = np.array(window) * \
        (rescale_signal - min(rescale_signal))

    out_scale = MinMaxScaler().fit_transform(
        np.array(signal_data_tapered).reshape(-1, 1))
    return out_scale.reshape(-1)


def interp(ys, mul):
    """
    handy func
    :param ys:
    :param mul:
    :return:
    """
    # linear extrapolation for last (mul - 1) points
    ys = list(ys)
    ys.append(2 * ys[-1] - ys[-2])
    # make interpolation function
    xs = np.arange(len(ys))
    fn = scipy.interpolate.interp1d(xs, ys, kind="cubic")
    # call it on desired data points
    new_xs = np.arange(len(ys) - 1, step=1. / mul)
    return fn(new_xs)


"""
Equation (3) from the paper
A dynamical model for generating synthetic electrocardiogram signals
"""


def ecg_dynamic_template(width, sfecg=256, N=256, Anoise=0, hrmean=60,
                         hrstd=1, lfhfratio=0.5, sfint=512,
                         ti=np.array([-70, -15, 0, 15, 100]),
                         ai=np.array([1.2, -5, 30, -7.5, 0.75]),
                         bi=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
                         ):
    """
    EXPOSE
    :param width:
    :param sfecg:
    :param N:
    :param Anoise:
    :param hrmean:
    :param hrstd:
    :param lfhfratio:
    :param sfint:
    :param ti:
    :param ai:
    :param bi:
    :return:
    """
    # convert to radians
    ti = ti * np.pi / 180

    # adjust extrema parameters for mean heart rate
    hrfact = np.sqrt(hrmean / 60)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * bi
    ti = np.multiply([hrfact2, hrfact, 1, hrfact, hrfact2], ti)

    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # calculate time scales for rr and total output
    sampfreqrr = 1
    trr = 1 / sampfreqrr
    rrmean = (60 / hrmean)
    Nrr = 2 ** (np.ceil(np.log2(N * rrmean / trr)))

    rr0 = rr_process(flo, fhi, flostd, fhistd,
                     lfhfratio, hrmean, hrstd, sampfreqrr, Nrr)

    # upsample rr time series from 1 Hz to sfint Hz
    rr = interp(rr0, sfint)
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tecg = 0
    i = 0
    while i < len(rr):
        tecg = tecg + rr[i]
        ip = int(np.round(tecg / dt))
        rrn[i: ip + 1] = rr[i]
        i = ip + 1
    Nt = ip
    x0 = [1, 0, 0.04]
    tspan = np.arange(0, (Nt - 1) * dt, dt)
    args = (rrn, sfint, ti, ai, bi)
    solv_ode = solve_ivp(ordinary_differential_equation, [tspan[0], tspan[-1]],
                         x0, t_eval=np.arange(20.5, 21.5, 0.00001), args=args)
    Y = (solv_ode.y)[2]

    # if len(Y) > width:
    #     z = squeeze_template(Y,125)
    return Y


def ordinary_differential_equation(t, x_equations, rr=None,
                                   sfint=None, ti=None, ai=None, bi=None):
    """
    handy
    :param t:
    :param x_equations:
    :param rr:
    :param sfint:
    :param ti:
    :param ai:
    :param bi:
    :return:
    """
    x = x_equations[0]
    y = x_equations[1]
    z = x_equations[2]

    ta = np.arctan2(y, x)
    r0 = 1
    a0 = 1.0 - np.sqrt(x ** 2 + y ** 2) / r0
    ip = int(1 + np.floor(t * sfint))
    try:
        w0 = 2 * np.pi / rr[ip]
    except Exception:
        w0 = 2 * np.pi / rr[-1]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * x - w0 * y
    dx2dt = a0 * y + w0 * x

    dti = np.fmod(ta - ti, 2 * np.pi)
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * np.divide(dti, bi) ** 2))
    dx3dt = dx3dt - 1.0 * (z - zbase)

    return [dx1dt, dx2dt, dx3dt]


def rr_process(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n):
    """
    handy
    :param flo:
    :param fhi:
    :param flostd:
    :param fhistd:
    :param lfhfratio:
    :param hrmean:
    :param hrstd:
    :param sfrr:
    :param n:
    :return:
    """
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)
    """
    Generating RR-intervals which have a bimodal power spectrum
    consisting of the sum of two Gaussian distributions
    """
    df = sfrr / n
    w = np.arange(0, n).T * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / \
        np.sqrt(2 * np.pi * np.power(c1, 2))
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / \
        np.sqrt(2 * np.pi * np.power(c2, 2))
    Hw = Hw1 + Hw2
    """
    An RR-interval time series T(t)
    with power spectrum is S(f)
    generated by taking the inverse Fourier transform of
    a sequence of complex numbers with amplitudes sqrt(S(f))
    and phases which are randomly distributed between 0 and 2pi
    """
    Hw0_half = np.array(Hw[0:int(n / 2)])
    Hw0 = np.append(Hw0_half, np.flip(Hw0_half))
    Sw = (sfrr / 2) * (Hw0 ** .5)

    ph0 = 2 * np.pi * np.random.rand(int(n / 2) - 1, 1)
    # ph0 = 2 * np.pi * 0.001*np.arange(127).reshape(-1,1)
    ph = np.vstack((0, ph0, 0, -np.flip(ph0)))

    # create the complex number
    SwC = np.multiply(Sw.reshape(-1, 1), np.exp(1j * ph))
    inverse_res = np.fft.ifft(SwC.reshape(-1))
    x = (1 / n) * np.real(inverse_res)

    """
        By multiplying this time series by an appropriate scaling constant
        and adding an offset value, the resulting time series can be given
        any required mean and standard deviation
    """
    xstd = np.std(x)
    ratio = rrstd / xstd
    rr = rrmean + x * ratio
    return rr