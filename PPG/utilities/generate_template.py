import numpy as np

from scipy.special import erf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import scipy.integrate as integrate
from scipy import signal
from scipy.signal import argrelextrema


def smooth(x, window_len=11, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

# def erf(z):
#     func = lambda x: np.power(np.e, -x ** 2)
#     if not isinstance(z,int):
#         z_res = []
#         for z_ in z:
#             z_res.append(2/np.pi * integrate.quad(func,0,z_)[0])
#         return np.array(z_res)
#     return 2/np.pi * integrate.quad(func,0,z)[0]

# def omega(x):
#     return 0.5*(1+erf(x/np.sqrt(2)))

# def omega(x):
#     if not isinstance(x, int):
#         z_res = []
#         for z in x:
#             z_res.append(integrate.quad(gaussian_func, z,-np.inf)[0])
#         return np.array(z_res)
#     return integrate.quad(gaussian_func, 0, x)[0]

def ppg_dual_doublde_frequency_template(width):
    t = np.linspace(0, 1, width, False)  # 1 second
    sig = np.sin(2 * np.pi * 2 * t - np.pi / 2) + np.sin(2 * np.pi * 1 * t - np.pi / 4)
    sig_scale = MinMaxScaler().fit_transform(np.array(sig).reshape(-1, 1))
    return sig_scale.reshape(-1)

def compute_omega(x):
    return (1 + erf(x/np.sqrt(2))) / 2

def gaussian_func(x): #aka pdf
    # 1. / (np.sqrt(2. * np.pi)) * np.exp(-.5 * (x1) ** 2)
    return 1/(np.sqrt(2*np.pi)) * np.exp(-(x**2)/2)
    # return 1/(np.sqrt(2*np.pi))*np.power(np.e,-(x**2)/2)

# def pdf(x):
#     return 1/sqrt(2*pi) * exp(-x**2/2)

# def skew_func(x,alpha=1):
#     return 2*gaussian_func(x)*omega(alpha*x)

def skew_func(x,e=0,w=1,a=0):
    """
    :param x: input
    :param e: location
    :param w: scale
    :param a:
    :return:
    """
    t = (x-e) / w
    return 2 / w * gaussian_func(t) * compute_omega(a*t)

def ppg_absolute_dual_skewness_template(width,e_1=1,w_1=2.5,e_2=3,w_2=3,a=4):
    x = np.linspace(0, 11, width, False)
    p_1 = skew_func(x,e_1,w_1,a)
    p_2 = skew_func(x,e_2,w_2,a)
    p_ = np.max([p_1, p_2], axis=0)
    sig_scale = MinMaxScaler().fit_transform(np.array(p_).reshape(-1, 1))
    return sig_scale.reshape(-1)

def squeeze_template(s,width):
    s = np.array(s)
    total_len = len(s)
    span_unit = 2
    out_res = []
    for i in range(width):
        if i == 0:
            centroid = (total_len/width)*i
        else:
            centroid = (total_len/width)*i
        left_point = int(centroid)-span_unit
        right_point = int(centroid+span_unit)
        if left_point <0:
            left_point=0
        if right_point >len(s):
            left_point=len(s)
        out_res.append(np.mean(s[left_point:right_point]))
    return np.array(out_res)

def ppg_nonlinear_dynamic_system_template(width):
    x1 = 0.15
    x2 = 0.15
    u = 0.5
    beta = 1
    gamma1 = -0.25
    gamma2 = 0.25
    x1_list = [x1]
    x2_list = [x2]

    dt = 0.1
    for t in np.arange(1,width,dt):
        y1 = 0.5 * (np.abs(x1+1) - np.abs(x1-1))
        # y1 = np.tanh(x1*2)
        y2 = 0.5 * (np.abs(x2 + 1) - np.abs(x2 - 1))
        # y2 = np.tanh(x2 * 2)
        dx1 = -x1 + (1 + u) * y1 - beta * y2 + gamma1
        dx2 = -x2 + (1 + u) * y2 + beta * y1 + gamma2

        x1 = x1 + dx1*dt
        x2 = x2 + dx2*dt

        x1_list.append(x1)
        x2_list.append(x2)

    local_minima = argrelextrema(np.array(x2_list), np.less)[0]
    # local_minima = np.where(x2_list==np.min(x2_list))[0]
    s = np.array(x2_list[local_minima[-2]:local_minima[-1]+1])

    rescale_signal = squeeze_template(s,width)

    window = signal.windows.cosine(len(rescale_signal), 0.5)
    signal_data_tapered = np.array(window) * (rescale_signal - min(rescale_signal))

    out_scale = MinMaxScaler().fit_transform(np.array(signal_data_tapered).reshape(-1, 1))
    return out_scale.reshape(-1)

def rescale(arr, width=50):
    n = len(arr)
    factor = int(np.ceil(width/n))
    return np.interp(np.linspace(0, n, factor*n+1), np.arange(n), arr)

def scale_pattern(s,window_size):
    scale_res = []
    if len(s) == window_size:
        return np.array(s)
    if len(s)<window_size:
        #spanning the signal
        span_ratio = (window_size/len(s))
        for idx in range(0,int(window_size)):
            if idx-span_ratio<0:
                scale_res.append(s[0])
            else:
                scale_res.append(np.mean(s[int(idx/span_ratio)]))
    else:
        squeeze_ratio = int(np.ceil(len(s)/window_size))
        for idx in range(0,int(window_size)):
            if idx-squeeze_ratio<0:
                scale_res.append(np.mean(s[:idx+squeeze_ratio]))
            elif idx+squeeze_ratio>=window_size:
                scale_res.append(np.mean(s[idx - squeeze_ratio:]))
            else:
                scale_res.append(np.mean(s[idx - squeeze_ratio:idx + squeeze_ratio]))
    scale_res = smooth_window(scale_res, span_size=5)
    return np.array(scale_res)

def smooth_window(s,span_size=5):
    for i in range(0,len(s)):
        if i-span_size<0:
            s[i] = np.mean(s[:i+span_size])
        elif i+span_size>=len(s):
            s[i] = np.mean(s[i-span_size:])
        else:
            s[i] = np.mean(s[i-span_size:i+span_size])
    return s

def ecg_dynamic_equation_template(width):

    times = [-0.2,-0.05,0,0.05,0.3] #PQRST
    theta_i = [-np.pi/3,-np.pi/12,0,np.pi/12,np.pi/2]
    a_i = [1.2,-5,30,-7.5,0.75]
    b_i = [0.25,0.1,0.1,0.1,0.4]

    A = 0.15
    f2 = 0.25
    t = 0.1
    x0 = 0
    y0 = 0
    z0 = A * np.sin(2 * np.pi * f2 * t)

    z = z0
    x = x0
    y = y0

    alpha = 1 - np.sqrt(x ** 2 + y ** 2)
    omega = 2*np.pi/times
    theta = np.arctan2(y, x)

    for dt in np.arange(-0.5,0.5,0.01):

        dx = alpha * x - omega * y
        dy = alpha * y + omega * x
        dz = compute_dz(theta,theta_i,a_i,b_i,z,z0)


    return z

def compute_dz(theta,theta_i,a_i,b_i,z,z0):
    dz = 0
    for i in range(len(theta_i)): #PQRST
        delta_theta = (theta-theta_i[i])//(2*np.pi)
        dz = dz - a_i[i]*delta_theta*np.exp\
            (-(theta_i[i]**2)/(2*(b_i[i]**2)))
    dz = dz - (z-z0)
    return dz

if __name__ == "__main__":
    width = 50 #any number

    # first template window
    template_1 = ppg_absolute_dual_skewness_template(width, e_1=2, w_1=2, e_2=3.5)

    # second template window
    template_2 = ppg_dual_doublde_frequency_template(width)

    template_3 = ppg_nonlinear_dynamic_system_template(width)

    plt.plot(template_1)
    plt.plot(template_2)
    plt.plot(template_3)

    plt.show()