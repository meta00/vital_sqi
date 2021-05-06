import numpy as np

def sdnn_sqi(nn_intervals):
    """
    Function returning the standard deviation of the NN intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The standard deviation of the NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    return np.std(nn_intervals, ddof=1)

def sdsd_sqi(nn_intervals):
    """
    Function returning the standard deviation of the successive differences from NN intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The standard deviation of the differences of NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    # get successive differences
    sd = np.diff(nn_intervals)
    return np.std(sd)

def rmssd_sqi(nn_intervals):
    """
    Function returning the root mean square of the successive differences from NN intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The root mean square of the differences of NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    sd = np.diff(nn_intervals)
    return np.sqrt(np.mean(sd**2))

def cvsd_sqi(nn_intervals):
    """
    Function returning the covariance successive differences (differences of the NN intervals)

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The covariance of the differences of NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.

    :param nn_intervals:
    :return:
    """
    cvsd = rmssd_sqi(nn_intervals) / mean_nn_sqi(nn_intervals)
    return cvsd

def cvnn_sqi(nn_intervals, compute_interpolation=False):
    """
    Function returning the covariance of the NN intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The covariance of the NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.

    """
    return sdsd_sqi(nn_intervals)/mean_nn_sqi(nn_intervals)

def mean_nn_sqi(nn_intervals, compute_interpolation=False):
    """
    Function returning the mean of the NN intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The mean of the NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    return np.mean(nn_intervals)

def median_nn_sqi(nn_intervals, compute_interpolation=False):
    """
    Function returning the median of the NN intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The median of the NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    return np.median(nn_intervals)

def pnn_50_sqi(nn_intervals, compute_interpolation=False):
    """
    Function returning the percentage of nn intervals
    that exceed the previous 50ms

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The percentage of the outlier NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    sd = np.diff(nn_intervals)
    nn_50 = np.sum(np.abs(sd) > 50)
    pnn_50 = 100 * nn_50 / len(sd)
    return pnn_50

def pnn_20_sqi(nn_intervals, compute_interpolation=False):
    """
    Function returning the percentage of nn intervals
    that exceed the previous 20ms

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The percentage of the outlier NN intervals

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    sd = np.diff(nn_intervals)
    nn_20 = np.sum(np.abs(sd) > 20)
    pnn_20 = 100 * nn_20 / len(sd)
    return pnn_20

def hr_mean_sqi(nn_intervals):
    """
    Function returning the mean heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The mean heart rate

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    nn_bpm = np.divide(60000,nn_intervals)
    return np.mean(nn_bpm)

def hr_min_sqi(nn_intervals):
    """
    Function returning the min heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The minimum heart rate

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """

    nn_bpm = np.divide(60000, nn_intervals)
    return np.min(nn_bpm)

def hr_max_sqi(nn_intervals):
    """
    Function returning the max heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The maximum heart rate

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    nn_bpm = np.divide(60000, nn_intervals)
    return np.max(nn_bpm)

def hr_std_sqi(nn_intervals):
    """
    Function returning the standard deviation of the heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    : float
        The standard deviation of the heart rate

    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    nn_bpm = np.divide(60000, nn_intervals)
    return np.std(nn_bpm)

def peak_frequency_sqi(freqs,pows,f_min=0.04,f_max=0.15):
    """
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ---------
    freqs : list
        The frequencies mapping with the power spectral
    pows : list
        The powers of the relevant frequencies
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max: float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    ---------
    : float
        The frequency having greatest power in the examined band

    Notes
    ---------
    Compute the PSD before using this sqi
    """
    f_power = (pows[freqs>=f_min & freqs<f_max])
    f_peak = f_power[np.argmax(f_pows)]
    return f_peak

def absolute_power_sqi(freqs,pows,f_min=0.04,f_max=0.15):
    """
    Compute the cummulative power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ---------
    freqs : list
        The frequencies mapping with the power spectral
    pows : list
        The powers of the relevant frequencies
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max: float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    ---------
    : float
        The cummulative power of the examined band

    Notes
    ---------
    Compute the PSD before using this sqi
    """
    filtered_pows = pows[freqs >= f_min and freqs < f_max]
    abs_pow = np.sum(filtered_pows)
    return abs_pow

def log_power_sqi(freqs,pows,f_min=0.04,f_max=0.15):
    """
    Compute the logarithm power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ---------
    freqs : list
        The frequencies mapping with the power spectral
    pows : list
        The powers of the relevant frequencies
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max: float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    ---------
    : float
        The logarithmic power of the examined band

    Notes
    ---------
    Compute the PSD before using this sqi
    """
    filtered_pows = pows[freqs >= f_min and freqs < f_max]
    log_pow = np.sum(np.log(filtered_pows))
    return log_pow

def relative_power_sqi(freqs,pows,f_min=0.04,f_max=0.15):
    """
    Compute the relative power with respect to the total power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ---------
    freqs : list
        The frequencies mapping with the power spectral
    pows : list
        The powers of the relevant frequencies
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max: float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    ---------
    : float
        Relative power with respect to the total power

    Notes
    ---------
    Compute the PSD before using this sqi
    """
    filtered_pows = pows[freqs >= f_min and freqs < f_max]
    relative_pow = np.sum(np.log(filtered_pows))/np.sum(pows)
    return relative_pow

def normalized_power_sqi(freqs,pows,
                    lf_min=0.04,lf_max=0.15,
                    hf_min=0.15,hf_max=0.4):
    """
    Compute the relative power with respect to the total power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ---------
    freqs : list
        The frequencies mapping with the power spectral
    pows : list
        The powers of the relevant frequencies
    lf_min : float
        the lower bound of the low-frequency band
    lf_max: float
        the upper bound of the low-frequency band
    hf_min : float
        the lower bound of the high-frequency band
    hf_max: float
        the upper bound of the high-frequency band

    Returns
    ---------
    : float
        Relative power with respect to the total power

    Notes
    ---------
    Compute the PSD before using this sqi
    """
    lf_filtered_pows = pows[freqs >= lf_min & freqs < lf_max]
    hf_filtered_pows = pows[freqs >= hf_min & freqs < hf_max]
    lf_power = np.sum(lf_filtered_pows)
    hf_power = np.sum(hf_filtered_pows)
    return np.linalg.norm(lf_power,hf_power)

def lf_hf_ratio_sqi(freqs,pows,
                    lf_min=0.04,lf_max=0.15,
                    hf_min=0.15,hf_max=0.4):
    """
    Compute the ratio power between the lower frequency and the high frequency.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ---------
    freqs : list
        The frequencies mapping with the power spectral
    pows : list
        The powers of the relevant frequencies
    lf_min : float
        the lower bound of the low-frequency band
    lf_max: float
        the upper bound of the low-frequency band
    hf_min : float
        the lower bound of the high-frequency band
    hf_max: float
        the upper bound of the high-frequency band

    Returns
    ---------
    : float
        Ratio power between low-frequency power and high-frequency power

    Notes
    ---------
    Compute the PSD before using this sqi
    """
    lf_filtered_pows = pows[freqs >= lf_min & freqs < lf_max]
    hf_filtered_pows = pows[freqs >= hf_min & freqs < hf_max]
    ratio = np.sum(lf_filtered_pows)/np.sum(hf_filtered_pows)
    return ratio

def poincare_features_sqi(nn_intervals):
    """
    Function returning the poincare features of mapping nn intervals

    Parameters
    ---------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    ---------
    sd1 : float
        The standard deviation of the second group
    sd2 : float
        The standard deviation of the second group
    area: float
        The area of the bounding eclipse
    ratio: float
        The ratio between sd1 and sd2
    Notes
    ---------
    If the purpose is to compute the HRV feature, the input
    must pass the preprocessing steps - remove the invalid peaks then do the
    interpolation - to obtain the normal to normal intervals.

    If the purpose is to compute SQI, input the raw RR intervals -
    obtained from the peak detection algorithm.
    """
    group_i = nn_intervals[:-1]
    group_j = nn_intervals[1:]

    sd1 = np.std(group_j-group_i)
    sd2 = np.std(group_j+group_i)

    area = np.pi * sd1 * sd2
    ratio = sd1/sd2

    return sd1,sd2,area,ratio