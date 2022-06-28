"""Heart rate variability SQIs
This module allows to compute time and frequency domain HRV
to use those as signal quality indexes, including:
HR
- HR: Mean, median, min, max, std of heart rate
- HR: Ratio of HR out of a defined range
HRV time domain
- SDNN
- SDSD
- RMSSD
- CVSD
- CVNN
- mean NN
- median NN
- pNNx
HRV frequency domain
- Peak frequency
- Absolute power
- Log power
- Relative power
- Normalised power
- Lf Hf ratio
- Poincare features
"""

import numpy as np
from vital_sqi.common.power_spectrum import calculate_psd
import warnings
from hrvanalysis import get_time_domain_features, \
    get_frequency_domain_features,  get_nn_intervals, get_csi_cvi_features, \
    get_geometrical_features
import sys,os
from vital_sqi.common.rpeak_detection import PeakDetector


def nn_mean_sqi(nn_intervals):
    """

    Parameters
    ----------
    nn_intervals :
        

    Returns
    -------
    float
        The arithmetic mean of NN intervals.
    
    """

    return np.mean(nn_intervals)


def sdnn_sqi(nn_intervals):
    """Function returning the standard deviation of the NN intervals

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The standard deviation of the NN intervals
    """
    return np.std(nn_intervals, ddof=1)


def sdsd_sqi(nn_intervals):
    """Function returning the standard deviation of the successive differences
    from NN intervals.

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The standard deviation of the differences of NN intervals
    """
    # get successive differences
    sd = np.diff(nn_intervals)
    return np.std(sd)


def rmssd_sqi(nn_intervals):
    """Function returning the root mean square of the successive differences
    from NN intervals.

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The root mean square of the differences of NN intervals
    """
    sd = np.diff(nn_intervals)
    return np.sqrt(np.mean(sd**2))


def cvsd_sqi(nn_intervals):
    """Function returning the covariance successive differences (differences of
    the NN intervals)

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The covariance of the differences of NN intervals
    """
    cvsd = rmssd_sqi(nn_intervals) / mean_nn_sqi(nn_intervals)
    return cvsd


def cvnn_sqi(nn_intervals):
    """Function returning the covariance of the NN intervals

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    interpolation :
        interpolation: bool
        Options to do interpolation of NN interval before calculating covariance of NN.

    Returns
    -------
    float
        The covariance of the NN intervals
    """
    return sdsd_sqi(nn_intervals)/mean_nn_sqi(nn_intervals)


def mean_nn_sqi(nn_intervals):
    """Function returning the mean of the NN intervals

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The mean of the NN intervals
    """
    return np.mean(nn_intervals)


def median_nn_sqi(nn_intervals):
    """Function returning the median of the NN intervals

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The median of the NN intervals
    """
    return np.median(nn_intervals)


def pnn_sqi(nn_intervals, exceed=50):
    """Function returning the percentage of nn intervals
    that exceed the previous 50ms

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    exceed : int
        Number of ms that is different among successive NN
        intervals, e.g., 50 for pNN50.

    Returns
    -------
    float
        The percentage of the outlier NN intervals
    """
    sd = np.diff(nn_intervals)
    nn_exceed = np.sum(np.abs(sd) >= exceed)
    pnn_exceed = 100 * nn_exceed / len(sd)
    return pnn_exceed


def hr_mean_sqi(nn_intervals):
    """Function returning the mean heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    int
        The mean heart rate
    """
    nn_bpm = np.divide(60000, nn_intervals)
    return int(np.round(np.mean(nn_bpm)))


def hr_median_sqi(nn_intervals):
    """Function returning the median heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    int
        The median heart rate
    """
    nn_bpm = np.divide(60000, nn_intervals)
    return int(np.round(np.median(nn_bpm)))


def hr_min_sqi(nn_intervals):
    """Function returning the min heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    int
        The minimum heart rate
    """

    nn_bpm = np.divide(60000, nn_intervals)
    return int(np.round(np.min(nn_bpm)))


def hr_max_sqi(nn_intervals):
    """Function returning the max heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    int
        The maximum heart rate

    """
    nn_bpm = np.divide(60000, nn_intervals)
    return int(np.round(np.max(nn_bpm)))


def hr_std_sqi(nn_intervals):
    """Function returning the standard deviation of the heart rate .
    The input nn_interval in ms is converted into
    heart rate bpm (beat per minute) unit

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    float
        The standard deviation of the heart rate

    """
    nn_bpm = np.divide(60000, nn_intervals)
    return np.std(nn_bpm)


def hr_range_sqi(nn_intervals, range_min=40, range_max=200):
    """Percentage of heart beats that are out of defined range.

    Parameters
    ----------
    nn_intervals :

    range_min :
         (Default value = 40)
    range_max :
         (Default value = 200)

    Returns
    -------
        float
            The percentage of heart rate out of range with decimal 2.
    """
    nn_bpm = np.divide(60000, nn_intervals)
    out = sum(range_min >= nn_bpm) + sum(nn_bpm >= range_max)
    out = round(100*out/len(nn_bpm), 2)
    return out


def peak_frequency_sqi(nn_intervals, freqs=None, pows=None, f_min=0.04, f_max=0.15):
    """The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    freqs : list
        The frequencies mapping with the power spectral.
        Default is None.
    pows : list
        The powers of the relevant frequencies (Default value = None)
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max : float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    -------
    float
        The frequency having greatest power in the examined band

    Notes
    ---------
    If freqs and pows are assigned by computing the PSD before using this sqi,
    freqs & pows is used directly instead of nn_intervals.
    Otherwise, the frequencies and powers will be computed from nn intervals
    using welch method as default
    """
    if freqs is None or pows is None:
        freqs, pows = calculate_psd(nn_intervals)
    assert len(freqs) == len(pows), \
        "Length of the frequencies and the relevant powers must be the same"
    # f_power = (pows[f_min <= freqs < f_max])
    f_power = pows[np.where((f_min <= freqs) & (freqs < f_max))[0]]
    f_peak = f_power[np.argmax(f_power)]
    return f_peak


def absolute_power_sqi(nn_intervals, freqs=None, pows=None, f_min=0.04, f_max=0.15):
    """Compute the cummulative power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    freqs : list
        The frequencies mapping with the power spectral (Default value = None)
    pows : list
        The powers of the relevant frequencies (Default value = None)
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max : float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    -------
    float
        The cummulative power of the examined band

    Notes
    ---------
    If freqs and pows are assigned by computing the PSD before using this sqi,
    freqs & pows is used directly instead of nn_intervals.
    Otherwise, the frequencies and powers will be computed from nn intervals
    using welch method as default
    """
    if freqs is None or pows is None:
        freqs, pows = calculate_psd(nn_intervals)
    assert len(freqs) == len(pows), \
        "Length of the frequencies and the relevant powers must be the same"
    filtered_pows = pows[np.where((f_min <= freqs) & (freqs < f_max))[0]]
    abs_pow = np.sum(filtered_pows)
    return abs_pow


def log_power_sqi(nn_intervals, freqs=None, pows=None, f_min=0.04, f_max=0.15):
    """Compute the logarithm power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    freqs : list
        The frequencies mapping with the power spectral (Default value = None)
    pows : list
        The powers of the relevant frequencies (Default value = None)
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max : float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    -------
    float
        The logarithmic power of the examined band

    Notes
    ---------
    If freqs and pows are assigned by computing the PSD before using this sqi,
    freqs & pows is used directly instead of nn_intervals.
    Otherwise, the frequencies and powers will be computed from nn intervals
    using welch method as default
    """
    if freqs is None or pows is None:
        freqs, pows = calculate_psd(nn_intervals)
    assert len(freqs) == len(pows), \
        "Length of the frequencies and the relevant powers must be the same"
    filtered_pows = pows[np.where((f_min <= freqs) & (freqs < f_max))[0]]
    log_pow = np.sum(np.log(filtered_pows))
    return log_pow


def relative_power_sqi(nn_intervals, freqs=None, pows=None, f_min=0.04, f_max=0.15):
    """Compute the relative power with respect to the total power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    freqs : list
        The frequencies mapping with the power spectral (Default value = None)
    pows : list
        The powers of the relevant frequencies (Default value = None)
    f_min : float
        The lower bound of the band .
        Default value = 0.04 is the lower bound of heart rate low-band
    f_max : float
        The upper bound of the band
        Default value = 0.15 is the upper bound of heart rate low-band

    Returns
    -------
    float
        Relative power with respect to the total power

    Notes
    ---------
    If freqs and pows are assigned by computing the PSD before using this sqi,
    freqs & pows is used directly instead of nn_intervals.
    Otherwise, the frequencies and powers will be computed from nn intervals
    using welch method as default
    """
    if freqs is None or pows is None:
        freqs, pows = calculate_psd(nn_intervals)
    assert len(freqs) == len(pows), \
        "Length of the frequencies and the relevant powers must be the same"
    filtered_pows = pows[np.where((f_min <= freqs) & (freqs < f_max))[0]]
    relative_pow = np.sum(np.log(filtered_pows))/np.sum(pows)
    return relative_pow


def normalized_power_sqi(nn_intervals, freqs=None, pows=None,
                         lf_min=0.04, lf_max=0.15, hf_min=0.15, hf_max=0.4):
    """Compute the relative power with respect to the total power of the examined band.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    freqs : list
        The frequencies mapping with the power spectral (Default value = None)
    pows : list
        The powers of the relevant frequencies (Default value = None)
    lf_min : float
        the lower bound of the low-frequency band (Default value = 0.04)
    lf_max : float
        the upper bound of the low-frequency band (Default value = 0.15)
    hf_min : float
        the lower bound of the high-frequency band (Default value = 0.15)
    hf_max : float
        the upper bound of the high-frequency band (Default value = 0.4)

    Returns
    -------
    float
        Relative power with respect to the total power

    Notes
    ---------
    If freqs and pows are assigned by computing the PSD before using this sqi,
    freqs & pows is used directly instead of nn_intervals.
    Otherwise, the frequencies and powers will be computed from nn intervals
    using welch method as default
    """
    if freqs is None or pows is None:
        freqs, pows = calculate_psd(nn_intervals)
    assert len(freqs) == len(pows), \
        "Length of the frequencies and the relevant powers must be the same"
    # lf_filtered_pows = pows[freqs >= lf_min & freqs < lf_max]
    lf_filtered_pows = pows[np.where((freqs >= lf_min) & (freqs < lf_max))[0]]
    # hf_filtered_pows = pows[freqs >= hf_min & freqs < hf_max]
    hf_filtered_pows = pows[np.where((freqs >= hf_min) & (freqs < hf_max))[0]]
    lf_power = np.sum(lf_filtered_pows)
    hf_power = np.sum(hf_filtered_pows)
    return np.linalg.norm([lf_power, hf_power])


def lf_hf_ratio_sqi(nn_intervals, freqs=None, pows=None,
                    lf_min=0.04, lf_max=0.15, hf_min=0.15, hf_max=0.4):
    """Compute the ratio power between the lower frequency and the high frequency.
    The function mimics features obtaining from the frequency domain of HRV.
    Main inputs are frequencies and power density - compute by using
    power spectral density power_spectrum in common package
    See. calculate_psd, calculate_spectrogram, calculate_power_wavelet

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval
    freqs : list
        The frequencies mapping with the power spectral (Default value = None)
    pows : list
        The powers of the relevant frequencies (Default value = None)
    lf_min : float
        the lower bound of the low-frequency band (Default value = 0.04)
    lf_max : float
        the upper bound of the low-frequency band (Default value = 0.15)
    hf_min : float
        the lower bound of the high-frequency band (Default value = 0.15)
    hf_max : float
        the upper bound of the high-frequency band (Default value = 0.4)

    Returns
    -------
    float
        Ratio power between low-frequency power and high-frequency power

    Notes
    ---------
    If freqs and pows are assigned by computing the PSD before using this sqi,
    freqs & pows is used directly instead of nn_intervals.
    Otherwise, the frequencies and powers will be computed from nn intervals
    using welch method as default
    """
    if freqs is None or pows is None:
        freqs, pows = calculate_psd(nn_intervals)
    assert len(freqs) == len(pows), \
        "Length of the frequencies and the relevant powers must be the same"
    lf_filtered_pows = pows[np.where((freqs >= lf_min) & (freqs < lf_max))[0]]
    hf_filtered_pows = pows[np.where((freqs >= hf_min) & (freqs < hf_max))[0]]
    ratio = np.sum(lf_filtered_pows)/np.sum(hf_filtered_pows)
    return ratio

def poincare_features_sqi(nn_intervals):
    """Function returning the poincare features of mapping nn intervals

    Parameters
    ----------
    nn_intervals : list
        Normal to Normal Interval

    Returns
    -------
    sd1 : float
        The standard deviation of the second group
    sd2 : float
        The standard deviation of the second group
    area : float
        The area of the bounding eclipse
    ratio : float
        

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

    poincare_features_dict = {
        "poincare_features_sd1_sqi":sd1,
        "poincare_features_sd2_sqi": sd2,
        "poincare_features_area_sqi": area,
        "poincare_features_ratio_sqi": ratio,
    }

    return poincare_features_dict


def get_all_features_hrva(s, sample_rate=100, rpeak_method=0,wave_type='ecg'):
    """

    Parameters
    ----------
    data_sample :
        Raw signal
    rpeak_method :
        return: (Default value = 0)
    sample_rate :
        (Default value = 100)

    Returns
    -------


    """

    # if rpeak_method in [1, 2, 3, 4]:
    #     detector = PeakDetector()
    #     peak_list = detector.ppg_detector(data_sample, rpeak_method)[0]
    # else:
    #     rol_mean = rolling_mean(data_sample, windowsize=0.75, sample_rate=100.0)
    #     peaks_wd = detect_peaks(data_sample, rol_mean, ma_perc=20,
    #                             sample_rate=100.0)
    #     peak_list = peaks_wd["peaklist"]
    if wave_type =='ppg':
        detector = PeakDetector(wave_type='ppg')
        peak_list, trough_list = detector.ppg_detector(s, detector_type=rpeak_method)
    else:
        detector = PeakDetector(wave_type='ecg')
        peak_list, trough_list = detector.ecg_detector(s, detector_type=rpeak_method)

    if len(peak_list) < 2:
        warnings.warn("Peak Detector cannot find more than 2 peaks to process")
        return [],[],[],[]

    rr_list = np.diff(peak_list) * (1000 / sample_rate)  # 1000 milisecond

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    nn_list = get_nn_intervals(rr_list)
    sys.stdout = old_stdout

    nn_list_non_na = np.copy(nn_list)
    nn_list_non_na[np.where(np.isnan(nn_list_non_na))[0]] = -1

    time_domain_features = get_time_domain_features(rr_list)
    frequency_domain_features = get_frequency_domain_features(rr_list)
    geometrical_features = get_geometrical_features(rr_list)
    csi_cvi_features = get_csi_cvi_features(rr_list)

    return time_domain_features, frequency_domain_features, geometrical_features, csi_cvi_features

