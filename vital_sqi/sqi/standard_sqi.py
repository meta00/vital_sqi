"""Signal quality indexes based on dynamic template matching"""

import numpy as np
from scipy.stats import kurtosis, skew, entropy
from vital_sqi.common.rpeak_detection import PeakDetector
import vital_sqi.preprocess.preprocess_signal as sqi_pre

"""
Most of the sqi scores are obtained from the following paper Elgendi,
Mohamed, Optimal signal quality index for photoplethysmogram
signals, Bioengineering.
"""


def perfusion_sqi(x, y):
    """The perfusion index is the ratio of the pulsatile blood flow to the
    nonpulsatile or static blood in peripheral tissue.
    In other words, it is the difference of the amount of light absorbed
    through the pulse of when light is transmitted through the finger,
    which can be defined as follows:
    PSQI=[(ymax−ymin)/x¯|]×100 where PSQI is the perfusion index, x¯ is the
    statistical mean of the x signal (raw PPG signal), and y is the filtered
    PPG signal

    Parameters
    ----------
    x :
        float, mean of the raw signal
    y :
        list, array of filtered signal

    Returns
    -------

    """
    return (np.max(y)-np.min(y))/np.abs(np.mean(x))*100


def kurtosis_sqi(x, axis=0, fisher=True, bias=True,
                 nan_policy='propagate'):
    """Expose
    Kurtosis is a measure of whether the data are heavy-tailed or
    light-tailed relative to a normal distribution. That is, data sets with
    high kurtosis tend to have heavy tails, or outliers.
    Data sets with low kurtosis tend to have light tails, or lack of outliers.
    A uniform distribution would be the extreme case.

    Kurtosis is a statistical measure used to describe the distribution of
    observed data around the mean. It represents a heavy tail and peakedness
    or a light tail and flatness of a distribution relative to the normal
    distribution, which is defined as:

    Parameters
    ----------
    x :
        list, the array of signal
    axis :
         (Default value = 0)
    fisher :
         (Default value = True)
    bias :
         (Default value = True)
    nan_policy :
         (Default value = 'propagate')

    Returns
    -------

    """

    return kurtosis(x, axis, fisher, bias, nan_policy)


def skewness_sqi(x, axis=0, bias=True, nan_policy='propagate'):
    """Expose
    Skewness is a measure of symmetry, or more precisely, the lack of
    symmetry. A distribution, or data set, is symmetric if it looks the same
    to the left and right of the center point.

    Skewness is a measure of the symmetry (or the lack of it) of a
    probability distribution, which is defined as:
    SSQI=1/N∑i=1N[xi−μˆx/σ]3
    where μˆx and σ are the empirical estimate of the mean and standard
    deviation of xi,respectively; and N is the number of samples in the PPG
    signal.

    Parameters
    ----------
    x :
        list, the array of signal
    axis :
         (Default value = 0)
    bias :
         (Default value = True)
    nan_policy :
         (Default value = 'propagate')

    Returns
    -------

    """
    return skew(x, axis, bias, nan_policy)


def entropy_sqi(x, qk=None, base=None, axis=0):
    """Expose
    Calculate the entropy information from the template distribution. Using
    scipy package function.

    Parameters
    ----------
    x :
        list the input signal
    qk :
        list, array against which the relative entropy
        is computed (Default value = None)
    base :
        float, (Default value = None)
    axis :
        return: (Default value = 0)

    Returns
    -------

    """
    x_ = x - min(x)
    return entropy(x_, qk, base, axis)


def signal_to_noise_sqi(a, axis=0, ddof=0):
    """Expose
    A measure used in science and engineering that compares the level of a
    desired signal to the level of background noise.

    Parameters
    ----------
    a :
        param axis:
    ddof :
        return: (Default value = 0)
    axis :
         (Default value = 0)

    Returns
    -------

    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def zero_crossings_rate_sqi(y, threshold=1e-10, ref_magnitude=None,
                            pad=True, zero_pos=True, axis=-1):
    """Reuse the function from librosa package.
    This is the rate of sign-changes in the processed signal, that is,
    the rate at which the signal changes from positive to negative or back.

    Parameters
    ----------
    y :
        list, array of signal
    threshold :
        float > 0, default=1e-10 if specified, values where
        -threshold <= y <= threshold are clipped to 0.
    ref_magnitude :
        float >0 If numeric, the threshold is scaled
        relative to ref_magnitude.
        If callable, the threshold is scaled relative
        to ref_magnitude(np.abs(y)). (Default value = None)
    pad :
        boolean, if True, then y[0] is considered a valid
        zero-crossing. (Default value = True)
    zero_pos :
        the crossing marker. (Default value = True)
    axis :
        axis along which to compute zero-crossings. (Default value = -1)

    Returns
    -------
    type
        float, indicator array of zero-crossings in `y` along the
        selected axis.

    """
    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    crossings = np.pad(
        (y_sign[tuple(slice_post)] != y_sign[tuple(slice_pre)]),
        padding,
        mode="constant",
        constant_values=pad,
    )

    return np.mean(crossings, axis=0, keepdims=True)[0]


def mean_crossing_rate_sqi(y, threshold=1e-10, ref_magnitude=None,
                           pad=True, zero_pos=True, axis=-1):
    """Expose
    Same as zero crossing rate but this function interests in the rate of
    crossing signal mean

    Parameters
    ----------
    y :
        param threshold:
    ref_magnitude :
        param pad: (Default value = None)
    zero_pos :
        param axis: (Default value = True)
    threshold :
         (Default value = 1e-10)
    pad :
         (Default value = True)
    axis :
         (Default value = -1)

    Returns
    -------

    """
    return zero_crossings_rate_sqi(y-np.mean(y), threshold, ref_magnitude,
                                   pad, zero_pos, axis)


def msq_sqi(y, peaks_1, peak_detect2=6):
    """
    MSQ SQI as defined in Elgendi et al
    "Optimal Signal Quality Index for Photoplethysmogram Signals"
    with modification of the second algorithm used.
    Instead of Bing's, a SciPy built-in implementation is used.
    The SQI tracks the agreement between two peak detectors
    to evaluate quality of the signal.

    Parameters
    ----------
    x : sequence
        A signal with peaks.

    peaks_1 : array of int  
        Already computed peaks arry from the primary peak_detector

    peak_detect2 : int
        Type of the second peak detection algorithm, default = Scipy

    Returns
    -------
    msq_sqi : number
        MSQ SQI value for the given signal

    """
    detector = PeakDetector(wave_type='ppg')
    peaks_2,_ = detector.ppg_detector(y, detector_type=peak_detect2, preprocess=False)
    if len(peaks_1)==0:
        return 0.0
    return len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)

def per_beat_sqi(sqi_func, troughs, signal, taper, **kwargs):
    """
    Perform a per-beat application of the selected SQI function on the signal segment

    Parameters
    ----------
    sqi_func : function 
        An SQI function to be performed.

    troughs : array of int  
        Idices of troughs in the signal provided by peak detector to be able to extract individual beats

    signal : 
        Signal array containing one segment of the waveform

    taper : bool
        Is each beat need to be tapered or not before executing the SQI function

    **kwargs : dict
        Additional positional arguments that needs to be fed into the SQI function

    Returns
    -------
    calculated_SQI : array
        An array with SQI values for each beat of the signal

    """
    #Remove first and last trough as they might be on the edge
    troughs = troughs[1:-1]
    if len(troughs) > 2:
        sqi_vals = []
        for idx, beat_start in enumerate(troughs[:-1]):
            single_beat = signal[beat_start:troughs[idx+1]]
            if taper:
                single_beat = sqi_pre.tapering(single_beat)
            if len(kwargs) != 0:
                args = tuple(kwargs.values()) 
                sqi_vals.append(sqi_func(single_beat, *args))
            else:
                sqi_vals.append(sqi_func(single_beat))
        return sqi_vals

    else:
        raise Exception("Not enough peaks in the signal to generate per beat SQI")
