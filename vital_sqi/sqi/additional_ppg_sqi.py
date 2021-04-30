import numpy as np
import sys
from scipy import signal
from scipy.stats import kurtosis,entropy,skew
from numpy import NaN, Inf, arange, isscalar, asarray, array
def billauer_peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    billauer_peakdet Detect peaks in a vector
            [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
            maxima and minima ("peaks") in the vector V.
            MAXTAB and MINTAB consists of two columns. Column 1
            contains indices in V, and column 2 the found values.
          
            With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
            in MAXTAB and MINTAB are replaced with the corresponding
            X-values.
    
            A point is considered a maximum peak if it has the maximal
            value, and was preceded (to the left) by a value lower by
            DELTA.
    
     Eli Billauer, 3.4.05 (Explicitly not copyrighted).
     This function is released to the public domain; Any use is allowed.

    Parameters
    ----------
    v :
        Vector of input signal to detect peaks
    delta : 
        Parameter for determining peaks and valleys. A point is considered a maximum peak if 
        it has the maximal value, and was preceded (to the left) by a value lower by delta.
    x :
        (Optional) Replace the indices of the resulting max and min vectors with corresponding x-values

    Returns
    -------
    max : array
        Array containing the maxima points (peaks)
    min : array
        Array containing the minima points (valleys)
   
    """
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]   
        if lookformax:
            if this < mx-delta:
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab) , array(mintab)

def scipy_find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None):
    """
    SciPy implementation of a peak detector. Used in MSQ SQI calculation to determine difference 
    between two types of peak detectors (SciPy and Billauer by default)

    Parameters
    ----------
    s : sequence
        A signal with peaks.

    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, None, an array matching x or a 2-element 
        sequence of the former. The first element is always interpreted as the minimal and the second, 
        if supplied, as the maximal required height.

    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighboring samples. 
        Either a number, None, an array matching x or a 2-element sequence of the former. 
        The first element is always interpreted as the minimal and the second, if supplied, 
        as the maximal required threshold.

    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. 
        Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.

    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, None, an array matching x or a 2-element 
        sequence of the former. The first element is always interpreted as the minimal and the second, 
        if supplied, as the maximal required prominence.

    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, None, an array matching x or a 2-element
         sequence of the former. The first element is always interpreted as the minimal and the second, 
         if supplied, as the maximal required width.

    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if one of the arguments 
        prominence or width is given. See argument wlen in peak_prominences for a full description of its effects.

    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if width is given. See argument 
        rel_height in peak_widths for a full description of its effects.

    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number, None, an array matching x or a 2-element
         sequence of the former. The first element is always interpreted as the minimal and the second, 
         if supplied as the maximal required plateau size.

    Returns
    -------
    peaks : ndarray
        Indices of peaks in x that satisfy all given conditions.

    properties : dict
        A dictionary containing properties of the returned peaks which were calculated as 
        intermediate results during evaluation of the specified conditions.

    """
    return signal.find_peaks(x, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)

def msq_sqi(x, b_delta, s_height, s_thresh, s_dist):
    """
    MSQ SQI as defined in Elgendi et al "Optimal Signal Quality Index for Photoplethysmogram Signals" 
    with modification of the second algorithm used. Instead of Bing's, a SciPy built-in implementation is used. 
    The SQI tracks the agreement between two peak detectors to evaluate quality of the signal.

    Parameters
    ----------
    x : sequence
        A signal with peaks.

    b_delta : float  
        Delta used for Billauer's peak detection algorithm

    s_height : number or ndarray or sequence, optional
        Height parameter for SciPy peak detection algorithm

    s_thresh : number or ndarray or sequence, optional
        Threshold parameter for SciPy peak detection algorithm

    s_dist : number, optional
        Distance parameter for SciPy peak detection algorithm

    Returns
    -------
    msq_sqi : number
        MSQ SQI value for the given signal

    """
    peaks_1,_ = billauer_peakdet(x, delta=b_delta)
    peaks_2,_ = scipy_find_peaks(x, height=s_height, threshold=s_thresh, distance=s_dist)
    if len(peaks_1)==0:
        return 0.0
    return len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)

def acf(sig):
    """
    Calculate autocorrelation array of a signal

    Parameters
    ----------
    sig : sequence
        A signal for autocorrelation calculation.

    Returns
    -------
    autocorr : array
        Array containing the autocorrelation of provided signal

    """
    return np.array([1]+[np.corrcoef(sig[:-i], sig[i:])[0,1] for i in range(1,len(sig))])

def acf_peaks_locs_sqi(sig):
    """
    Calculate autocorrelation SQI's as defined in Elgendi et al. 4 SQI's are calculated as part of this function,
    corresponding to the correlation values of the first two peaks of the correlogram as well as the time 
    lags associated with these two peaks

    Parameters
    ----------
    sig : sequence
        A signal to calculate acf SQI's from.

    Returns
    -------
    acf_sqi : touple of 4 numbers
        A touple containing 4 numbers, representing the value and location of first and second peak of the correlogram. 
        If it is not possible to calculate some or all ACF SQI's for any reason, (eg. the correlogram does not have enough peaks) 
        a 0 is returned in place of the SQI.

    """
    corrs=acf(sig)
    peaks=scipy_find_peaks(corrs)
    if len(peaks[0])!=0:
        if len(peaks[0])>=2:
            return(peaks[0][0], corrs[peaks[0][0]], peaks[0][1], corrs[peaks[0][1]])
        else:
            return(peaks[0][0], corrs[peaks[0][0]], 0, 0)
    else:
        return(0, 0, 0, 0)

#TODO: Add functions to calculate metrics (mean, median, std) for Skewness, Kurtosis, Entropy, SNR, Relative power on beat-by-beat basis for given signal snippet