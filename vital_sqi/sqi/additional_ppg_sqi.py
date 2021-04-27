import numpy as np
import pandas as pd 
import sys
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from scipy.stats import kurtosis,entropy,skew
from numpy import NaN, Inf, arange, isscalar, asarray, array
def billauer_peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
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

def scipy_find_peaks(sig,height=None,threshold=None,distance=None):
    return signal.find_peaks(sig,height=height,threshold=threshold,distance=distance)

def cross_zero(detrended_signal):
    zero_crossings = len(np.where(np.diff(np.sign(detrended_signal)))[0])/len(detrended_signal)
    return zero_crossings

def noise_ratio(filtered_signal):
    return np.var(filtered_signal)/np.var(abs(filtered_signal))

def relative_power(filtered_signal):
    f,PSD = signal.welch(filtered_signal,125,nperseg=len(filtered_signal))
    indices1 = [i for i in range(len(f)) if f[i]>=1.0 and f[i]<=2.4] # indices of PSD's 1 Hz to 2.25 Hz
    indices2 = [i for i in range(len(f)) if f[i]>=0 and f[i]<=8] #indices of PSDS from 0 Hz to 8 Hz
    return (PSD[indices1].sum()/PSD[indices2].sum())

def get_msq(peaks_1, peaks_2):
    if len(peaks_1)==0:
        return 0.0
    return len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

def calculate_spo2(Red_sig, IR_sig, filtR, filtIR):    
    dc_r = np.median(Red_sig)
    dc_ir = np.median(IR_sig)
    #get the root mean square of the AC portion of the signals
    rms_r = np.sqrt(np.mean(filtR**2))
    rms_ir = np.sqrt(np.mean(filtIR**2))
    #calculate the absorption ratio
    r = (rms_r/dc_r)/(rms_ir/dc_ir)
    #using the ratio calculate the spo2 level
    #spo2 = 114.515 - 37.313*abs(r)
    spo2 = 110 - 25*abs(r) #standard forumla for spo2 level is 110 - 25*((ACrms of Red/DC of red)/(ACrms of IR/DC of IR))
    return spo2

def skewness_features(filtered_signal, mins):
    skewness_total = skew(filtered_signal)
    n_segments = len(mins)-1
    skew_scores=[]
    for i in range(n_segments):
        sig = filtered_signal[mins[i]:mins[i+1]]
        skew_scores.append(skew(sig))
    return (skewness_total, np.mean(skew_scores), np.median(skew_scores), np.std(skew_scores))

def kurtosis_features(filtered_signal, mins):
    kurtosis_total = kurtosis(filtered_signal)
    n_segments = len(mins)-1
    kurtosis_scores=[]
    for i in range(n_segments):
        sig = filtered_signal[mins[i]:mins[i+1]]
        kurtosis_scores.append(kurtosis(sig))
    return (kurtosis_total, np.mean(kurtosis_scores), np.median(kurtosis_scores), np.std(kurtosis_scores))

def entropy_features(filtered_signal, mins, qk=None, base=None, axis=0):
    entropy_total = entropy(filtered_signal, qk, base, axis)
    n_segments = len(mins)-1
    entropy_scores=[]
    for i in range(n_segments):
        sig = filtered_signal[mins[i]:mins[i+1]]
        entropy_scores.append(entropy(sig, qk, base, axis))
    return (entropy_total, np.mean(entropy_scores), np.median(entropy_scores), np.std(entropy_scores))

def snr_features(filtered_signal, mins):
    snr_total = noise_ratio(filtered_signal)
    n_segments = len(mins)-1
    snr_scores=[]
    for i in range(n_segments):
        sig = filtered_signal[mins[i]:mins[i+1]]
        snr_scores.append(noise_ratio(sig))
    return (snr_total, np.mean(snr_scores), np.median(snr_scores), np.std(snr_scores))

def relative_power_features(filtered_signal, mins):
    relative_power_total = relative_power(filtered_signal)
    n_segments = len(mins)-1
    relative_powers=[]
    for i in range(n_segments):
        sig = filtered_signal[mins[i]:mins[i+1]]
        relative_powers.append(relative_power(sig))
    return (relative_power_total, np.mean(relative_powers), np.median(relative_powers), np.std(relative_powers))

def acf(sig):
    return np.array([1]+[np.corrcoef(sig[:-i], sig[i:])[0,1] for i in range(1,len(sig))])

def acf_peaks_locs_features(filtered_signal):
    corrs=acf(filtered_signal)
    peaks=scipy_find_peaks(corrs)
    if len(peaks[0])!=0:
        if len(peaks[0])>=2:
            return(peaks[0][0], corrs[peaks[0][0]], peaks[0][1], corrs[peaks[0][1]])
        else:
            return(peaks[0][0], corrs[peaks[0][0]], 0, 0)
    else:
        return(0, 0, 0, 0)