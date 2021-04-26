import numpy as np
from scipy import signal
from scipy import interpolate

def calculate_power(freq,pow,fmin,fmax):
    """
    compute the power within the band range
    Parameters
    ----------
    freq
    pow
    fmin
    fmax

    Returns
    -------

    """

    # case heatmap spectrogram -> compute the total power with time series -or the mean power
    if pow.ndim == 2:
        pow = np.mean(pow,axis=1)

    band = pow[(freq >= fmin and freq < fmax)]
    band_power = np.sum(band)/(2*np.power(len(pow),2))

    return band_power

def calculate_Spectrum(rr_intervals, method='welch',
                           sampling_frequency=4,
                           interpolation_method="linear",
                           power_type='density',
                           ):
    """
    Returns the frequency and power of the signal.

    Parameters
    ---------
    rr_intervals : array-like
        list of RR interval (in ms)
    method : str
        Method used to calculate the psd or powerband or spectrogram.
        available methods are:
        'welch': apply welch method to compute PSD
        'lomb': apply lomb method to compute PSD
        'ar': method to compute the periodogram - if compute PSD then power_type = 'density'
        'spectrogram': method to compute the spectrogram, output an extra list represent timestamp
        'powerband': compute the power density at 4 power band.
            The min-max boundary of power need to be defined

    sampling_frequency : int
        Frequency of the spectrum need to be observed. Common value range from 1 Hz to 10 Hz,
        by default set to 4 Hz. Detail can be found from the ECG book
    interpolation_method : str
        Kind of interpolation as a string, by default "linear". Applicable for welch
    power_type: str
        'density':
        'spectrogram':

    Returns
    ---------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """

    # create timestamp to do the interpolation
    ts_rr = [np.sum(rr_intervals[:i]) / 1000 for i in range(len(rr_intervals))]
    # convert each RR interval (unit ms) to bpm - representative
    bpm_list = (1000 * 60) / rr_intervals

    first = ts_rr[0]
    last = ts_rr[-1]
    # interpolate the data

    if method == 'welch':
        # ---------- Interpolation of signal ---------- #
        # funct = interpolate.interp1d(x=timestamp_list, y=nn_intervals, kind=interpolation_method)
        interpolator = interpolate.interp1d(ts_rr, bpm_list, kind=interpolation_method)
        # create timestamp for the interpolate rr
        time_offset = 1 / sampling_frequency
        # ts_interpolate = [np.sum(RR[:i]) / 100 for i in range(len(IRR))]
        ts_interpolate = np.arange(0, last - first, time_offset)

        # timestamps_interpolation = _create_interpolated_timestamp_list(nn_intervals, sampling_frequency)
        nni_interpolation = interpolator(ts_interpolate)

        # ---------- Remove DC Component ---------- #
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)

        #  --------- Compute Power Spectral Density  --------- #
        freq, psd = signal.welch(x=nni_normalized, fs=sampling_frequency, window='hann',
                                 nfft=4096)

    elif method == 'lomb':
        freq = np.linspace(1e-5, sampling_frequency, 100)
        psd = signal.lombscargle(ts_rr, bpm_list, freq)

    elif method == 'ar':
        freq, psd = signal.periodogram(bpm_list, sampling_frequency, window='boxcar',
                                       nfft=None, detrend='constant',
                                       return_onesided=True,
                                       scaling=power_type, axis=- 1)

    elif method == 'spectrogram':
        freq, t, psd = signal.spectrogram(bpm_list, sampling_frequency)
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    return freq,psd