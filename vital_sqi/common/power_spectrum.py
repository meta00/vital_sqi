import numpy as np
from scipy import signal
from scipy import interpolate
import pycwt as wavelet
from statsmodels.tsa.ar_model import AutoReg

mother_wave_dict = {
    'gaussian': wavelet.DOG(),
    'paul': wavelet.Paul(),
    'mexican_hat': wavelet.MexicanHat()
}


def calculate_power(freq, pow, fmin, fmax):
    """
    Compute the power within the band range

    Parameters
    ----------
    freq: array-like
        list of all frequencies need to be computed
    pow: array-like
        the power of relevant frequencies
    fmin: float
        lower bound of the selected band
    fmax: float
        upper bound of the selected band

    Returns
    -------
        :float
        The absolute power of the selected band
    """

    # case heatmap spectrogram -> compute the total power with time series
    # or the mean power
    if pow.ndim == 2:
        pow = np.mean(pow, axis=1)

    band = pow[(freq >= fmin and freq < fmax)]
    band_power = np.sum(band)/(2*np.power(len(pow), 2))

    return band_power


def get_interpolated_data(ts_rr, bpm_list, sampling_frequency,
                          interpolation_method="linear"):
    """
    Method to interpolate the outlier hr data

    Parameters
    ----------
    ts_rr: array-like
        list of timestamp indicates the appearance of r peaks (in ms)
    bpm_list: array-like
        the heart rate list indicates the HRV index in beat-per-minute unit
    sampling_frequency:
        examining frequency of heart rate to create the offset for timestamp
    interpolation_method : str
        Kind of interpolation as a string, by default "linear".
        Applicable for welch

    Returns
    -------
        :numpy-array
        The interpolated hr in bpm unit
    """
    first = ts_rr[0]
    last = ts_rr[-1]
    # interpolate the data
    # ---------- Interpolation of signal ---------- #
    interpolator = interpolate.interp1d(ts_rr, bpm_list,
                                        kind=interpolation_method)
    # create timestamp for the interpolate rr
    time_offset = 1 / sampling_frequency
    ts_interpolate = np.arange(0, last - first, time_offset)

    nni_interpolation = interpolator(ts_interpolate)
    return nni_interpolation


def get_time_and_bpm(rr_intervals):
    """
    Method to generate timestamps from frequencies
    and convert the rr intervals to hr unit

    Parameters
    ----------
    rr_intervals: array-like
        list of RR interval (in ms)

    Returns
    -------
    ts_rr : list
        the generated time for each heart beat (in s).
    bpm_list : list
        the beat per minute index of to the peak
    """
    # create timestamp to do the interpolation
    ts_rr = [np.sum(rr_intervals[:i]) / 1000 for i in range(len(rr_intervals))]
    # convert each RR interval (unit ms) to bpm - representative
    bpm_list = (1000 * 60) / rr_intervals
    return ts_rr, bpm_list


def calculate_psd(rr_intervals, method='welch',
                  hr_sampling_frequency=4,
                  power_type='density',
                  max_lag=3):
    """
    Returns the frequency and spectral power from the rr intervals.
    This method is used to compute HRV frequency domain features

    Parameters
    ---------
    rr_intervals : array-like
        list of RR interval (in ms)
    method : str
        Method used to calculate the psd or powerband or spectrogram.
        available methods are:
        'welch': apply welch method to compute PSD
        'lomb': apply lomb method to compute PSD
        'ar': method to compute the periodogram - if compute PSD then
        power_type = 'density'

    hr_sampling_frequency : int
        Frequency of the spectrum need to be observed. Common value range
        from 1 Hz to 10 Hz,
        by default set to 4 Hz. Detail can be found from the ECG book

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
    ts_rr, bpm_list = get_time_and_bpm(rr_intervals)

    if method == 'welch':
        nni_interpolation = get_interpolated_data(ts_rr,
                                                  bpm_list,
                                                  hr_sampling_frequency)
        # ---------- Remove DC Component ---------- #
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)

        #  --------- Compute Power Spectral Density  --------- #
        freq, psd = signal.welch(x=nni_normalized,
                                 fs=hr_sampling_frequency,
                                 window='hann',
                                 nfft=4096)

    elif method == 'lomb':
        freq = np.linspace(0, hr_sampling_frequency, 2**8)
        a_frequencies = np.asarray(2 * np.pi / freq)
        psd = signal.lombscargle(ts_rr, rr_intervals, a_frequencies,
                                 normalize=True)

    elif method == 'ar':
        freq, psd_ = signal.periodogram(rr_intervals, hr_sampling_frequency,
                                        window='boxcar', nfft=None,
                                        detrend='constant',
                                        return_onesided=True,
                                        scaling=power_type, axis=- 1)
        model = AutoReg(psd_, max_lag)
        res = model.fit()
        psd = model.predict(res.params)
    else:
        raise ValueError("Not a valid method. Choose between 'ar', 'lomb' "
                         "and 'welch'")

    return freq, psd


def calculate_spectrogram(rr_intervals, hr_sampling_frequency=4):
    """
    Method to compute the spectrogram, output an extra list represent timestamp

    Parameters
    ----------
    rr_intervals: array-like
        list of RR interval (in ms)
    hr_sampling_frequency: int
        values = The range of heart rate frequency * 2
    Returns
    -------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    t: list
        Time points of the corresponding psd
    """
    ts_rr, bpm_list = get_time_and_bpm(rr_intervals)
    freq, t, psd = signal.spectrogram(bpm_list, hr_sampling_frequency)
    return freq, psd, t


def calculate_power_wavelet(rr_intervals, heart_rate=4,
                            mother_wave='morlet'):
    """
    Method to calculate the spectral power using wavelet method.

    Parameters
    ----------
    rr_intervals: array-like
        list of RR interval (in ms)
    heart_rate: int
        values = The range of heart rate frequency * 2
    mother_wave: str
        The main waveform to transform data.
        Available waves are:
        'gaussian':
        'paul': apply lomb method to compute PSD
        'mexican_hat':

    Returns
    -------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """
    dt = 1 / heart_rate
    if mother_wave in mother_wave_dict.keys():
        mother_morlet = mother_wave_dict[mother_wave]
    else:
        mother_morlet = wavelet.Morlet()

    wave, scales, freqs, coi, fft, fftfreqs = \
        wavelet.cwt(rr_intervals, dt, wavelet=mother_morlet)
    powers = (np.abs(wave)) ** 2
    return freqs, powers
