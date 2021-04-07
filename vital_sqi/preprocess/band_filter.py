""" Filtering of raw signals by bandpass"""
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy import signal

class BandpassFilter:
    def __init__(self,band_type="butter",fs=100):
        """

        :param band_type: type of bandpass.
            "butter": butterworth
            "cheby1": chebyshev-1
            "cheby2": chebyshev-2
            "ellip" : Elliptic (Cauer) digital and analog filter design
            "bessel": Bessel/Thomson digital and analog filter design.
        :param fs: sampling frequency

        """
        self.band_type = band_type
        self.fs = fs

    def signal_bypass(self,cutoff,order,a_pass,rp,rs,btype='high'):
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        if self.band_type == 'cheby1':
            b, a = signal.cheby1(order, a_pass, normal_cutoff, btype=btype, analog=False)
        elif self.band_type == 'cheby2':
            b, a = signal.cheby2(order, a_pass, normal_cutoff, btype=btype, analog=False)
        elif self.band_type == 'ellip':
            b, a = signal.ellip(order, rp, rs, normal_cutoff, btype=btype, analog=False)
        elif self.band_type == 'bessel':
            b, a = signal.bessel(order, normal_cutoff, btype=btype, analog=False)
        else:
            b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
        return b, a

    def signal_lowpass_filter(self,data,cutoff,order=3,a_pass=3,rp=4,rs=40):
        """
            EXPOSE
            Low pass filter as described in scipy package
            :param data: list, array of input signal
            :param cutoff:
            :param order:
            :param a_pass:
            :param rp: The maximum ripple allowed below unity gain in the passband.
                    Specified in decibels, as a positive number.
            :param rs: The minimum attenuation required in the stop band.
                    Specified in decibels, as a positive number
            :return:
            """
        b, a = self.signal_bypass(cutoff, order, a_pass, rp, rs,btype='low')
        y = lfilter(b, a, data)
        return y

    def signal_highpass_filter(self,data, cutoff, order=5, a_pass=3,rp=4,rs=40):
        """
            High pass filter as described in scipy package
            :param data: list, array of input signal
            :param cutoff:
            :param fs:
            :param order:
            :return:
            """
        b, a = self.signal_bypass(cutoff, order,a_pass,rp,rs,btype='high')
        y = signal.filtfilt(b, a, data)
        return y

