"""
Class containing signal, header and sqi
"""


class SignalSQI:
    """ """
    def __init__(self, wave_type=None, signals=None, sampling_rate=None,
                 start_datetime=None, sqi_indexes=None, info=None):
        self.signals = signals
        self.sampling_rate = sampling_rate
        self.start_datetime = start_datetime
        self.wave_type = wave_type
        self.sqi_indexes = sqi_indexes
        self.info = info

    def update_info(self, info):
        """

        Parameters
        ----------
        info :
            

        Returns
        -------
        object of SignalSQI class
        
        """
        self.info = info
        return self

    def update_signal(self, signals):
        """

        Parameters
        ----------
        signals : numpy.ndarray of shape (m, n)
        m is the number of rows and n is the number of channels of the signal.
            

        Returns
        -------
        object of class SignalSQI
        
        """
        self.signals = signals
        return self

    def update_sqi_indexes(self, sqi_indexes):
        """

        Parameters
        ----------
        sqi_indexes : numpy.ndarray of shape (m, n)
        m is the number of signal segments, n is the number of SQIs.
            

        Returns
        -------
        object of class SignalSQI
        
        """
        self.sqi_indexes = sqi_indexes
        return self

    def update_sampling_rate(self, sampling_rate):
        """

        Parameters
        ----------
        sampling_rate : float
        Note: sampling_rate must be correct to reliably infer RR intervals,
        etc.

            

        Returns
        -------
        object of class SignalSQI
        """
        self.sampling_rate = sampling_rate
        return self

    def update_start_datetime(self, start_datetime):
        """

        Parameters
        ----------
        start_datetime : datetime
        start date and

        Returns
        -------
        object of si
        """
        self.start_datetime = start_datetime
        return self
