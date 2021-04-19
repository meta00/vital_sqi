from pyedflib import highlevel
from wfdb import rdsamp, wrsamp
import numpy as np
import pandas as pd
from vital_sqi.common import utils


class SignalSQI:
    """ """
    def __init__(self, wave_type=None, signals=None, sampling_rate=None,
                 sqi_indexes=None, info=None):
        self.signals = signals
        self.sampling_rate = sampling_rate
        self.wave_type = wave_type
        self.sqi_indexes = sqi_indexes
        self.info = info

def ECG_reader(file_name,
               file_type=None,
               channel_num=None,
               channel_name=None):
    """

    Parameters
    ----------
    file_name :
        
    file_type :
        (Default value = None)
    channel_num :
        (Default value = None)
    channel_name :
        (Default value = None)

    Returns
    -------

    
    """
    if file_type == 'edf':
        signals, signal_headers, header = highlevel.read_edf(
                edf_file=file_name,
                ch_nrs=channel_num,
                ch_names=channel_name)
        sampling_rate = signal_headers[0]['sample_rate']
        signals = signals.transpose()
        out = SignalSQI(signals=signals, wave_type='ecg',
                        sampling_rate=sampling_rate)
        info = [header, signal_headers]
    elif file_type == 'mit':
        signals, info = rdsamp(record_name=file_name,
                               channels=channel_num,
                               channel_names=channel_name,
                               warn_empty=True,
                               return_res=64)
        sampling_rate = info['fs']
        out = SignalSQI(signals=signals, wave_type='ecg',
                        sampling_rate=sampling_rate)
    elif file_type == 'csv':
        if channel_name is None and channel_num is None:
            raise Exception("Missing channel info")
        else:
            if channel_name is None:
                use_cols = channel_num
            else:
                use_cols = channel_name
        out = pd.read_csv(file_name,
                          usecols=use_cols,
                          skipinitialspace=True)
        sampling_rate = utils.calculate_sampling_rate(out[0])
        signals = out.drop(0)
        info = {'sampling_rate': sampling_rate}
        out = SignalSQI(signals=signals, wave_type='ecg',
                        sampling_rate=sampling_rate)
    else:
        raise ValueError(f"File type {0} not supported".format(file_type))

    return out, info


def ECG_writer(signal_sqi, file_name, file_type, info=None):
    """

    Parameters
    ----------
    signal_sqi : SignalSQI object containing signals, sampling rate and sqi info.
        
    file_name : name of file to write, with extension. For edf file_type,
    possible extensions are edf, edf+, bdf, bdf+. For mit file_type,
    possible extensions are... :
        
    file_type : edf or mit or csv

    info : list
        In case of writing edf file: A list containing signal_headers and
        header (in
        order). signal_headers is a list of dict with one signal header for
        each signal channel. header (dict) contain ecg file information.
        In case of writing wfdb record (mit file): A dict containing header
        as defined in .hea file.
        (Default value = None)

    Returns
    -------
    bool
    """
    signals = signal_sqi.signals
    sampling_rate = signal_sqi.sampling_rate
    if file_type == 'edf':
        if info is not None:
            signal_headers = info[0]
            header = info[1]
            highlevel.write_edf(file_name, signals, signal_headers,
                                header, file_type=-1, digital=False,
                                block_size=-1)
        else:
            highlevel.write_edf_quick(file_name, signals, sampling_rate)
    if file_type == 'mit':
        if info is None:
            raise Exception("Header dict needed")
        else:
            wrsamp(record_name=file_name,
                   fs=sampling_rate,
                   units=info['units'],
                   sig_name = info['sig_name'],
                   p_signal=signals,
                   base_date=info['base_date'],
                   base_time=info['base_time'],
                   comments=info['comments'])
    if file_type == 'csv':
        np.savetxt(file_name, signals)
    return True
def PPG_reader(file_name):
    np.loadtxt(file_name, skirows=1)
    # time, pleth,