from pyedflib import highlevel
from wfdb import rdsamp, wrsamp
import numpy as np
import pandas as pd
import datetime as dt
import os
import glob
# from vital_sqi.common import generate_timestamp, utils
from vital_sqi.common import utils
from vital_sqi.common.utils import generate_timestamp
from vital_sqi.data.signal_sqi_class import SignalSQI


def ECG_reader(file_name, file_type=None, channel_num=None,
               channel_name=None, sampling_rate=None,
               start_datetime=None):
    """

    Parameters
    ----------
    file_name : str

    file_type :
        (Default value = None)
    channel_num : frm 0
        (Default value = None)
    channel_name :
        (Default value = None)
    sampling_rate :
        (Default value = None)
    start_datetime : optional
        (Default value = None)

    Returns
    -------
    object of SignalSQI class

    """
    if file_type == 'mit':
        assert file_type == 'mit' and glob.glob(file_name + '.*'), \
            'Files not found'
    else:
        assert os.path.isfile(file_name), 'File not found'
    assert file_type in ['edf', 'mit', 'csv'], \
        'Only edf, mit (Physionet.org), and csv are supported.'
    assert isinstance(channel_num, list) or channel_num is None, \
        'Channel num must be a list starting from 0 or None'
    assert isinstance(channel_name, list) or channel_name is None, \
        'Channel name must be a str list or None.'
    assert not (channel_name is not None and channel_num is not None), \
        'Specify either channel name or channel index(s) or None'
    assert isinstance(start_datetime, str) or start_datetime is None, \
        'Start datetime must be None or a string.'
    assert isinstance(sampling_rate, float) or \
           isinstance(sampling_rate, int) or \
           sampling_rate is None, \
        'Sampling rate must be a number or None.'

    if isinstance(sampling_rate, float):
        sampling_rate = round(sampling_rate)
    if start_datetime is not None:
        start_datetime = utils.parse_datetime(start_datetime)

    if file_type == 'edf':
        signals, signal_headers, header = highlevel.read_edf(
            edf_file=file_name,
            ch_nrs=channel_num,
            ch_names=channel_name)
        if sampling_rate is None:
            try:
                sampling_rate = signal_headers[0]['sample_rate']
            except KeyError:
                print("sampling_rate is not defined and could not be "
                      "obtained from the signal's header.")
        else:
            signal_headers[0]['sample_rate'] = sampling_rate
        if start_datetime is None:
            try:
                start_datetime = header['startdate']
            except KeyError:
                print("start datetime is not defined and could not be "
                      "obtained from the signal's header.")
                pass
        else:
            header['startdate'] = start_datetime
        signals = signals.transpose()
        info = [header, signal_headers]
        out = SignalSQI(signals=signals,
                        wave_type='ecg',
                        sampling_rate=sampling_rate,
                        start_datetime=start_datetime,
                        info=info)

    if file_type == 'mit':
        signals, info = rdsamp(record_name=file_name,
                               channels=channel_num,
                               channel_names=channel_name,
                               warn_empty=True,
                               return_res=64)
        if sampling_rate is None:
            try:
                sampling_rate = info['fs']
            except KeyError:
                print("sampling_rate is not defined and could not be "
                      "obtained from the signal's header.")
        else:
            info['fs'] = sampling_rate
        if start_datetime is None:
            try:
                date = info['base_date']
                time = info['base_time']
                if date is not None and isinstance(date, dt.date):
                    start_datetime = dt.datetime(year=date.year,
                                                 month=date.month,
                                                 day=date.day, hour=0,
                                                 minute=0,
                                                 second=0, microsecond=0)
                    if time is not None and isinstance(time, dt.time):
                        start_datetime.hour = time.hour
                        start_datetime.minute = time.minute
                        start_datetime.second = time.second
                        start_datetime.microsecond = time.microsecond
            except KeyError:
                print("start datetime is not defined and could not be "
                      "obtained from the signal's header.")
                pass
        else:
            info['base_date'] = start_datetime.date()
            info['base_time'] = start_datetime.time()

        out = SignalSQI(signals=signals, wave_type='ecg',
                        sampling_rate=sampling_rate,
                        start_datetime=start_datetime,
                        info=info)
    if file_type == 'csv':
        use_cols = None
        if channel_name is not None:
            use_cols = channel_name
        if channel_num is not None:
            use_cols = channel_num
        out = pd.read_csv(file_name,
                          usecols=use_cols,
                          skipinitialspace=True)
        timestamps = out.iloc[:, 0].to_numpy()
        if start_datetime is None:
            try:
                start_datetime = utils.parse_datetime(timestamps[0])
            except Exception:
                pass
        if sampling_rate is None:
            sampling_rate = utils.calculate_sampling_rate(timestamps)
            assert sampling_rate is not None, 'Sampling rate not found nor ' \
                                              'inferred'
        signals = out.drop(0).to_numpy()
        out = SignalSQI(signals=signals, wave_type='ecg',
                        sampling_rate=sampling_rate,
                        start_datetime=start_datetime)
    return out


def ECG_writer(signal_sqi, file_name, file_type, info=None):
    """

    Parameters
    ----------
    signal_sqi : SignalSQI object containing signals, sampling rate and sqi

    file_name : name of file to write, with extension. For edf file_type,

    possible extensions are edf, edf+, bdf, bdf+. For mit file_type, :

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


    """
    signals = signal_sqi.signals
    sampling_rate = signal_sqi.sampling_rate
    start_datetime = signal_sqi.start_datetime
    assert isinstance(sampling_rate, int) or isinstance(sampling_rate,
                                                        float), \
        'sampling rate must be either int or float'
    if file_type == 'edf':
        signals = signals.transpose()
        if info is not None:
            signal_headers = info[1]
            header = info[0]
            annotations = header['annotations']
            # issue https://github.com/holgern/pyedflib/issues/119
            for i in range(len(annotations)):
                if isinstance(header['annotations'][i][1], bytes):
                    header['annotations'][i][1] = \
                        float(str(header['annotations'][i][1], 'utf-8'))
            highlevel.write_edf(file_name, signals, signal_headers,
                                header, file_type=-1, digital=False,
                                block_size=-1)
        else:
            highlevel.write_edf_quick(file_name, signals, sampling_rate)
        return os.path.isfile(file_name)
    if file_type == 'mit':
        if info is None:
            raise Exception("Header dict needed")
        else:
            wrsamp(record_name=file_name.split('/')[-1],
                   fs=sampling_rate,
                   units=info['units'],
                   sig_name=info['sig_name'],
                   p_signal=signals,
                   base_date=info['base_date'],
                   base_time=info['base_time'],
                   comments=info['comments'],
                   write_dir='/'.join(file_name.split('/')[:-1]))
        return glob.glob(file_name + '.*')
    if file_type == 'csv':
        timestamps = generate_timestamp(start_datetime, sampling_rate,
                                        signals.shape[0])
        signals = pd.DataFrame(np.hstack((np.array(timestamps).reshape(-1, 1),
                                          signals)))
        signals.to_csv(path_or_buf=file_name, index=False, header=True)
        return os.path.isfile(file_name)


def PPG_reader(file_name, signal_idx, timestamp_idx, info_idx,
               timestamp_unit='ms', sampling_rate=None,
               start_datetime=None):
    """

    Parameters
    ----------
    file_name : str
        absolute path to ppg file

    signal_idx : list
        name of one column containing signal

    timestamp_idx : list
        name of one column containing timestamps

    info_idx : list
        name of the columns for other info

    timestamp_unit : str
        unit of timestamp, only 'ms' or 's' accepted
         (Default value = 'ms')
    sampling_rate : float
        if None, sampling_rate can be inferred from the
        timestamps
         (Default value = None)
    start_datetime : str
        in '%Y-%m-%d '%H:%M:%S.%f' format
         (Default value = None)

    Returns
    -------
    object of class SignalSQI

    """
    cols = timestamp_idx + signal_idx + info_idx
    tmp = pd.read_csv(file_name,
                      usecols=cols,
                      skipinitialspace=True,
                      skip_blank_lines=True)
    timestamps = tmp[timestamp_idx[0]]
    if start_datetime is None:
        start_datetime = timestamps[0]
    if isinstance(start_datetime, str):
        try:
            start_datetime = dt.datetime.strptime(start_datetime, '%Y-%m-%d '
                                                                  '%H:%M:%S')
        except Exception:
            start_datetime = None
            pass
    else:
        start_datetime = None
    if sampling_rate is None:
        if timestamp_unit is None:
            raise Exception("Missing sampling_rate, not able to infer "
                            "sampling_rate without timestamp_unit")
        elif timestamp_unit == 'ms':
            timestamps = timestamps / 1000
        elif timestamp_unit != 's':
            raise Exception("Timestamp unit must be either second (s) or "
                            "millisecond (ms)")
        sampling_rate = utils.calculate_sampling_rate(timestamps.to_numpy())
    signals = tmp[signal_idx].to_numpy().T
    info = tmp[info_idx].to_dict('list')
    out = SignalSQI(signals=signals, wave_type='ppg',
                    sampling_rate=sampling_rate,
                    start_datetime=start_datetime,
                    info=info)
    return out


def PPG_writer(signal_sqi, file_name, file_type='csv'):
    """

    Parameters
    ----------
    signal_sqi : object of class SignalSQI

    file_name : str
        absolute path

    file_type : str
        file type to write, either 'csv' or 'xlsx'
    Returns
    -------
    bool
    """
    timestamps = utils.generate_timestamp(
        start_datetime=signal_sqi.start_datetime,
        sampling_rate=signal_sqi.sampling_rate,
        signal_length=len(signal_sqi.signals[0]))
    signals = signal_sqi.signals[0]
    timestamps = np.array(timestamps)
    out_df = pd.DataFrame({'time': timestamps, 'pleth': signals})
    if file_type == 'csv':
        out_df.to_csv(file_name, index=False, header=True)
    if file_type == 'xlsx':
        out_df.to_excel(file_name, index=False, header=True)
    return os.path.isfile(file_name)

# import os, tempfile
# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/example.edf')
# out = ECG_reader(file_in, 'edf')
# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/out.edf')
# out1 = ECG_reader(file_in, 'edf')
# file_out = '/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests/test_data/out.edf'
# out.sampling_rate = 15.8
# out.info[0]['annotations'][0][1] = float(str(out.info[0]['annotations'][0][
#                                                 1], 'utf-8'))
# ECG_writer(out, file_out, file_type='edf', info=out.info)

# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/a103l')
# out = ECG_reader(file_in, 'mit')
# file_out = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/out_mit')
# ECG_writer(out, file_out, file_type='mit', info=out.info)
# out = PPG_reader('/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests'
#             '/test_data/ppg_smartcare.csv', timestamp_idx = [
#     'TIMESTAMP_MS'], signal_idx = ['PLETH'], info_idx = ['PULSE_BPM',
#                                                          'SPO2_PCT','PERFUSION_INDEX'],
#                  start_datetime = '2020-04-12 10:00:00')
# out.sampling_rate = 2
# PPG_writer(out, '/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests'
#             '/test_data/ppg_smartcare_w.csv')
# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/ecg_test2.csv')
# out = ECG_reader(file_in, 'csv', channel_name = ['Time', '1'])
# file_out = '/Users/haihb/Documents/Work/Oucru/innovation ' \
#            '/vital_sqi/tests/test_data/ecg_test_write.csv'
# ECG_writer(out, file_out, file_type = 'csv')

