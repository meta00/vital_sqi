from pyedflib import highlevel
from wfdb import rdsamp, wrsamp
import numpy as np
import pandas as pd
import datetime as dt
import os
import glob
from vital_sqi.common import utils
from vital_sqi.common.utils import generate_timestamp
from vital_sqi.data.signal_sqi_class import SignalSQI


def ECG_reader(file_name, file_type, channel_num=None,
               channel_name=None, sampling_rate=None,
               start_datetime=None):
    """

    Parameters
    ----------
    file_name : str
        Path to ECG file.
    file_type : str
       Supported types include 'edf', 'mit' or 'csv'.
    channel_num : list
        List of channel ids to read, starting from 0.
        (Default value = None)
    channel_name : list
        List of channel names to read.
        (Default value = None)
    sampling_rate : int or float
        (Default value = None)
    start_datetime : str
        In '%Y-%m-%d '%H:%M:%S.%f' format. If none or not convertible
        to datetime, it is assigned to now.
        (Default value = None)

    Returns
    -------
        out: SignalSQI
            SignalSQI object.
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
           sampling_rate is None, 'Sampling rate must be a number or None.'

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
        signals = pd.DataFrame(signals.transpose())
        timestamps = generate_timestamp(start_datetime, sampling_rate,
                                        len(signals))
        signals.insert(0, 'timestamps', timestamps)
        info = [header, signal_headers]
        out = SignalSQI(signals=signals,
                        wave_type='ecg',
                        start_datetime=start_datetime,
                        sampling_rate=sampling_rate,
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
        timestamps = generate_timestamp(start_datetime, sampling_rate,
                                        len(signals))
        signals = pd.DataFrame(signals)
        signals["timestamps"] = timestamps
        out = SignalSQI(signals=signals, wave_type='ecg',
                        sampling_rate=sampling_rate,
                        info=info)
    if file_type == 'csv':
        use_cols = None
        if channel_name is not None:
            use_cols = channel_name
        if channel_num is not None:
            use_cols = channel_num
        signals = pd.read_csv(file_name,
                          usecols=use_cols,
                          skipinitialspace=True)
        try:
            # start_datetime in datetime and column in second
            if start_datetime is not None and isinstance(signals.iloc[:, 0],
                                                         float):
                start_datetime = pd.Timestamp(start_datetime)
                timestamps[0] = start_datetime
                for i in range(1, len(signals)):
                    timestamps[i] = timestamps[i-1] + \
                                    pd.Timedelta(seconds=timestamps[i])
            # no start_datetime and column in datetime
            if start_datetime is None:
                timestamps = signals.iloc[:, 0].apply(pd.Timestamp)
            if sampling_rate is None:
                sampling_rate = utils.calculate_sampling_rate(signals.iloc[:,
                                                              0])
        except ValueError:
            assert sampling_rate is not None, \
                    'Sampling rate is not found nor able to be inferred ' \
                    'from the from the signal.'
            # if first column does not contain datetime. start_datetime = now
            # if none.
            timestamps = generate_timestamp(start_datetime, sampling_rate,
                                            len(signals))
        start_datetime = timestamps[0]
        out = SignalSQI(signals=signals,
                        info=[],
                        wave_type='ecg',
                        start_datetime=start_datetime,
                        sampling_rate=sampling_rate)
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
        
    info : list or dict
        In case of writing edf file: A list containing signal_headers and
        header (in order). signal_headers is a list of dict with one signal
        header for each signal channel. header (dict) contain ecg file
        information.
        In case of writing wfdb record (mit file): A dict containing header
        as defined in .hea file.
        (Default value = None)

    Returns
    -------

    
    """
    signals = signal_sqi.signals.loc[:, signal_sqi.signals.columns !=
                                'timestamps'].to_numpy()
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
            # issue https://github.com/holgern/pyedflib/issues/119 - fixed to
            # be checked
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


def PPG_reader(file_name, signal_idx, timestamp_idx, info_idx=[],
               timestamp_unit='ms', sampling_rate=None,
               start_datetime=None):
    """

    Parameters
    ----------
    file_name : str
        Path to ppg file.
    signal_idx : list
        Name or index of the signal column.
    timestamp_idx : list
        Name or index of the timestamp column.
    info_idx : list
        Name or indexes of the columns for other information.
        (Default value = [])
    timestamp_unit : str
        Unit of timestamp, only 'ms' or 's' accepted.
        (Default value = 'ms')
    sampling_rate : int or float
        if None, sampling_rate can be inferred from the
        timestamps.
        (Default value = None)
    start_datetime : str
        In '%Y-%m-%d '%H:%M:%S.%f' format. If none or not convertible
        to datetime, it is assigned to now.
        (Default value = None)

    Returns
    -------
        out: SignalSQI
            SignalSQI object.
    
    """
    cols = timestamp_idx + signal_idx + info_idx
    tmp = pd.read_csv(file_name,
                      usecols=cols,
                      skipinitialspace=True,
                      skip_blank_lines=True)
    for i in range(0, len(info_idx)):
        if isinstance(cols[i], str):
            info_idx[i] = tmp.columns.get_loc(info_idx[i])
    if isinstance(signal_idx[0], str):
        signal_idx[0] = tmp.columns.get_loc(signal_idx[0])
    if isinstance(timestamp_idx[0], str):
        timestamp_idx[0] = tmp.columns.get_loc(timestamp_idx[0])

    timestamps = tmp.iloc[:, timestamp_idx[0]]
    if isinstance(start_datetime, str):
        try:
            start_datetime = pd.Timestamp(start_datetime)
        except Exception:
            start_datetime = None
            pass
    if start_datetime is None:
        start_datetime = pd.Timestamp.now()

    if timestamp_unit is None:
        raise Exception("Missing sampling_rate, not able to infer "
                        "sampling_rate without timestamp_unit")
    elif timestamp_unit == 'ms':
        timestamps = timestamps / 1000
    elif timestamp_unit != 's':
        raise Exception("Timestamp unit must be either second (s) or "
                        "millisecond (ms)")

    for i in range(0, len(timestamps)):
        timestamps[i] = start_datetime + pd.Timedelta(timestamps[i],
                                                      unit='seconds')
    if sampling_rate is None:
        sampling_rate = utils.calculate_sampling_rate(timestamps)
    
    info = pd.DataFrame(tmp.iloc[:, info_idx])
    signals = tmp.iloc[:, signal_idx]
    signals.insert(0, 'timestamps', timestamps)
    out = SignalSQI(signals=signals, wave_type='ppg',
                    sampling_rate=sampling_rate,
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
         (Default value = 'csv')

    Returns
    -------

    
    """
    timestamps = utils.generate_timestamp(
        start_datetime=signal_sqi.start_datetime,
        sampling_rate=signal_sqi.sampling_rate,
        signal_length=len(signal_sqi.signals))
    signals = signal_sqi.signals.iloc[:, 1]
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
# print(out)

# import os, tempfile
#
# file_in = os.path.abspath('/Users/haihb/Documents/Oucru/innovation'
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

# file_in = os.path.abspath('/Users/haihb/Documents/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/a103l')
# out = ECG_reader(file_in, 'mit')

# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/a103l')
# out = ECG_reader(file_in, 'mit')
# file_out = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/out_mit')
# ECG_writer(out, file_out, file_type='mit', info=out.info)
# out = PPG_reader('/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests/test_data/ppg_smartcare.csv',
#                 timestamp_idx = ['TIMESTAMP_MS'], signal_idx = ['PLETH'], info_idx = ['PULSE_BPM',
#                                                         'SPO2_PCT','PERFUSION_INDEX'],
#                  start_datetime = '2020-04-12 10:00:00')
# out.sampling_rate = 2
#PPG_writer(out, 'D:/Workspace/oucru/medical_signal/Github/vital_sqi/vital_sqi/dataset/ppg_smartcare_w.csv')
# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/ecg_test2.csv')
# out = ECG_reader(file_in, 'csv', channel_name = ['Time', '1'])
# file_out = '/Users/haihb/Documents/Work/Oucru/innovation ' \
#            '/vital_sqi/tests/test_data/ecg_test_write.csv'
# ECG_writer(out, file_out, file_type = 'csv')

# import os
# file_name = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                             '/vital_sqi/tests/test_data/ecg_test1.csv')
# out = ECG_reader(file_name, 'csv', channel_name=['Time', '1'])