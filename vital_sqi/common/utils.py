import numpy as np
import datetime as dt
from datetimerange import DateTimeRange
import dateparser
def check_valid_signal(x):
    """Check whether signal is valid, i.e. an array_like numeric, or raise errors.

    Parameters
    ----------
    x :
        array_like, array of signal

    Returns
    -------

    
    """
    if isinstance(x, dict) or isinstance(x, tuple):
        raise ValueError("Expected array_like input, instead found {"
                        "0}:".format(type(x)))
    if len(x) == 0:
        raise ValueError("Empty signal")
    types = []
    for i in range(len(x)):
        types.append(str(type(x[i])))
    type_unique = np.unique(np.array(types))
    if len(type_unique) != 1 and (type_unique[0].find("int") != -1 or
                                  type_unique[0].find("float") != -1):
        raise ValueError("Invalid signal: Expect numeric array, instead found "
                         "array with types {0}: ".format(type_unique))
    if type_unique[0].find("int") == -1 and type_unique[0].find("float") == -1:
        raise ValueError("Invalid signal: Expect numeric array, instead found "
                         "array with types {0}: ".format(type_unique))
    return True


def calculate_sampling_rate(timestamps):
    """

    Parameters
    ----------
    x : array_like of timestamps, float (unit second)

    Returns
    -------
    float : sampling rate
    """
    if isinstance(timestamps[0], float):
        timestamps_second = timestamps
    else:
        try:
            v_parse_datetime = np.vectorize(parse_datetime)
            timestamps = v_parse_datetime(timestamps)
            timestamps_second = []
            timestamps_second.append(0)
            for i in range(1, len(timestamps)):
                timestamps_second.append((timestamps[i] - timestamps[
                i - 1]).total_seconds())
        except Exception:
            sampling_rate = None
            pass
    steps = np.diff(timestamps_second)
    sampling_rate = round(1 / np.min(steps[steps != 0]))
    return sampling_rate


def generate_timestamp(start_datetime, sampling_rate, signal_length):
    """

    Parameters
    ----------
    start_datetime :
        
    sampling_rate : float
        
    signal_length : int
        

    Returns
    -------
    list : list of timestamps with length equal to signal_length.
    """
    number_of_seconds = (signal_length - 1)/sampling_rate
    if start_datetime is None:
        start_datetime = dt.datetime.now()
    end_datetime = start_datetime + dt.timedelta(seconds=number_of_seconds)
    time_range = DateTimeRange(start_datetime, end_datetime)
    timestamps = []
    for value in time_range.range(dt.timedelta(seconds=1/sampling_rate)):
        timestamps.append(value)
    if len(timestamps) != signal_length:
        raise Exception("Timestamp series generated is not valid, please "
                        "check sampling rate.")
    return timestamps

def parse_datetime(string, type='datetime'):
    """
    A simple dateparser that detects common  datetime formats

    Parameters
    ----------
    string : str
        a date string in format as denoted below.

    Returns
    -------
    datetime.datetime
        datetime object of a time.

    """
    # some common formats.
    date_formats = ['%Y-%m-%d',
                    '%d-%m-%Y',
                    '%d.%m.%Y',
                    '%Y.%m.%d',
                    '%d %b %Y',
                    '%Y/%m/%d',
                    '%d/%m/%Y']
    datime_formats = ['%Y-%m-%d %H:%M:%S.%f',
                      '%d-%m-%Y %H:%M:%S.%f',
                      '%d.%m.%Y %H:%M:%S.%f',
                      '%Y.%m.%d %H:%M:%S.%f',
                      '%d %b %Y %H:%M:%S.%f',
                      '%Y/%m/%d %H:%M:%S.%f',
                      '%d/%m/%Y %H:%M:%S.%f',
                      '%Y-%m-%d %I:%M:%S.%f',
                      '%d-%m-%Y %I:%M:%S.%f',
                      '%d.%m.%Y %I:%M:%S.%f',
                      '%Y.%m.%d %I:%M:%S.%f',
                      '%d %b %Y %I:%M:%S.%f',
                      '%Y/%m/%d %I:%M:%S.%f',
                      '%d/%m/%Y %I:%M:%S.%f']
    if type == 'date':
        formats = date_formats
    if type == 'datetime':
        formats =  datime_formats
    for f in formats:
        try:
            return datetime.strptime(string, f)
        except:
            pass
    try:
        return dateparser.parse(string)
    except:
        raise ValueError('Datetime string must be of standard Python format '
                         '(https://docs.python.org/3/library/time.html), '
                         'e.g., `%d-%m-%Y`, eg. `24-01-2020`')
