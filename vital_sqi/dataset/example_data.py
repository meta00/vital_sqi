from vital_sqi.data.signal_io import ECG_reader,PPG_reader
import os

def load_ecg():
    module_path = os.path.dirname(__file__)
    file_name = "example.edf"
    ecg_data = ECG_reader(os.path.join(module_path,file_name), 'edf')
    return ecg_data

def load_ppg():
    module_path = os.path.dirname(__file__)
    file_name = "ppg_smartcare.csv"
    ppg_data = PPG_reader(os.path.join(module_path,file_name),
                          signal_idx=['PLETH'],
                          timestamp_idx=['TIMESTAMP_MS'],
                          info_idx=['SPO2_PCT', 'PULSE_BPM', 'PERFUSION_INDEX'])
    return ppg_data