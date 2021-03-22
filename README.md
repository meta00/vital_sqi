# ECG and PPG SQI analyses

## PPG
1. Data format
   
2. Preprocessing
2.1. Trimming: First and last 5 minutes of each recording are trimmed as they usually contain unstable signals. Wearables need some time to pick up signals.
2.2. Noise removal: The following is considered noise, thus removed. The recording is then split into files.
   - PLETH column: 0 or unchanged values for xxx time
   - Invalid values: SpO2 < 80 and Pulse > 200 or Pulse < 40
   - Lost connection: sampling rate reduced due to (possible) Bluetooth connection lost. Timestamp column shows missing timepoints. If the missing duration is larger than 1 cycle (xxx ms), recording is split. If not, missing timepoints are interpolated.
3. Filtering

## ECG preprocessing
