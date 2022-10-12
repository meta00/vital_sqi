Introduction
============

The introduction... some of this content can be just copied/pasted froom the
manuscript and/or viceversa.

Background
----------

The background

ECG signals
~~~~~~~~~~~

Brief intro to ECG?

PPG signals
~~~~~~~~~~~

Brief intro to PPG?

SQI indexes
------------

The Signal Quality Indexes (SQIs) are .... These are classified into the
following three domains: (i) statistical domain where ... (ii) signal processing
domain where and (iii) dynamic time warping domain. A summary of the SQI metrics
is presented below.

======================= ============================================== =============
Acronym                        Domain                                      Status
======================= ============================================== =============
``perfusion``
``kurtosis``               statistical
``skewness``               statistical
``entropy``                signal processing
``signal_to_noise``
``zero_crossings_rate``    signal processing
``mean_crossing_rate``     signal processing
``DTW``                    dynamic time warping
``rpeaks``
``XXXXX``
======================= ============================================== =============

.. warning:: Include [R] in the previous table.

| [1] Optimal Signal Quality Index for Photoplethysmogram Signals, (Mohamed Elgendi et al)
| [2] ...
| [3] ...

- **Perfusion**

    Brief description

    See: :py:func:`vital_sqi.sqi.standard_sqi.perfusion_sqi`



- **Kurtosis**

    Brief description or link to scipy function.

    See: :py:func:`vital_sqi.sqi.standard_sqi.kurtosis_sqi`



- **Skewness**

    Brief description or link to scipy function.

    See: :py:func:`vital_sqi.sqi.standard_sqi.skewness_sqi`



- **Signal to noise**

    Brief description

    See: :py:func:`vital_sqi.sqi.standard_sqi.signal_to_noise_sqi`



- **Zero crossings rate**

    Brief description

    See: :py:func:`vital_sqi.sqi.standard_sqi.zero_crossings_rate_sqi`



- **Mean crossing rate**

    Brief description

    See: :py:func:`vital_sqi.sqi.standard_sqi.mean_crossing_rate_sqi`



- **Dynamic time warping**

    Brief description

    See: :py:func:`vital_sqi.sqi.dtw_sqi.dtw_sqi`

