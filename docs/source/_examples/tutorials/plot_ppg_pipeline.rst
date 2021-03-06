
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "_examples\tutorials\plot_ppg_pipeline.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__examples_tutorials_plot_ppg_pipeline.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr__examples_tutorials_plot_ppg_pipeline.py:


PPG pipeline
====================

.. warning:: Include code to show how to create a pipeline to
             clean an ecg signal. It is probably in one of the
             jupyter notebooks (maybe ecg_qc.ipynb).

.. GENERATED FROM PYTHON SOURCE LINES 13-16

Load the data
-------------


.. GENERATED FROM PYTHON SOURCE LINES 16-26

.. code-block:: default
   :lineno-start: 17


    # First, lets load the data (ppg sample data)

    # Libraries
    import pandas as pd

    # Load data
    #data = pd.read_csv(path)









.. GENERATED FROM PYTHON SOURCE LINES 27-34

Preprocessing
-------------

Wearable devices need some time to pick up stable signals. For this reason,
it is a common practice to trim the data. In the following example, the f
first and last 5 minutes of each recording are trimmed to exclude unstable
signals.

.. GENERATED FROM PYTHON SOURCE LINES 34-38

.. code-block:: default
   :lineno-start: 35


    # Trim data
    #data = trim(data, start=5, end=5)








.. GENERATED FROM PYTHON SOURCE LINES 39-49

Now, lets remove the following noise:

  - ``PLETH`` is 0 or unchanged values for xxx time
  - ``SpO2`` < 80
  - ``Pulse`` > 200 bpm or ``Pulse`` < 40 bpm
  - ``Perfusion`` < 0.2
  - ``Lost connection``: sampling rate reduced due to (possible) Bluetooth connection
    lost. Timestamp column shows missing timepoints. If the missing duration is
    larger than 1 cycle (xxx ms), recording is split. If not, missing timepoints
    are interpolated.

.. GENERATED FROM PYTHON SOURCE LINES 49-65

.. code-block:: default
   :lineno-start: 50


    # Remove invalid PLETH
    #idxs_1 = data.PLETH == 0
    #idxs_2 = unchanged(period=xxxx)
    #data = data[~(idxs_1 | idxs_2)]

    # Remove invalid ranges
    #data = data[data.SpO2>=80]
    #data = data[data.Pulse.between(40, 200)]
    #data = data[data.Perfusion>=0.2]

    # Remove lost connection
    #data = data[lost_connection(min_fs, max_fs) ??

    # The recording is then split into files.








.. GENERATED FROM PYTHON SOURCE LINES 66-67

Lets filter the data with a band pass filter; high pass filter (cut off at 1Hz)

.. GENERATED FROM PYTHON SOURCE LINES 69-70

Lets detrend the signal

.. GENERATED FROM PYTHON SOURCE LINES 70-77

.. code-block:: default
   :lineno-start: 71


    # Lets split the data
    #4.1. Cut data by time domain. Split data into sub segments of 30 seconds
    #4.2. Apply the peak and trough detection methods in peak_approaches.py to get single PPG cycles in each segment
    #4.3. Shift baseline above 0 and tapering each single PPG cycle to compute the mean template
    #Notes: the described process is implemented in split_to_segments.py








.. GENERATED FROM PYTHON SOURCE LINES 78-82

SQI scores
----------

Lets compute the SQI scores

.. GENERATED FROM PYTHON SOURCE LINES 84-85

Visualization
-------------


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.313 seconds)


.. _sphx_glr_download__examples_tutorials_plot_ppg_pipeline.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_ppg_pipeline.py <plot_ppg_pipeline.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_ppg_pipeline.ipynb <plot_ppg_pipeline.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
