import pytest
import numpy as np
import pandas as pd
from vital_sqi.common.utils import check_valid_signal


class TestCheckInvalidSignal(object):

    def test_on_empty_signal(self):
        x = [[], np.array([]), pd.Series()]
        for i in x:
            with pytest.raises(ValueError) as exec_info:
                check_valid_signal(i)
            assert exec_info.match("Empty signal")

    def test_on_invalid_signal(self):
        x = [[1, True], [1, '1'],
             np.array([1, '1']),
             pd.Series([1, True], dtype = object)]
        for i in x:
            with pytest.raises(ValueError) as exec_info:
                check_valid_signal(i)
            assert exec_info.match("Invalid signal")

    def test_on_invalid_input_type(self):
        x = [(1, 1), {'a': 1, 'b': 2}]
        for i in x:
            with pytest.raises(ValueError) as exec_info:
                check_valid_signal(i)
            assert exec_info.match("Expected array_like input")
