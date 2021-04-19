import pytest
from vital_sqi.sqi.dtw_sqi import dtw_sqi

class TestDtwSqi(object):
    x = [0, 1, 2, 3]

    def test_on_invalid_template_type(self):
        template_types = [2.1, 4]
        for i in template_types:
            with pytest.raises(ValueError) as exc_info:
                dtw_sqi(self.x, i)
            assert exc_info.match("Invalid template type")

    def test_on_valid_template_type_0(self):
        template_types = [0, 1, 2, 3]
        for i in template_types:
            assert type(dtw_sqi(self.x, i)) is float

    def test_on_trace_equal_0(self):
        x = [0]
        template_types = [0, 1, 2, 3]
        for i in template_types:
            assert type(dtw_sqi(x, i)) is float
