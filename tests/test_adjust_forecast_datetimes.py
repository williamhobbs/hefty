import pytest
import pandas as pd

from hefty.utilities import adjust_forecast_datetimes


def test_tz_warning():
    available_date = ('2026-09-24 10:00')
    run_length_needed = 15
    lead_time_to_start_needed = 1
    with pytest.warns(
        UserWarning,
        match='You have provided a timezone-naive'
    ) as warn:
        init_date, run_length, lead_time_to_start = adjust_forecast_datetimes(
            available_date, run_length_needed, lead_time_to_start_needed,
            model='gfs')
        assert init_date == pd.Timestamp('2026-09-24 00:00:00+0000', tz='UTC')
        assert run_length == 15
        assert lead_time_to_start == 11
