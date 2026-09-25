import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose
import pandas as pd

from hefty.solar import get_solar_forecast_fast


# ignore xarray FutureWarnings (see https://github.com/blaylockbk/Herbie/issues/525).
# ignore grib file removal warnings
@pytest.mark.filterwarnings('ignore:.*Will not remove GRIB.*')
@pytest.mark.filterwarnings('ignore:.*In a future version.*')
def test_hrrr():
    latitude = 33.5
    longitude = -86.8
    # init_date = '2026-09-20 18:00'
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 3
    model = 'hrrr'
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority=None,
        decomp_model=None)
    # hard-coded reference
    valid_time = init_date + pd.Timedelta(hours=lead_time_to_start) + pd.Timedelta('30min')
    data = {
        'valid_time': [valid_time],
        'point': [0],
        'temp_air': [24.256072998046875],
        'wind_speed': [2.9900758266448975],
        'wind_direction': [104.61888885498047],
        'csi_ghi': [0.631432385621549],
        'csi_dni': [0.4797910002746542],
        'lead_time': [3.5],
        'ghi_clear': [753.1687745026209],
        'dni_clear': [959.9273121182977],
        'ghi': [474.07211375495996],
        'dni': [460.10400390316994],
        'dhi': [152.48033190527235]
        }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")
    # assert pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False)
    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False)

