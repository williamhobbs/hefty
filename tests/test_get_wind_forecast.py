import pytest
from pandas.testing import assert_frame_equal
import pandas as pd

from hefty.wind import get_wind_forecast


# ignore xarray FutureWarnings (see https://github.com/blaylockbk/Herbie/issues/525).
# ignore grib file removal warnings
pytestmark = [
    pytest.mark.filterwarnings('ignore:.*Will not remove GRIB.*'),
    pytest.mark.filterwarnings('ignore:.*In a future version.*')
]


def test_hrrr():
    # obtain reference dataframe with:
    # rd.to_dict(orient='list')  # copy output
    # valid_time = init_date + pd.Timedelta(hours=lead_time_to_start) + pd.Timedelta('30min')
    # then paste into "data" below
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 3
    model = 'hrrr'
    rd = get_wind_forecast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2)
    # hard-coded reference
    valid_time = (init_date +
                  pd.Timedelta(hours=lead_time_to_start) +
                  pd.Timedelta('30min'))
    data = {
        'valid_time': [valid_time],
        'point': [0],
        'temp_air_2m': [24.256072998046875],
        'pressure_0m': [99680.0],
        'wind_speed_10m': [2.9900758266448975],
        'wind_speed_80m': [3.66703462600708],
        'wind_direction_10m': [104.61888885498047],
        'wind_direction_80m': [105.27836608886719],
        'lead_time': [3.5]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_gfs_remove_gribs():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 3
    model = 'gfs'
    remove_gribs = True
    rd = get_wind_forecast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, remove_gribs=remove_gribs)
    # hard-coded reference
    valid_time = (init_date +
                  pd.Timedelta(hours=lead_time_to_start) +
                  pd.Timedelta('30min'))
    data = {
        'valid_time': [valid_time],
        'point': [0],
        'pressure_80m': [98680.078125],
        'temp_air_80m': [21.327072143554688],
        'temp_air_2m': [23.263565063476562],
        'pressure_0m': [99590.3984375],
        'wind_speed_10m': [2.7109782695770264],
        'wind_speed_100m': [4.106931686401367],
        'wind_speed_80m': [3.9878408908843994],
        'wind_direction_10m': [95.7496566772461],
        'wind_direction_100m': [97.1158447265625],
        'wind_direction_80m': [97.1925048828125],
        'lead_time': [3.5]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)
