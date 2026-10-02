import pytest
from pandas.testing import assert_frame_equal
import pandas as pd

from hefty.solar import get_solar_forecast


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
    rd = get_solar_forecast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority=None,
        decomp_model=None)
    # hard-coded reference
    valid_time = (init_date +
                  pd.Timedelta(hours=lead_time_to_start) +
                  pd.Timedelta('30min'))
    data = {
        'valid_time': [valid_time],
        'point': [0],
        'temp_air': [24.256072998046875],
        'wind_speed': [2.9900758266448975],
        'wind_direction': [104.61888885498047],
        'csi_ghi': [0.6292502775807292],
        'csi_dni': [0.47865940464350987],
        'lead_time': [3.5],
        'ghi_clear': [753.1687745026209],
        'dni_clear': [959.9273121182977],
        'ghi': [472.43381233914823],
        'dni': [459.0188404040822],
        'dhi': [151.60051044249167]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_gfs():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 3
    model = 'gfs'
    rd = get_solar_forecast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority=None,
        decomp_model=None)
    # hard-coded reference
    valid_time = (init_date +
                  pd.Timedelta(hours=lead_time_to_start) +
                  pd.Timedelta('30min'))
    data = {
        'valid_time': [valid_time],
        'point': [0],
        'temp_air': [23.263565063476562],
        'wind_speed': [2.7109782695770264],
        'wind_direction': [95.7496566772461],
        'lead_time': [3.5],
        'ghi_csi': [0.5939355873630466],
        'ghi': [447.33373846772014],
        'dni': [0.0],
        'dhi': [447.33373846772014],
        'ghi_clear': [753.1687745026209]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_ifs():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 3  # has to be at least 4, see GH #101
    lead_time_to_start = 3
    model = 'ifs'
    rd = get_solar_forecast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority=None,
        decomp_model=None)
    # hard-coded reference
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 15:30:00+00:00', '2026-09-24 16:30:00+00:00',
             '2026-09-24 17:30:00+00:00']),
        'point': [0, 0, 0],
        'temp_air': [20.94012451171875, 21.971435546875, 23.00274658203125],
        'wind_speed': [2.938384532928467, 2.871222972869873,
                       2.8040616512298584],
        'wind_direction': [96.08087158203125, 89.96724700927734,
                           83.85362243652344],
        'lead_time': [3.5, 4.5, 5.5],
        'ghi_csi': [0.3231279972722611, 0.3231279972722611,
                    0.3231279972722611],
        'ghi': [243.3699177130351, 279.78824199483597, 294.62145468044685],
        'dni': [8.79092829420811, 12.595251349949525, 13.32957547777653],
        'dhi': [237.2254584815041, 269.83786742850236, 283.60020385717013],
        'ghi_clear': [753.1687745026209, 865.8743419224428, 911.7794099166367]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_cams():
    # get cams api key
    import os
    from dotenv import load_dotenv
    load_dotenv()  # Load variables from .env locally
    cams_api_key = os.getenv("CAMS_API_KEY")
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 3
    lead_time_to_start = 3
    model = 'cams'
    cams_area = [50, -125, 20, -65]  # approx. CONUS
    rd = get_solar_forecast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, cams_api_key=cams_api_key,
        cams_area=cams_area)
    rd.to_dict(orient='list')
    # hard-coded reference
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 15:30:00+00:00', '2026-09-24 16:30:00+00:00',
             '2026-09-24 17:30:00+00:00']),
        'point': [0, 0, 0],
        'temp_air': [22.341873225508635, 22.911826903976618,
                     23.788398479624277],
        'wind_speed': [2.8496116136644716, 2.8169419054201055,
                       2.8401152104367062],
        'wind_direction': [103.88997928071872, 97.27503899507383,
                           91.24254131275154],
        'ghi': [303.9862726382586, 503.8141734705507, 544.4980214855266],
        'dni': [105.27155017726712, 328.7821481625715, 370.7863917035832],
        'dhi': [230.40624483499937, 244.07298736436843, 237.92187945939042],
        'ghi_clear': [678.9016332193258, 785.1906050794173, 828.7721381052273],
        'dni_clear': [840.6378638909612, 875.1202873351015, 889.0207824053701],
        'lead_time': [3.5, 4.5, 5.5],
        'direct_horiz_clear': [587.5676504564902,
                               691.3537814881844,
                               735.0662477084148]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)
