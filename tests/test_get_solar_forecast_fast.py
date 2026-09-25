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

    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_gfs():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 3
    model = 'gfs'
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority=None,
        decomp_model=None)
    valid_time = init_date + pd.Timedelta(hours=lead_time_to_start) + pd.Timedelta('30min')
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

    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_hrrr_other_params():
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
        attempts=2, hrrr_hour_middle=False,
        hrrr_coarsen_window=10, priority=None,
        decomp_model=None)
    valid_time1 = init_date + pd.Timedelta(hours=lead_time_to_start)
    valid_time2 = init_date + pd.Timedelta(hours=lead_time_to_start+1)
    data = {
        'valid_time': [valid_time1, valid_time2],
        'point': [0, 0],
        'ghi': [402.9310302734375, 591.2730102539062],
        'dni': [262.1600036621094, 502.7799987792969],
        'temp_air': [22.825775146484375, 24.101348876953125],
        'wind_speed': [3.240015745162964, 3.2148265838623047],
        'wind_direction': [108.54864501953125, 94.069091796875],
        'lead_time': [3.0, 4.0],
        'dhi': [236.48487923850308, 213.70769262088743],
        'ghi_clear': [674.681002563978, 817.3830180294453],
        'dni_clear': [938.4819870380996, 975.5930978834822]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_gfs_more():
    # get hardcoded valid times with
    # rd.index  # copy output
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 121
    model = 'gfs'
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority='dynamical',
        decomp_model='erbs')
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-29 12:30:00+00:00', '2026-09-29 13:30:00+00:00',
            '2026-09-29 14:30:00+00:00']),
        'point': [0, 0, 0],
        'temp_air': [18.25, 20.75, 23.25],
        'wind_speed': [1.5242455005645752, 1.78822922706604, 2.052213191986084],
        'wind_direction': [105.40540313720703,
                        111.26321411132812,
                        117.12102508544922],
        'lead_time': [120.5, 121.5, 122.5],
        'ghi_csi': [0.917293933552123, 0.917293933552123, 0.917293933552123],
        'ghi': [124.70278528048202, 328.6619124858044, 519.2225172904367],
        'dni': [349.9548599217125, 604.9525757024749, 720.5913775255551],
        'dhi': [67.04305177247089, 105.92832798209416, 126.62847660399292],
        'ghi_clear': [135.9463752230256, 358.29509000795, 566.0372300510076]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")
    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)

def test_ifs():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 3
    lead_time_to_start = 3
    model = 'ifs'
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority='google',
        decomp_model=None)
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 15:30:00+00:00',
            '2026-09-24 16:30:00+00:00',
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
    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)

def test_ifs_ens():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 00:00:00+0000')
    run_length = 3
    lead_time_to_start = 18
    model = 'ifs_ens'
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=0,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority='dynamical',
        decomp_model=None)
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 18:30:00+00:00',
            '2026-09-24 19:30:00+00:00',
            '2026-09-24 20:30:00+00:00'],
            dtype='datetime64[ns, UTC]'),
        'point': [0, 0, 0],
        'temp_air': [24.22916603088379, 24.9375, 25.64583396911621],
        'wind_speed': [2.751185894012451, 2.7751731872558594, 2.7991604804992676],
        'wind_direction': [64.79106140136719, 59.31123352050781, 53.83140563964844],
        'lead_time': [18.5, 19.5, 20.5],
        'ghi_csi': [0.7110017944688483, 0.7110017944688483, 0.7110017944688483],
        'ghi': [630.5753338474505, 564.0577841234704, 454.5918092647134],
        'dni': [350.66526297230183, 326.0851572150319, 329.759633861356],
        'dhi': [347.63148915692693, 325.52157056331725, 254.79483984928584],
        'ghi_clear': [886.8829006521986, 793.3282145157566, 639.3680196043877]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    pd.testing.assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)
