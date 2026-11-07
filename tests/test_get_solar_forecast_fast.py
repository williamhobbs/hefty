import pytest
from pandas.testing import assert_frame_equal
import pandas as pd

from hefty.solar import get_solar_forecast_fast


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
    rd = get_solar_forecast_fast(
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
        'ghi_csi': [0.6292502775807292],
        'dni_csi': [0.47865940464350987],
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
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority=None,
        decomp_model=None)
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

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_gfs_more():
    # get hardcoded valid times with
    # rd.index  # copy output
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 1
    lead_time_to_start = 121
    model = 'gfs'
    remove_gribs = True
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority='dynamical',
        decomp_model='erbs', remove_gribs=remove_gribs)
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-29 12:30:00+00:00', '2026-09-29 13:30:00+00:00',
             '2026-09-29 14:30:00+00:00']),
        'point': [0, 0, 0],
        'temp_air': [18.25, 20.75, 23.25],
        'wind_speed': [1.5242455005645752, 1.78822922706604,
                       2.052213191986084],
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
    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


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
    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


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
        'wind_speed': [2.751185894012451, 2.7751731872558594,
                       2.7991604804992676],
        'wind_direction': [64.79106140136719, 59.31123352050781,
                           53.83140563964844],
        'lead_time': [18.5, 19.5, 20.5],
        'ghi_csi': [0.7110017944688483, 0.7110017944688483,
                    0.7110017944688483],
        'ghi': [630.5753338474505, 564.0577841234704, 454.5918092647134],
        'dni': [350.66526297230183, 326.0851572150319, 329.759633861356],
        'dhi': [347.63148915692693, 325.52157056331725, 254.79483984928584],
        'ghi_clear': [886.8829006521986, 793.3282145157566, 639.3680196043877]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_ifs_ens_set_member_dynamical():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 00:00:00+0000')
    run_length = 3
    lead_time_to_start = 18
    model = 'ifs_ens'
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=2,
        attempts=2, hrrr_hour_middle=None,
        hrrr_coarsen_window=None, priority='dynamical',
        decomp_model=None)
    # hard-coded reference
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 18:30:00+00:00', '2026-09-24 19:30:00+00:00',
             '2026-09-24 20:30:00+00:00']),
        'point': [0, 0, 0],
        'temp_air': [24.6875, 25.0625, 25.4375],
        'wind_speed': [3.3296890258789062, 3.1125760078430176,
                       2.8954625129699707],
        'wind_direction': [56.30704879760742, 49.509033203125,
                           42.71101760864258],
        'lead_time': [18.5, 19.5, 20.5],
        'ghi_csi': [0.54492838262211, 0.54492838262211,
                    0.54492838262211],
        'ghi': [483.28766462760797, 432.30706082455754, 348.40978082332043],
        'dni': [107.05566639438575, 93.80106603814811, 96.02668567202704],
        'dhi': [396.9068763260129, 363.69016543520104, 290.2284911499296],
        'ghi_clear': [886.8829006521985, 793.3282145157566, 639.3680196043877]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")
    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_hrrr_subhourly_interval():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 3
    lead_time_to_start = 3
    model = 'hrrr'
    output_interval = 15
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, output_interval=output_interval)
    # only use first 3 rows
    rd = rd.iloc[:3]
    # hard-coded reference
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 15:07:30+00:00', '2026-09-24 15:22:30+00:00',
             '2026-09-24 15:37:30+00:00']),
        'point': [0, 0, 0],
        'temp_air': [23.631500244140625, 24.047882080078125,
                     24.464263916015625],
        'wind_speed': [3.0211355686187744, 3.000429391860962,
                       2.979722499847412],
        'wind_direction': [109.44141387939453, 106.22640228271484,
                           103.0113754272461],
        'ghi_csi': [0.5356024424978777, 0.598034332553112,
                    0.6604662226083463],
        'dni_csi': [0.3410329784870825, 0.43278392925803405,
                    0.5245348800289856],
        'lead_time': [3.125, 3.375, 3.625],
        'ghi_clear': [695.5476023007343, 734.820059983353, 770.6235422942924],
        'dni_clear': [944.4596247775394, 955.1578743641159,
                      964.3383583094261],
        'ghi': [372.46496310295225, 439.36152608733687, 508.8701249176209],
        'dni': [322.06768727098057, 413.34978211133097, 505.7994999586545],
        'dhi': [162.48049641531028, 156.61716625749398, 148.17278498652348]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)


def test_gfs_dynamical_subhourly_interval():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 12:00:00+0000')
    run_length = 3
    lead_time_to_start = 3
    model = 'gfs'
    priority = 'dynamical'
    output_interval = 15
    rd = get_solar_forecast_fast(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, member=None,
        attempts=2, priority=priority,
        output_interval=output_interval)
    # only use first 3 rows
    rd = rd.iloc[:3]
    # hard-coded reference
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 15:07:30+00:00', '2026-09-24 15:22:30+00:00',
             '2026-09-24 15:37:30+00:00']),
        'point': [0, 0, 0],
        'temp_air': [22.890625, 23.171875, 23.453125],
        'wind_speed': [2.9044125080108643, 2.781244993209839,
                       2.6580777168273926],
        'wind_direction': [97.39085388183594, 96.27518463134766,
                           95.1595230102539],
        'lead_time': [3.125, 3.375, 3.625],
        'ghi_csi': [0.5980380050167194, 0.5980380050167194,
                    0.5980380050167194],
        'ghi': [415.9639004740937, 439.45032271871054, 460.86216585259615],
        'dni': [138.04931763670572, 137.2237903769144, 136.45542410425466],
        'dhi': [325.95731649739537, 345.5848982218976, 363.5526420242853],
        'ghi_clear': [695.5476023007343, 734.820059983353, 770.6235422942924]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")

    assert_frame_equal(rd, rd_test, check_dtype=False, rtol=1e-4)
