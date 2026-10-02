import pytest
from pandas.testing import assert_frame_equal
import pandas as pd

from hefty.solar import get_solar_forecast_ensemble


# ignore xarray FutureWarnings (see https://github.com/blaylockbk/Herbie/issues/525).
# ignore grib file removal warnings
pytestmark = [
    pytest.mark.filterwarnings('ignore:.*Will not remove GRIB.*'),
    pytest.mark.filterwarnings('ignore:.*In a future version.*')
]


def test_gefs_dynamical():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 00:00:00+0000')
    run_length = 3
    lead_time_to_start = 12
    model = 'gefs'
    rd = get_solar_forecast_ensemble(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, attempts=2,
        priority='dynamical')
    # calculate ensemble mean
    rd_mean = (rd.reset_index().
               set_index(['valid_time', 'member']).
               groupby('valid_time').mean())
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 12:30:00+00:00',
             '2026-09-24 13:30:00+00:00',
             '2026-09-24 14:30:00+00:00']),
        'temp_air': [20.22916603088379, 20.9375, 21.64583396911621],
        'wind_speed': [2.0, 2.0, 2.0],
        'wind_direction': [0.0, 0.0, 0.0],
        'lead_time': [12.5, 13.5, 14.5],
        'ghi_csi': [0.5723515483072679, 0.5723515483072679,
                    0.5723515483072679],
        'ghi': [85.34012971013142, 214.04410060660624, 334.11042080758943],
        'dni': [82.90832137573915, 100.88303273952344, 96.21481001087798],
        'dhi': [70.49070597469898, 175.46280618707635, 280.23928602630895],
        'ghi_clear': [149.1043921564032, 373.97313109336835,
                      583.7503572685746],
        'member': [1, 1, 1],
        'point': [0, 0, 0]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")
    ens_mean = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 12:30:00+00:00',
             '2026-09-24 13:30:00+00:00',
             '2026-09-24 14:30:00+00:00']),
        'temp_air': [20.22916603088379, 20.9375, 21.64583396911621],
        'wind_speed': [2.0, 2.0, 2.0],
        'wind_direction': [0.0, 0.0, 0.0],
        'lead_time': [12.5, 13.5, 14.5],
        'ghi_csi': [0.4352765171486868, 0.4352765171486868,
                    0.4352765171486868],
        'ghi': [64.90164050941117, 162.78172200951067, 254.09282239616678],
        'dni': [54.38486708131039, 68.40495170413656, 65.72301782349156],
        'dhi': [55.160954379419564, 136.6212120626575, 217.29418868269252],
        'ghi_clear': [149.1043921564032, 373.97313109336835,
                      583.7503572685746],
        'point': [0.0, 0.0, 0.0]
    }
    ens_mean = pd.DataFrame(ens_mean).set_index('valid_time')
    ens_mean.index = ens_mean.index.astype("datetime64[ns, UTC]")
    # check number of members
    assert len(rd['member'].unique()) == 30
    # check dataframe filtered to first member
    assert_frame_equal(rd[rd['member'] == 1], rd_test, check_dtype=False,
                       rtol=1e-4)
    # check ens mean
    assert_frame_equal(rd_mean, ens_mean, check_dtype=False, rtol=1e-4)


def test_ifs_ens_herbie():
    latitude = 33.5
    longitude = -86.8
    init_date = pd.Timestamp('2026-09-24 00:00:00+0000')
    run_length = 3
    lead_time_to_start = 12
    model = 'ifs_ens'
    rd = get_solar_forecast_ensemble(
        latitude, longitude, init_date, run_length,
        lead_time_to_start, model, attempts=2,
        priority='google')
    # calculate ensemble mean
    rd_mean = (rd.reset_index().
               set_index(['valid_time', 'member']).
               groupby('valid_time').mean())
    data = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 12:30:00+00:00',
             '2026-09-24 13:30:00+00:00',
             '2026-09-24 14:30:00+00:00']),
        'temp_air': [19.980615615844727, 20.318374633789062, 20.6561336517334],
        'wind_speed': [2.0, 2.0, 2.0],
        'wind_direction': [0.0, 0.0, 0.0],
        'lead_time': [12.5, 13.5, 14.5],
        'ghi_csi': [0.45368580233594746, 0.45368580233594746,
                    0.45368580233594746],
        'ghi': [67.64654578729154, 169.66630003218128, 264.83924920128925],
        'dni': [12.247644558019703, 31.234553751781146, 32.589060820572655],
        'dhi': [65.45291237649356, 157.72108504769972, 246.5924773408872],
        'ghi_clear': [149.1043921564032, 373.97313109336835,
                      583.7503572685746],
        'member': [1, 1, 1],
        'point': [0, 0, 0]
    }
    rd_test = pd.DataFrame(data).set_index('valid_time')
    rd_test.index = rd_test.index.astype("datetime64[ns, UTC]")
    ens_mean = {
        'valid_time': pd.DatetimeIndex(
            ['2026-09-24 12:30:00+00:00',
             '2026-09-24 13:30:00+00:00',
             '2026-09-24 14:30:00+00:00']),
        'temp_air': [19.980615615844727, 20.318374633789062, 20.6561336517334],
        'wind_speed': [2.0, 2.0, 2.0],
        'wind_direction': [0.0, 0.0, 0.0],
        'lead_time': [12.5, 13.5, 14.5],
        'ghi_csi': [0.4001616797032185, 0.4001616797032185,
                    0.4001616797032185],
        'ghi': [59.665864016433694, 149.64971630219418, 233.5945234919467],
        'dni': [26.841203587944925, 34.03071874697709, 34.70594853031559],
        'dhi': [54.85842853791204, 136.6351474135355, 214.16249602364493],
        'ghi_clear': [149.1043921564032, 373.97313109336835,
                      583.7503572685746],
        'point': [0.0, 0.0, 0.0]
    }
    ens_mean = pd.DataFrame(ens_mean).set_index('valid_time')
    ens_mean.index = ens_mean.index.astype("datetime64[ns, UTC]")
    # check number of members
    assert len(rd['member'].unique()) == 50
    # check dataframe filtered to first member
    assert_frame_equal(rd[rd['member'] == 1], rd_test, check_dtype=False,
                       rtol=1e-4)
    # check ens mean
    assert_frame_equal(rd_mean, ens_mean, check_dtype=False, rtol=1e-4)
