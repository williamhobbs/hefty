
import pytest
# from .conftest import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose
import pandas as pd

from hefty.utilities import model_input_formatter

def test_gfs_solar():
    init_date = '2026-09-20 00:00'
    run_length = 12
    lead_time_to_start = 0
    model = 'gfs'
    resource_type = 'solar'
    with pytest.warns(
         UserWarning,
        match='You have specified a lead_time_to_start less than 1 h'
    ) as warn:
        date, fxx_range, product, search_str = model_input_formatter(
            init_date, run_length, lead_time_to_start,
            model, resource_type)
        assert date == pd.Timestamp('2026-09-20 00:00:00')
        assert fxx_range == range(1, 13, 1)
        assert product == 'pgrb2.0p25'
        assert search_str == 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'


def test_gfs_wind():
    init_date = '2026-09-20 00:00'
    run_length = 2
    lead_time_to_start = 120
    model = 'gfs'
    resource_type = 'wind'
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start,
        model, resource_type)
    assert fxx_range == [120, 123]
    assert product == 'pgrb2.0p25'
    assert search_str == (
                     '[UV]GRD:10 m above|[UV]GRD:80 m above|'
                     '[UV]GRD:100 m above|:TMP:2 m above|PRES:surface|'
                     ':TMP:80 m above|PRES:80 m above'
                 )


def test_gfs_lead_to_start_over_120():
    init_date = '2026-09-20 00:00'
    run_length = 5
    lead_time_to_start = 124
    model = 'gfs'
    resource_type = 'solar'
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start,
        model, resource_type)
    assert fxx_range == range(123, 130, 3)


def test_gefs_fxx_less_than_3():
    init_date = '2026-09-20 00:00'
    run_length = 14
    lead_time_to_start = 2
    model = 'gefs'
    resource_type = 'solar'
    full_ens = False
    get_ens_temp = False
    get_ens_wind = False
    member = None

    with pytest.warns(
        UserWarning,
        match='You have specified a lead_time_to_start less than 3') as warn:
            date, fxx_range, product, search_str = model_input_formatter(
                init_date, run_length, lead_time_to_start,
                model, resource_type,
                full_ens, get_ens_temp,
                get_ens_wind, member
                )

            assert date == pd.Timestamp('2026-09-20 00:00:00')
            assert fxx_range == range(3, 17, 3)
            assert product == 'atmos.25'
            assert search_str == 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'


def test_gefs_fxx_max_over_240():  
    init_date = '2026-09-20 00:00'
    run_length = 250
    lead_time_to_start = 3
    model = 'gefs'
    resource_type = 'solar'
    date, fxx_range, product, search_str = model_input_formatter(
         init_date, run_length, lead_time_to_start,
         model, resource_type)
    assert product == 'atmos.5'


def test_gefs_full_ens():  
    init_date = '2026-09-20 00:00'
    run_length = 24
    lead_time_to_start = 3
    model = 'gefs'
    resource_type = 'solar'
    full_ens = True
    date, fxx_range, product, search_str = model_input_formatter(
         init_date, run_length, lead_time_to_start,
         model, resource_type, full_ens)
    assert search_str == 'DSWRF'


def test_gefs_full_ens_temp():
    init_date = '2026-09-20 00:00'
    run_length = 24
    lead_time_to_start = 3
    model = 'gefs'
    resource_type = 'solar'
    full_ens = True
    get_ens_temp = True
    date, fxx_range, product, search_str = model_input_formatter(
         init_date, run_length, lead_time_to_start,
         model, resource_type, full_ens, get_ens_temp)
    assert search_str == 'DSWRF|:TMP:2 m above'


def test_gefs_full_ens_temp_wind():  
    init_date = '2026-09-20 00:00'
    run_length = 24
    lead_time_to_start = 3
    model = 'gefs'
    resource_type = 'solar'
    full_ens = True
    get_ens_temp = True
    get_ens_wind = True
    date, fxx_range, product, search_str = model_input_formatter(
         init_date, run_length, lead_time_to_start,
         model, resource_type, full_ens, get_ens_temp,
         get_ens_wind)
    assert search_str == 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'


def test_gefs_wind():
    init_date = '2026-09-20 00:00'
    run_length = 24
    lead_time_to_start = 250
    model = 'gefs'
    resource_type = 'wind'
    date, fxx_range, product, search_str = model_input_formatter(
         init_date, run_length, lead_time_to_start,
         model, resource_type)

    assert product == 'atmos.5b'
    assert search_str == (
                    '[UV]GRD:80 m above|[UV]GRD:100 m above|'
                    ':TMP:80 m above|PRES:80 m above'
                )
    assert fxx_range == range(252, 277, 6)


@pytest.mark.parametrize(
    "model_in,resource_type_in,product_out,search_str_out",
    [
        ("ifs", "solar", "oper", ":ssrd|:2t|:10[uv]"),
        ("aifs", "solar", "oper", ":ssrd|:2t|:10[uv]"),
        ("ifs", "wind", "oper", ":10[uv]|:100[uv]|:2t|:sp"),
        ("aifs", "wind", "oper", ":10[uv]|:100[uv]|:2t|:sp"),
    ],
)
def test_ifs(model_in, resource_type_in, product_out, search_str_out):
    init_date = '2026-09-20 00:00'
    run_length = 24
    lead_time_to_start = 0
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start,
        model_in, resource_type_in)
    assert product == product_out
    assert search_str == search_str_out

def test_ifs_pre_50r1():
    init_date = '2026-05-11 06:00'  # before 50r1 updgrade on 05-12
    run_length = 24
    lead_time_to_start = 0
    model = 'ifs'
    resource_type = 'solar'
    date, fxx_range, product1, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start,
        model, resource_type)
    model = 'ifs_ens'
    date, fxx_range, product2, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start,
        model, resource_type, member=0)
    assert product1 == 'scda'
    assert product2 == 'enfo'
