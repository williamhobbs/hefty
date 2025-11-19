import pandas as pd
import warnings


def model_input_formatter(init_date, run_length, lead_time_to_start=0,
                          model='gfs', resource_type='solar'):
    """
    Helper function to format model-specific inputs for Herbie.

    In the case where the user selects an invalid intitialization date, or
    combination of init date and lead time, it tries to update the init date
    and lead time to match a valid init date for the selected model, but this
    hasn't been fully tested.

    Parameters
    ----------
    init_date : pandas-parsable datetime
        Targetted initialization datetime.

    run_length : int
        Length of the forecast in hours.

    lead_time_to_start : int
        Number of hours from the init_date to the first interval in the
        forecast.

    model : {'gfs', 'ifs', 'aifs', 'hrrr', 'gefs'}
        Forecast model name, case insensitive. Default is 'gfs'.

    resource_type : {'solar, 'wind'}
        Resrouce type. Default is 'solar'.

    Returns
    -------
    date : pandas-parsable datetime
        initialization date, rounded down to the last valid date for the given
        model if needed.

    fxx_range : int or list of ints
        fxx (lead time) values.

    product : string
        model product, e.g., 'pgrb2.0p25' for 'gfs'

    search_str : string
        wgrib2-style search string for Herbie to select variables of interest.
    """

    if model == 'gfs':
        # GFS:
        # 0 to 120 by 1, 123 to 384 by 3
        # runs every 6 hours starting at 00z
        update_freq = '6h'
        # round down to last actual initialization time
        date = init_date.floor(update_freq)

        # offset in hours between selected init_date and fcast run
        init_offset = int((init_date - date).total_seconds()/3600)
        lead_time_to_start = lead_time_to_start + init_offset

        # maximum forecast horizon, update with new lead time
        fxx_max = run_length + lead_time_to_start

        # set forecast lead times
        if lead_time_to_start <= 120 and fxx_max > 120:
            fxx_max = round(fxx_max/3)*3
            fxx_range = [*range(lead_time_to_start, 120+1, 1),
                         *range(123, fxx_max + 1, 3)]
        elif lead_time_to_start > 120:
            fxx_max = round(fxx_max/3)*3
            lead_time_to_start = round(lead_time_to_start/3)*3
            fxx_range = range(lead_time_to_start, fxx_max, 3)
        else:
            fxx_range = range(lead_time_to_start, fxx_max, 1)

        # Herbie inputs
        product = 'pgrb2.0p25'
        if resource_type == 'solar':
            search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
        elif resource_type == 'wind':
            search_str = (
                '[UV]GRD:10 m above|[UV]GRD:80 m above|'
                '[UV]GRD:100 m above|:TMP:2 m above|PRES:surface|'
                ':TMP:80 m above|PRES:80 m above'
            )

    elif model == 'gefs':
        # GEFS:
        # 0.5 deg:
        #   0 to 384 by 3, 390 to 840 by 6 for 00z cycle only
        # 0.25 deg:
        #   0 to 240 by 3
        # runs every 6 hours starting at 00z
        update_freq = '6h'
        # round down to last actual initialization time
        date = init_date.floor(update_freq)

        # offset in hours between selected init_date and fcast run
        init_offset = int((init_date - date).total_seconds()/3600)
        lead_time_to_start = lead_time_to_start + init_offset

        # maximum forecast horizon, update with new lead time
        fxx_max = run_length + lead_time_to_start

        # set forecast lead times
        fxx_range = range(lead_time_to_start, fxx_max + 1, 3)

        # Herbie inputs
        if resource_type == 'solar':
            # solar radiation is not available for f00 (lead_time_to_start=0)
            # adjust accordingly
            if lead_time_to_start < 3:
                lead_time_to_start = 3
                warnings.warn(
                        ("You have specified a lead_time_to_start less"
                         "than 3 h. GHI in GEFS is only available "
                         "starting at F03. The lead_time_to_start has been"
                         "changed to 3 h."))

            if fxx_max <= 240:
                product = 'atmos.25'  # 0.25 deg, 'pgrb2.0p25'
                search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
            else:
                product = 'atmos.5'  # 0.5 deg, 'pgrb2.0p5'
                search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
        elif resource_type == 'wind':
            product = 'atmos.5b'  # 0.5 deg, 'pgrb2.0p5
            search_str = (
                '[UV]GRD:80 m above|[UV]GRD:100 m above|'
                ':TMP:80 m above|PRES:80 m above'
            )

    elif model == 'ifs':
        # From https://www.ecmwf.int/en/forecasts/datasets/open-data
        # For times 00z &12z: 0 to 144 by 3, 150 to 240 by 6.
        # For times 06z & 18z: 0 to 90 by 3.
        # From:
        # https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts+from+IFS+and+AIFS
        # Product "oper" runs 00z, 12z, 0h to 144h by 3h, 144h to 240h by 6h
        # Product "scda" runs 06z, 18z, 0h to 90h by 3h
        # **BUT**, see https://github.com/blaylockbk/Herbie/discussions/421
        # Starting 2024-11-12 06:00, 'scda' runs to 144h by 3h

        # round to last 6 hours to start
        date = init_date.floor('6h')
        init_offset = int((init_date - date).total_seconds()/3600)
        lead_time_to_start = lead_time_to_start + init_offset
        fxx_max = run_length + lead_time_to_start

        # pick init time based on forecast max lead time:
        # check if 'scda' product is ideal
        if init_date.hour == 6 or init_date.hour == 18:
            if init_date >= pd.to_datetime('2024-11-12 06:00'):
                scda_fxx_max = 144
            else:
                scda_fxx_max = 90
            if fxx_max > scda_fxx_max:  # forecast beyond scda
                update_freq = '12h'  # must use 'oper' runs
                warnings.warn(
                    ("You have specified an init_date which would have mapped "
                     "to a 06z or 18z. Those runs the IFS 'scda' product, and "
                     "'scda' only goes out 144 hours (90h prior to 2024-11-12)"
                     ". You will get forecasts from the 'oper' run 6 hours "
                     "earlier, instead."))
            else:
                update_freq = '6h'  # can use 'oper' or 'scda'
        else:
            update_freq = '6h'  # can use 'oper' or 'scda'
        # round down to last actual initialization time
        date = init_date.floor(update_freq)

        # offset in hours between selected init_date and fcast run
        init_offset = int((init_date - date).total_seconds()/3600)
        lead_time_to_start = lead_time_to_start + init_offset
        if lead_time_to_start > 141:
            run_length = max(run_length, 6)  # make sure it's long enough
        fxx_max = run_length + lead_time_to_start  # update this

        # set forecast intervals
        if lead_time_to_start <= 144 and fxx_max > 144:
            lead_time_to_start = round(lead_time_to_start/3)*3
            fxx_max = round(fxx_max/6)*6
            # make sure it goes to at least the next interval
            fxx_max = max(fxx_max, 150)
            fxx_range = [*range(lead_time_to_start, 145, 3),
                         *range(150, fxx_max + 1, 6)]
        elif lead_time_to_start > 144:
            lead_time_to_start = round(lead_time_to_start/6)*6
            fxx_max = round(fxx_max/6)*6
            fxx_range = range(lead_time_to_start, fxx_max + 1, 6)
        else:
            lead_time_to_start = round(lead_time_to_start/3)*3
            fxx_max = round(fxx_max/3)*3
            fxx_range = range(lead_time_to_start, fxx_max + 1, 3)

        # Herbie inputs
        if date.hour == 6 or date.hour == 18:
            product = 'scda'
        else:
            product = 'oper'

        if resource_type == 'solar':
            search_str = ':ssrd|10[uv]|2t:sfc'
        elif resource_type == 'wind':
            search_str = ':10[uv]|:100[uv]|:2t:sfc|:sp:'

    elif model == 'aifs':
        # From https://www.ecmwf.int/en/forecasts/datasets/set-ix,
        # https://www.ecmwf.int/en/forecasts/dataset/set-x
        # 4 forecast runs per day (00/06/12/18)
        # 6 hourly steps to 360 (15 days)

        # round to last 6 hours to start
        date = init_date.floor('6h')
        init_offset = int((init_date - date).total_seconds()/3600)
        lead_time_to_start = lead_time_to_start + init_offset
        fxx_max = run_length + lead_time_to_start

        update_freq = '6h'
        # round down to last actual initialization time
        date = init_date.floor(update_freq)

        # offset in hours between selected init_date and fcast run
        init_offset = int((init_date - date).total_seconds()/3600)
        lead_time_to_start = lead_time_to_start + init_offset
        if lead_time_to_start > 141:
            run_length = max(run_length, 6)  # make sure it's long enough
        fxx_max = run_length + lead_time_to_start  # update this

        # set forecast intervals
        fxx_range = range(lead_time_to_start, fxx_max + 1, 6)

        # Herbie inputs
        product = 'oper'  # deterministic

        if resource_type == 'solar':
            search_str = ':ssrd|10[uv]|2t:sfc'
        elif resource_type == 'wind':
            search_str = ':10[uv]|:100[uv]|:2t:sfc|:sp:'

    elif model == 'hrrr':
        # maximum forecast horizon
        fxx_max = run_length + lead_time_to_start
        product = 'sfc'

        if resource_type == 'solar':
            search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
        elif resource_type == 'wind':
            search_str = (
                '[UV]GRD:10 m above|[UV]GRD:80 m above|'
                ':TMP:2 m above|PRES:surface'
            )

        update_freq = '1h'

        # round down to last actual initialization time
        date = init_date.floor(update_freq)

        fxx_range = range(lead_time_to_start, fxx_max, 1)

    return date, fxx_range, product, search_str
