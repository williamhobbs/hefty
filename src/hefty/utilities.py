import pandas as pd
import warnings
import math
import numpy as np
from herbie import Herbie, FastHerbie
import xarray as xr
import time


def get_fcast_definition(model='gfs'):
    """
    Function that returns a forecast definition dictionary for the selected
    model.

    Parameters
    ----------

    model : {'gfs', 'ifs', 'aifs', 'hrrr', 'gefs', 'ifs_ens', 'aifs_ens'}
        Forecast model name. Default is 'gfs'.

    Returns
    -------
    fcast_definition : dictionary
        A dictionary with information about a forecast model, including the
        'Name' (e.g., 'gfs'), a list of one or more schedule dictionaries,
        'Forecast Schedule Dictionary', and a list of start dates that
        correspond with those schedules, 'Start Date of Schedule'.

    Notes
    -----
    ``delay_intercept`` and ``delay_slope`` values are based on this gist,
    https://gist.github.com/williamhobbs/9585ff5d1248ab5de4d9e8665d7c8ea6,
    and https://dynamical.org/status/, along with
    https://confluence.ecmwf.int/display/DAC/Dissemination+schedule and 
    https://confluence.ecmwf.int/display/CKB/CAMS%3A+Global+atmospheric+composition+forecast+data+documentation#heading-DataavailabilityHHMM.

    The ``'Forecast Schedule Dictionary'`` within ``fcast_definition``
    contains a list of one or more schedule dictionaries. The values in each
    dictionary are lists, where the elements in each list corresond to
    eachother. The keys are:

    - ``'start_date'``: date string, e.g., ``'2023-01-18 00:00'``, when the
    schedule was first available.
    - ``'end_hour'``: integer, e.g., ``144``, for the last forecast hour in
    the schedule.
    - ``'interval'``: number of hours in each interval of the schedule, e.g.,
    ``3`` for a schedule with 3h steps.
    - ``'first_cycle'``: integer hour of the day for the first cycle when the
    schedule variation of the model runs, e.g., ``0`` for a model that first
    initializes as 00z.
    - ``'update_period'``: number of hours between cycles for the schdeule,
    e.g., ``12`` for 12 hour updates.
    - ``'delay_intercept'``: intercept of a fit between forecast delivery
    delay in minutes and the forecast hour, e.g., ``515`` minutes. 
    - ``'delay_slope'``: slope of a fit between forecast delivery delay in
    minutes and the forecast hour, e.g., ``0.02`` for a model schedule that
    delivers one  forecast hours per 0.02 minues (50 hours per minute).
    - ``'product'``: string representing the model product for the schedule,
    e.g., ``'oper'`` for ECMWF IFS 'oper' schedule.

    """

    # ===========================================================
    # Forecast Definitions
    # ===========================================================
    # IFS
    # first available 2023-01-18 (https://herbie.readthedocs.io/en/stable/gallery/ecmwf_models/ecmwf.html#Data-Availability)
    fcast_sched_dict_ifs_1 = {
        'start_date': ['2023-01-18 00:00',
                       '2023-01-18 00:00',
                       '2023-01-18 06:00'],
        'start_hour': [0, 150, 0],
        'end_hour': [144, 240, 90],
        'interval': [3, 6, 3],
        'first_cycle': [0, 0, 6],
        'update_period': [12, 12, 12],
        'delay_intercept': [515, 515, 450],
        'delay_slope': [0.006, 0.006, 0.006],
        'product': ['oper', 'oper', 'scda'],
    }

    # Nov 2024 extended 'oper' and 'scda' horizons
    # https://github.com/blaylockbk/Herbie/discussions/421
    fcast_sched_dict_ifs_2 = {
        'start_date': ['2024-11-12 12:00',
                       '2024-11-12 12:00',
                       '2024-11-12 06:00'],
        'start_hour': [0, 150, 0],
        'end_hour': [144, 360, 144],
        'interval': [3, 6, 3],
        'first_cycle': [0, 0, 6],
        'update_period': [12, 12, 12],
        'delay_intercept': [515, 515, 450],
        'delay_slope': [0.006, 0.006, 0.006],
        'product': ['oper', 'oper', 'scda'],
    }

    # Approx Oct 1 2025, removed 1hr extra delay in releasing files
    # Start date is just a guess, needs confirmation. Descriptoin changed
    # sometime between Nov 6 [1] and Nov 22 [2] 2025, but the changes to the
    # ECMWF open data website are often delayed. Maybe it corresponded with
    # this press release [3]?
    #
    # [1] https://web.archive.org/web/20251106132450/https://www.ecmwf.int/en/forecasts/datasets/open-data
    # [2] https://web.archive.org/web/20251122204201/https://www.ecmwf.int/en/forecasts/datasets/open-data
    # [3] https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-makes-its-entire-real-time-catalogue-open-all
    fcast_sched_dict_ifs_3 = {
        **fcast_sched_dict_ifs_2,
        'start_date': ['2025-10-01 12:00',
                       '2025-10-01 12:00',
                       '2025-10-01 06:00'],
        'delay_intercept': [455, 455, 390],
    }

    fcast_definition_ifs = {
        'Name': 'ifs',
        'Start Date of Schedule': ['2023-01-18 00:00',
                                   '2024-11-12 06:00',
                                   '2025-10-01 06:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_ifs_1,
                                         fcast_sched_dict_ifs_2,
                                         fcast_sched_dict_ifs_3],
    }

    # IFS Ensemble
    # IFS ens does not have ssrd until sometime March 2024. '2024-03-12 12:00'
    # was the first init_date used in https://github.com/williamhobbs/PVSC-2025-daily-energy-forecaster,
    # so start there for now.
    # delays based on https://dynamical.org/status/ as of 2026-04-24.
    # From https://www.ecmwf.int/en/forecasts/datasets/open-data,
    # For times 00z &12z: 0 to 144 by 3, 150 to 360 by 6.
    # For times 06z & 18z: 0 to 144 by 3.
    fcast_sched_dict_ifs_ens_1 = {
        'start_date': ['2024-03-10 12:00',
                       '2024-03-10 12:00',
                       '2024-03-10 18:00'],  # https://herbie.readthedocs.io/en/stable/gallery/ecmwf_models/ecmwf.html#Data-Availability
        'start_hour': [0, 150, 0],
        'end_hour': [144, 360, 144],
        'interval': [3, 6, 3],
        'first_cycle': [0, 0, 6],
        'update_period': [12, 12, 12],
        'delay_intercept': [520, 520, 484],
        'delay_slope': [0.02, 0.02, 0.03],
        'product': ['enfo', 'enfo', 'enfo'],
    }

    # Approx Oct 1 2025, removed 1hr extra delay in releasing files
    # Start date is just a guess, needs confirmation. Descriptoin changed
    # sometime between Nov 6 [1] and Nov 22 [2] 2025, but the changes to the
    # ECMWF open data website are often delayed. Maybe it corresponded with
    # this press release [3]?
    #
    # [1] https://web.archive.org/web/20251106132450/https://www.ecmwf.int/en/forecasts/datasets/open-data
    # [2] https://web.archive.org/web/20251122204201/https://www.ecmwf.int/en/forecasts/datasets/open-data
    # [3] https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-makes-its-entire-real-time-catalogue-open-all
    fcast_sched_dict_ifs_ens_2 = {
        **fcast_sched_dict_ifs_2,
        'start_date': ['2025-10-01 12:00',
                       '2025-10-01 12:00',
                       '2025-10-01 06:00'],
        'delay_intercept': [460, 460, 424],
    }

    fcast_definition_ifs_ens = {
        'Name': 'ifs_ens',
        'Start Date of Schedule': ['2024-03-10 12:00',
                                   '2025-10-01 12:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_ifs_ens_1,
                                         fcast_sched_dict_ifs_ens_2],
    }

    # AIFS
    # First available 2024-02-01 (https://herbie.readthedocs.io/en/stable/gallery/ecmwf_models/ecmwf.html)
    fcast_sched_dict_aifs = {
        'start_date': ['2024-02-01 00:00'],
        'start_hour': [0],
        'end_hour': [360],
        'interval': [6],
        'first_cycle': [0],
        'update_period': [6],
        'delay_intercept': [339],
        'delay_slope': [0.008],
        'product': ['aifs'],
    }

    fcast_definition_aifs = {
        'Name': 'aifs',
        'Start Date of Schedule': ['2024-02-01 00:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_aifs],
    }

    # AIFS ENS
    # First available 2025-07-2, added one day, as I seem to recall some
    # variables were missing for a few days (https://herbie.readthedocs.io/en/stable/gallery/ecmwf_models/ecmwf.html)
    # Schedule is unverified, based on
    # https://confluence.ecmwf.int/display/DAC/Dissemination+schedule
    fcast_sched_dict_aifs_ens = {
        'start_date': ['2025-07-03 00:00'],
        'start_hour': [0, 0],
        'end_hour': [360, 144],
        'interval': [6, 6],
        'first_cycle': [0, 6],
        'update_period': [12, 12],
        'delay_intercept': [400, 400],
        'delay_slope': [0.125, 0.125],
        'product': ['enfo', 'enfo'],
    }

    fcast_definition_aifs_ens = {
        'Name': 'aifs_ens',
        'Start Date of Schedule': ['2025-07-03 00:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_aifs_ens],
    }

    # CAMS version of IFS
    # Runs 00z and 12z, 0-120h by 1h for single-level parameters
    # 00 UTC data available by 10:00 UTC
    # 12 UTC data available by 22:00 UTC
    fcast_sched_dict_cams = {
        'start_date': ['2016-01-01 00:00'],  # data starts sometime in 2015
        'start_hour': [0],
        'end_hour': [120],
        'interval': [1],
        'first_cycle': [0],
        'update_period': [12],
        'delay_intercept': [600],
        'delay_slope': [0.001],
        'product': ['cams'],
    }

    fcast_definition_cams = {
        'Name': 'cams',
        'Start Date of Schedule': ['2016-01-01 00:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_cams],
    }

    # HRRR (hourly)
    # HRRR v4 started Dec 3, 2020, https://rapidrefresh.noaa.gov/hrrr/
    fcast_sched_dict_hrrr = {
        'start_date': ['2020-12-03 01:00',
                       '2020-12-03 00:00'],
        'start_hour': [0, 0],
        'end_hour': [18, 48],
        'interval': [1, 1],
        'first_cycle': [0, 0],
        'update_period': [1, 6],
        'delay_intercept': [61, 63],
        'delay_slope': [1.862, 1.125],
        'product': ['18h', '48h'],
    }

    fcast_definition_hrrr = {
        'Name': 'hrrr',
        'Start Date of Schedule': ['2020-12-03 00:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_hrrr],
    }

    # GFS
    # Need to use GFSv15.1 and newer. Previous versions had a bug in solar
    # zenith angle (pg. 92, https://www.emc.ncep.noaa.gov/emc/docs/FV3GFS_OD_Briefs_10-01-18_4-1-2019.pdf),
    # and v15 had some other radiation bug, corrected 2018-09-17 18Z (pg. 34, https://www.weather.gov/media/sti/nggps/NGGPS/EMC%20MEG%20Evaluation%20of%20GFSv15_Manikin_SIP%20Meeting_20190514.pdf)
    # Maybe the same bug? See slide 115 (pg103) https://www.emc.ncep.noaa.gov/emc/docs/FV3GFS_OD_Briefs_10-01-18_4-1-2019.pdf
    # GFSv15.1 implemented June 12, 2019 (https://doi.org/10.1175/WAF-D-23-0094.1)
    # see also https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs/documentation.php,
    # https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs/implementations.php
    fcast_sched_dict_gfs = {
        'start_date': ['2019-06-13 00:00',
                       '2019-06-13 00:00'],
        'start_hour': [0, 123],
        'end_hour': [120, 384],
        'interval': [1, 3],
        'first_cycle': [0, 0],
        'update_period': [6, 6],
        'delay_intercept': [238, 238],
        'delay_slope': [0.263, 0.263],
        'product': ['pgrb2.0p25', 'pgrb2.0p25'],
    }

    fcast_definition_gfs = {
        'Name': 'gfs',
        'Start Date of Schedule': ['2019-06-13 00:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_gfs],
    }

    # GEFS
    # Need to use GEFSv12 and newer to correspond to GFSv15.1 and newer (see
    # comments on GFS above).
    # GEFSv12 is based on GFSv15.1 (https://journals.ametsoc.org/view/journals/mwre/150/3/MWR-D-21-0245.1.xml)
    # Implemented 2020-09-23 (https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php)
    fcast_sched_dict_gefs = {
        'start_date': ['2020-09-24 01:00',
                       '2020-09-24 00:00'],
        'start_hour': [0, 390],
        'end_hour': [384, 840],
        'interval': [3, 6],
        'first_cycle': [0, 0],
        'update_period': [6, 24],
        'delay_intercept': [235, 265],
        'delay_slope': [0.429, 0.332],
        'product': ['3-hourly', 'extended'],  # needs update
    }

    fcast_definition_gefs = {
        'Name': 'gefs',
        'Start Date of Schedule': ['2020-09-24 01:00'],
        'Forecast Schedule Dictionary': [fcast_sched_dict_gefs],
    }

    # ===========================================================

    if model == 'gfs':
        fcast_definition = fcast_definition_gfs

    elif model == 'gefs':
        fcast_definition = fcast_definition_gefs

    elif model == 'hrrr':
        fcast_definition = fcast_definition_hrrr

    elif model == 'ifs':
        # IFS Single, ENS is handled separately
        fcast_definition = fcast_definition_ifs

    elif model == 'aifs':
        # AIFS single, ENS is handled separately
        fcast_definition = fcast_definition_aifs

    elif model == 'ifs_ens':
        fcast_definition = fcast_definition_ifs_ens

    elif model == 'aifs_ens':
        fcast_definition = fcast_definition_aifs_ens

    elif model == 'cams':
        fcast_definition = fcast_definition_cams

    return fcast_definition


def adjust_forecast_datetimes(available_date, run_length_needed,
                              lead_time_to_start_needed, model='gfs'):
    """
    Helper function to adjust datetimes for use in
    `hefty.utilities.model_input_formatter`.

    [longer description]

    Parameters
    ----------
    available_date : pandas-parsable datetime
        Datetime at which the forecast outputs need to be available, assumed
        to be in UTC unless a timezone-aware value is provided.

    run_length_needed : int
        Length of the forecast that is needed, in hours, starting from the
        ``lead_time_to_start``.

    lead_time_to_start_needed : int, default 0
        Number of hours from the ``available_date`` (after rounding down to the
        hour) to the targetted first interval in the forecast.

    model : {'gfs', 'ifs', 'aifs', 'hrrr', 'gefs', 'ifs_ens', 'aifs_ens'}
        Forecast model name. Default is 'gfs'.

    Returns
    -------
    init_date : pandas-parsable datetime
        Model initialization datetime, adjusted to be the first time *before*
        the specified ``available_date`` for which model outputs are estimated
        to be available. Accounts for standard model initialization times and
        estimated delays between initilization and output availability
        time for the selected model. For example, GEFS has initilization
        times of 00:00, 06:00, 12:00, and 18:00 UTC, and model outputs are
        only available 3.5 to 6 hours after initialization. The function
        :py:func:`hefty.utilities.adjust_forecast_datetimes` can help with
        determining correct times to use.

    run_length : int
        Length of the forecast in hours, relative to the returned
        ``lead_time_to_start``.

    lead_time_to_start : int
        Number of hours from the ``init_date`` to the first interval needed in
        the forecast.
    """

    # convert to pandas datetime
    available_date = pd.to_datetime(available_date)

    # issue tz warning if available_date.tzinfo is None
    if available_date.tzinfo is None:
        available_date = available_date.tz_localize('UTC')
        warnings.warn(
            ("You have provided a timezone-naive available_date. "
             "It has been converted to UTC. If you did not intend "
             "to provide a time in UTC, please make available_date "
             "timezone-aware or convert it to UTC."))

    # round down to last hour
    available_date_floor = available_date.floor('1h')

    fxx_max_requested = run_length_needed + lead_time_to_start_needed

    fcast_definition = get_fcast_definition(model=model)

    # add an extra 2 minute sof delay for hrrr, 15 minutes for everything else
    if model == 'hrrr':
        delay_buffer = 2
    else:
        delay_buffer = 15

    # Find appropriate schedule (latest start date before available_date)
    sched_start_dates = [pd.Timestamp(x, tz='UTC') for x in
                         fcast_definition['Start Date of Schedule']]
    sched_start_date = max(date for date in sched_start_dates if
                           date < available_date)
    idx = sched_start_dates.index(sched_start_date)
    sched = fcast_definition['Forecast Schedule Dictionary'][idx]

    # schedule reference info
    # max possible forecast hour, delay
    max_model_fxx = max(sched['end_hour'])
    idx = sched['end_hour'].index(max_model_fxx)
    delay_intercept = sched['delay_intercept'][idx]
    delay_slope = sched['delay_slope'][idx]
    max_delay_minutes = (
        delay_intercept +
        (delay_slope * max_model_fxx) +
        delay_buffer
    )
    # max possible delay in hours, rounded up (ceiling)
    max_delay = math.ceil(max_delay_minutes / 60)

    max_period = max(sched['update_period'])

    # check if no schedules could go out far enough
    if fxx_max_requested > (max_model_fxx - max_delay):
        raise ValueError('The requested forecast goes too far out '
                         'from the available_date after accounting '
                         'for delays. Try a smaller run_length_needed, '
                         'lead_time_to_start_needed '
                         'or both.'
                         )

    # build sorted list of unique cycle times
    cycle_list = []
    for variation in range(len(sched['start_date'])):
        first_cycle = sched['first_cycle'][variation]
        update_period = sched['update_period'][variation]
        cycle_list += list(range(first_cycle, 24, update_period))
    cycle_list = list(set(cycle_list))  # remove duplicates w/ set()
    cycle_list = sorted(cycle_list)  # sort

    # create a list of "lookbacks", hours before available_date,
    # to iterate through.
    # hours between available_date and each hour in cycle_list
    lookbacks = [(available_date.hour - x) % 24 for x in cycle_list]
    lookbacks = sorted(lookbacks)
    lookback_start = min(lookbacks)  # the most recent lookback
    # if our available_date is not top of the hour, let's keep track of the
    # hour remainder to use later
    rem = (available_date - available_date.floor('1h')).total_seconds() / 3600
    # we want the lookbacks to cover at least the max of 24 hours or
    # (max_delay + max_period), so extend the list of lookbacks by integer
    # days, then trim it
    days_to_add = 1 + (max_delay + max_period) // 24
    list_of_days = list(range(days_to_add + 1))
    lookbacks = sum([[x + 24*y for x in lookbacks] for y in list_of_days], [])
    max_lookback = (24*days_to_add + lookback_start)
    lookbacks = [x for x in lookbacks if x <= max_lookback]

    # cycles that correspond to the lookbacks
    lookback_cycles = [(available_date.hour - x) % 24 for x in lookbacks]

    # check options
    found_match_length = False
    found_match_delay = False
    for i in range(len(lookbacks)):
        lookback = lookbacks[i]
        fxx_max = fxx_max_requested + lookback
        lead_time_to_start = lead_time_to_start_needed + lookback
        if (fxx_max > (max_model_fxx - max_delay)):
            raise ValueError('The requested forecast goes too far out '
                             'from the available_date after accounting '
                             'for delays. Try a smaller run_length_needed, '
                             'lead_time_to_start_needed '
                             'or both.'
                             )
        cycle = lookback_cycles[i]
        init_date = (available_date_floor - pd.Timedelta(hours=lookback))

        for variation in range(len(sched['start_date'])):
            first_cycle = sched['first_cycle'][variation]
            update_period = sched['update_period'][variation]
            start_hour = sched['start_hour'][variation]
            end_hour = sched['end_hour'][variation]
            interval = sched['interval'][variation]
            delay_intercept = sched['delay_intercept'][variation]
            delay_slope = sched['delay_slope'][variation]
            # list of cycles that this variation represents
            variation_cycles = list(range(first_cycle, 24, update_period))
            if cycle in variation_cycles:
                # if we aren't in the last list in the forecast schedule
                # variation
                if variation < len(sched['start_date']) - 1:
                    # start point of the next forecast schedule variation
                    next_start = sched['start_hour'][variation + 1]
                    # if the desired lead time falls between the end of this
                    # schdeule and the start of the next
                    if ((lead_time_to_start > end_hour) and
                            (lead_time_to_start < next_start)):
                        # then set lead time to stop of the current sched list
                        # ("round" down)
                        lead_time_to_start = end_hour
                    # if the desired fxx_max falls between schedules
                    if fxx_max > end_hour and fxx_max < next_start:
                        # then set it to the next start ("round" up)
                        fxx_max = next_start
                if ((lead_time_to_start >= start_hour) and
                        (lead_time_to_start <= end_hour)):
                    # round lead_time_to_start down
                    lead_time_to_start = (
                        interval * math.floor(lead_time_to_start/interval)
                        )
                if fxx_max >= start_hour and fxx_max <= end_hour:
                    # round fxx_max up
                    fxx_max = interval * math.ceil(fxx_max/interval)
                    found_match_length = True
                if fxx_max <= end_hour:
                    delay_minutes = (
                        delay_intercept +
                        (delay_slope * fxx_max) +
                        delay_buffer
                    )

                    delay = delay_minutes / 60

                    if delay <= lookback + rem:
                        found_match_delay = True
                        break
        else:
            continue
        break
    if found_match_delay & found_match_length:
        run_length = fxx_max - lead_time_to_start
    else:
        raise ValueError('Could not find a compatible init_date. Maybe '
                         'try a smaller run_length_needed, '
                         'lead_time_to_start_needed, or both.'
                         )

    return init_date, run_length, lead_time_to_start


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
        Model initialization datetime. Must be a valid inititialization time
        for the selected model and the model outputs must be available at the
        time the function is run. For example, GEFS has initilization times of
        00:00, 06:00, 12:00, and 18:00 UTC, and model outputs are only
        available 3.5 to 6 hours after initialization. The function
        :py:func:`hefty.utilities.adjust_forecast_datetimes` can help with
        determining correct times to use.

    run_length : int
        Length of the forecast in hours.

    lead_time_to_start : int, default 0
        Number of hours from the init_date to the first interval in the
        forecast.

    model : {'gfs', 'ifs', 'aifs', 'hrrr', 'gefs', 'ifs_ens', 'aifs_ens'}
        Forecast model name. Default is 'gfs'.

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
        # update_freq = '6h'
        # # round down to last actual initialization time
        # date = init_date.floor(update_freq)

        # # offset in hours between selected init_date and fcast run
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset

        # maximum forecast horizon, update with new lead time
        fxx_max = run_length + lead_time_to_start

        # Herbie inputs
        product = 'pgrb2.0p25'
        if resource_type == 'solar':
            search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
            # solar radiation is not available for f00 (lead_time_to_start=0)
            # adjust accordingly
            if lead_time_to_start < 1:
                lead_time_to_start = 1
                warnings.warn(
                        ("You have specified a lead_time_to_start less "
                         "than 1 h. GHI in GFS is only available "
                         "starting at F01. The lead_time_to_start has been "
                         "changed to 1 h."))
        elif resource_type == 'wind':
            search_str = (
                '[UV]GRD:10 m above|[UV]GRD:80 m above|'
                '[UV]GRD:100 m above|:TMP:2 m above|PRES:surface|'
                ':TMP:80 m above|PRES:80 m above'
            )

        # set forecast lead times
        if lead_time_to_start <= 120 and fxx_max > 120:
            fxx_max = round(fxx_max/3)*3
            fxx_range = [*range(lead_time_to_start, 120+1, 1),
                         *range(123, fxx_max + 1, 3)]
        elif lead_time_to_start > 120:
            fxx_max = round(fxx_max/3)*3
            lead_time_to_start = round(lead_time_to_start/3)*3
            fxx_range = range(lead_time_to_start, fxx_max + 1, 3)
        else:
            fxx_range = range(lead_time_to_start, fxx_max + 1, 1)

    elif model == 'gefs':
        # GEFS:
        # 0.5 deg:
        #   0 to 384 by 3, 390 to 840 by 6 for 00z cycle only
        # 0.25 deg:
        #   0 to 240 by 3
        # runs every 6 hours starting at 00z
        # update_freq = '6h'
        # # round down to last actual initialization time
        # date = init_date.floor(update_freq)

        # # offset in hours between selected init_date and fcast run
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset

        # maximum forecast horizon, update with new lead time
        fxx_max = run_length + lead_time_to_start

        # Herbie inputs
        if resource_type == 'solar':
            # solar radiation is not available for f00 (lead_time_to_start=0)
            # adjust accordingly
            if lead_time_to_start < 3:
                lead_time_to_start = 3
                warnings.warn(
                        ("You have specified a lead_time_to_start less "
                         "than 3 h. GHI in GEFS is only available "
                         "starting at F03. The lead_time_to_start has been "
                         "changed to 3 h."))

            if fxx_max <= 240:
                product = 'atmos.25'  # 0.25 deg, 'pgrb2a.0p25'
                search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
            else:
                product = 'atmos.5'  # 0.5 deg, 'pgrb2a.0p5'
                search_str = 'DSWRF|:TMP:2 m above|[UV]GRD:10 m above'
        elif resource_type == 'wind':
            # 2m temp and 10m wind are in pgrb2a, but 80 and 100m are in
            # pgrb2b (secondary parameters), so we will just get 80 and 100
            # to speed things up. Could have a second query for 2m and 10m
            # if needed
            product = 'atmos.5b'  # 0.5 deg, 'pgrb2b.0p5
            search_str = (
                '[UV]GRD:80 m above|[UV]GRD:100 m above|'
                ':TMP:80 m above|PRES:80 m above'
            )

        # set forecast lead times
        fxx_range = range(lead_time_to_start, fxx_max + 1, 3)

    elif model == 'ifs' or model == 'ifs_ens':
        # From https://www.ecmwf.int/en/forecasts/datasets/open-data
        # For times 00z &12z: 0 to 144 by 3, 150 to 360 by 6.
        # For times 06z & 18z: 0 to 144 by 3.
        # From:
        # https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts+from+IFS+and+AIFS
        # Product "oper" runs 00z, 12z, 0h to 144h by 3h, 144h to 240h by 6h
        # Product "scda" runs 06z, 18z, 0h to 90h by 3h
        # **BUT**, see https://github.com/blaylockbk/Herbie/discussions/421
        # Starting 2024-11-12 06:00, 'scda' runs to 144h by 3h
        # Starting 2024-11-12 12:00, 'oper' runs to 360h by 6h

        # # round to last 6 hours to start
        # date = init_date.floor('6h')
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset
        fxx_max = run_length + lead_time_to_start

        # # pick init time based on forecast max lead time:
        # # check if 'scda' product is ideal
        # if init_date.hour == 6 or init_date.hour == 18:
        #     if init_date >= pd.to_datetime('2024-11-12 06:00'):
        #         scda_fxx_max = 144
        #     else:
        #         scda_fxx_max = 90
        #     if fxx_max > scda_fxx_max:  # forecast beyond scda
        #         update_freq = '12h'  # must use 'oper' runs
        #         warnings.warn(
        #             ("You have specified an init_date which would have mapped "
        #              "to a 06z or 18z. Those runs the IFS 'scda' product, and "
        #              "'scda' only goes out 144 hours (90h prior to 2024-11-12)"
        #              ". You will get forecasts from the 'oper' run 6 hours "
        #              "earlier, instead."))
        #     else:
        #         update_freq = '6h'  # can use 'oper' or 'scda'
        # else:
        #     update_freq = '6h'  # can use 'oper' or 'scda'
        # # round down to last actual initialization time
        # date = init_date.floor(update_freq)

        # # offset in hours between selected init_date and fcast run
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset
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
        init_date = pd.to_datetime(init_date)
        # scda goes away/went away 2026-05-12
        # see https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+50r1
        if (init_date.tz_localize(None) <
            pd.to_datetime('2026-05-12 06:00') and
            (init_date.hour == 6 or
             init_date.hour == 18)):
            product = 'scda'
        else:
            product = 'oper'

        if resource_type == 'solar':
            search_str = ':ssrd|10[uv]|2t:sfc'
        elif resource_type == 'wind':
            search_str = ':10[uv]|:100[uv]|:2t:sfc|:sp:'

    elif model == 'aifs' or model == 'aifs_ens':
        # From https://www.ecmwf.int/en/forecasts/datasets/set-ix,
        # https://www.ecmwf.int/en/forecasts/dataset/set-x
        # 4 forecast runs per day (00/06/12/18)
        # 6 hourly steps to 360 (15 days)

        # # round to last 6 hours to start
        # date = init_date.floor('6h')
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset
        fxx_max = run_length + lead_time_to_start

        # update_freq = '6h'
        # # round down to last actual initialization time
        # date = init_date.floor(update_freq)

        # # offset in hours between selected init_date and fcast run
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset
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
            search_str = 'DSWRF|VBDSF|:TMP:2 m above|[UV]GRD:10 m above'
        elif resource_type == 'wind':
            search_str = (
                '[UV]GRD:10 m above|[UV]GRD:80 m above|'
                ':TMP:2 m above|PRES:surface'
            )

        # update_freq = '1h'

        # # round down to last actual initialization time
        # date = init_date.floor(update_freq)

        fxx_range = range(lead_time_to_start, fxx_max + 1, 1)

    elif model == 'cams':
        # From https://confluence.ecmwf.int/display/CKB/CAMS%3A+Global+atmospheric+composition+forecast+data+documentation
        # Runs 00z and 12z, 0-120h by 1h for single-level parameters
        # 00 UTC data available by 10:00 UTC
        # 12 UTC data available by 22:00 UTC
        # Data could be available earlier, no guarantee.
        # also see https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts
        product = None
        search_str = None

        # # round to last 12 hours to start
        # date = init_date.floor('12h')
        # init_offset = int((init_date - date).total_seconds()/3600)
        # lead_time_to_start = lead_time_to_start + init_offset

        # maximum forecast horizon
        fxx_max = run_length + lead_time_to_start
        fxx_range = range(lead_time_to_start, fxx_max + 1, 1)

    # strip tz from init_date if it has a tz
    init_date = init_date.tz_localize(None)

    return init_date, fxx_range, product, search_str


try:
    import dynamical_catalog
except ImportError:
    _has_dynamical_catalog = False
else:
    _has_dynamical_catalog = True


def get_fcast_df(latitude, longitude, init_date, fxx_range, model,
                 search_str, priority, product=None,
                 fast=False, attempts=2, resource_type='solar',
                 member=None, full_ens=False, get_ens_temp=False,
                 get_ens_wind=False, hrrr_coursen_window=None):
    """
    Function to return a dataframe of forecasted resource data.

    Parameters
    ----------
    latitude : float or list of floats
        Latitude in decimal degrees. Positive north of equator, negative
        to south.

    longitude : float or list of floats
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.

    init_date : pandas-parsable datetime
        Model initialization datetime. Note that this should be UTC and on the
        hour for the models currently available with hefty, and most models
        don't initialize every hour. See
        :py:func:`hefty.utilities.adjust_forecast_datetimes` for help
        determining appropriate init_date values.

    fxx_range : int or list of ints
        fxx (lead time) values. Expected to come from
        :py:func:`hefty.utilities.model_input_formatter`

    model : string, default 'gfs'
        Forecast model. Can be NOAA GFS ('gfs'), ECMWF IFS single ('ifs')
        of ensemble ('ifs_ens'), ECMWF AIFS single ('aifs') or esnsemble
        ('aifs_ens'), NOAA HRRR ('hrrr'), or NOAA GEFS ensemble ('gefs').
        ECMWF CAMS ('cams') is an experimental option. It requires cdsapi
        to be installed and a CDS API key to be passed via the
        'cams_api_key' parameter.

    search_str : string
        wgrib2-style search string for Herbie to select variables of
        interest.

    priority : list or string
        List of model sources to get the data in the order of download
        priority, or string for a single source. See Herbie docs.
        Typical values would be 'aws' or 'google'. Now includes option
        of 'dynamical' to get data from dynamical.org. To use 'dynamical',
        it must be a single string, not part of a list.

    product : string, default None
        Herbie product.

    fast : boolean, default False
        Use FastHerbie for herbie sources, default False

    attempts : int
        Number of attempts to try if using Herbie.

    resource_type : {'solar, 'wind'}
        Resrouce type. Default is 'solar'.

    member : int or string or None, default None
        Valid member for IFS ensemble, AIFS ensemble, or GEFS. Could be 0-51
        for IFS/AIFS, where 0 is the control and 1-50 are perturbed members,
        0-31 for GEFS, where 0 is control and 1-30 are perturbed members. Can
        also be 'avg' or 'mean' (case-insensitive) to get the ensemble mean.

    hrrr_coursen_window : int or None, default None
        If model is 'hrrr', optional setting that is the x and y window size
        for coarsening the xarray dataset, effectively applying spatial
        smoothing to the HRRR model. The HRRR has a native resolution of
        about 3 km, so a value of 10 results in approx. 30 x 30 km grid.
        Does not currently work with priority='dynamical'.

    Returns
    -------
    df_out : pandas.DataFrane
        raw output dataframe of forecasted parameters in the native time
        step. Requires further processing to get "proper" hourly data.
    """

    search_string_list = search_str.split('|')

    use_fastherbie = False
    use_herbie = False
    if priority.lower() in [x.lower() for x in [
          'aws', 'google', 'azure', 'nomads', 'ecmwf']]:
        if fast:
            use_fastherbie = True
        else:
            use_herbie = True
        if model == 'ifs_ens':
            model = 'ifs'
            print('product')

    if use_herbie:
        num_datasets = len(search_string_list)
        search_str = '|'.join(search_string_list)
        if model == 'hrrr':
            num_datasets -= 1  # DNI and GHI will show up in a single dataset
        i = []
        for fxx in fxx_range:
            # get solar, 10m wind, and 2m temp data
            # try n times based loosely on
            # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
            for attempts_remaining in reversed(range(attempts)):
                attempt_num = attempts - attempts_remaining
                try:
                    if attempt_num == 1:
                        # try downloading
                        ds = Herbie(
                            init_date,
                            model=model,
                            product=product,
                            fxx=fxx,
                            member=member,
                            priority=priority
                            ).xarray(search_str)
                        # address GH#77
                        if len(ds) < num_datasets:
                            msg = ('Parameters appear to be '
                                   'missing. Another download'
                                   ' will be attempted if there are attempts'
                                   ' remaining.')
                            raise ValueError(msg)
                        # merge - override avoids height conflict between 2m
                        # temp and 10m wind
                        ds = xr.merge(ds, compat='override')
                    else:
                        # after first attempt, set overwrite=True to overwrite
                        # partial files
                        ds = Herbie(
                            init_date,
                            model=model,
                            product=product,
                            fxx=fxx,
                            member=member,
                            priority=priority
                            ).xarray(search_str, overwrite=True)
                        # address GH#77
                        if len(ds) < num_datasets:
                            msg = ('Parameters appear to be '
                                   'missing. Another download'
                                   ' will be attempted if there are attempts'
                                   ' remaining.')
                            raise ValueError(msg)
                        # merge - override avoids height conflict between 2m
                        # temp and 10m wind
                        ds = xr.merge(ds, compat='override')
                except Exception as e:
                    print(e)
                    if attempts_remaining:
                        print('attempt ' + str(attempt_num)
                              + ' failed, pause for '
                              + str((attempt_num)**2) + ' min')
                        time.sleep(60*(attempt_num)**2)
                    else:
                        raise ValueError(f'download failed, ran out of '
                                         f'attempts with error: {e}')
                else:
                    break

            # calculate wind speed from u and v components
            ds = ds.herbie.with_wind('speed')

            if model == 'hrrr' and hrrr_coursen_window is not None:
                ds = ds.coarsen(x=hrrr_coursen_window,
                                y=hrrr_coursen_window,
                                boundary='trim').mean()

            # use pick_points for single point or list of points
            i.append(
                ds.herbie.pick_points(
                    pd.DataFrame(
                        {
                            "latitude": latitude,
                            "longitude": longitude,
                        }
                    )
                )
            )
        ts = xr.concat(i, dim="valid_time")  # concatenate
        # rename 'ssrd' to 'sdswrf' in ifs/aifs
        if model == 'ifs' or model == 'aifs':
            ts = ts.rename({'ssrd': 'sdswrf'})
        # convert to dataframe
        if model == 'hrrr':  # include direct, vbdsf
            df_temp = ts.to_dataframe()[['sdswrf', 'vbdsf',
                                         't2m', 'si10']]
        else:
            df_temp = ts.to_dataframe()[['sdswrf', 't2m', 'si10']]
        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')
        # rename wind speed
        df_temp = df_temp.rename(columns={'si10': 'wind_speed'})
        # convert air temperature units
        df_temp['temp_air'] = df_temp['t2m'] - 273.15

    elif use_fastherbie:
        i = []
        ds_dict = {}
        FH = FastHerbie([init_date], model=model, product=product,
                        fxx=fxx_range, member=member, priority=priority)
        for j in range(0, len(search_string_list)):
            # get solar, 10m wind, and 2m temp data
            # try n times based loosely on
            # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
            for attempts_remaining in reversed(range(attempts)):
                attempt_num = attempts - attempts_remaining
                try:
                    if attempt_num == 1:
                        # try downloading
                        FH.download(search_string_list[j])
                        ds_dict[j] = FH.xarray(search_string_list[j],
                                               remove_grib=True)
                        # calculate wind speed from u and v components if relevant
                        if ('uv' in search_string_list[j] or
                                'UV' in search_string_list[j]):
                            ds_dict[j] = ds_dict[j].herbie.with_wind('speed')
                        # merge - override avoids height conflict between 2m temp
                        # and 10m wind
                        ds = xr.merge(ds_dict.values(), compat='override')
                    else:
                        # after first attempt, set overwrite=True to overwrite
                        # partial files
                        FH.download(search_string_list[j])
                        ds_dict[j] = FH.xarray(search_string_list[j],
                                               remove_grib=True,
                                               overwrite=True)
                        # calculate wind speed from u and v components if relevant
                        if ('uv' in search_string_list[j] or
                                'UV' in search_string_list[j]):
                            ds_dict[j] = ds_dict[j].herbie.with_wind('speed')
                        # merge - override avoids height conflict between 2m temp
                        # and 10m wind
                        ds = xr.merge(ds_dict.values(), compat='override')
                except Exception as e:
                    print(e)
                    if attempts_remaining:
                        print(f'attempt {str(attempt_num)} failed, pause for '
                              f'{str((attempt_num)**2)} min')
                        time.sleep(60*(attempt_num)**2)
                    else:
                        raise ValueError(f'download failed, ran out of '
                                         f'attempts with error: {e}')
                else:
                    break

            if model == 'hrrr' and hrrr_coursen_window is not None:
                ds = ds.coarsen(x=hrrr_coursen_window,
                                y=hrrr_coursen_window,
                                boundary='trim').mean()

            # use pick_points for single point or list of points
            i.append(
                ds.herbie.pick_points(
                    pd.DataFrame(
                        {
                            "latitude": latitude,
                            "longitude": longitude,
                        }
                    )
                )
            )
        # convert to dataframe
        # rename 'ssrd' to 'sdswrf' in ifs/aifs
        if model == 'ifs' or model == 'aifs':
            df_temp = i[-1].to_dataframe()[['valid_time', 'ssrd',
                                            't2m', 'si10']]
            df_temp = df_temp.rename(columns={'ssrd': 'sdswrf'})
        elif model == 'hrrr':
            df_temp = i[-1].to_dataframe()[['valid_time', 'sdswrf', 'vbdsf',
                                            't2m', 'si10']]
        else:
            df_temp = i[-1].to_dataframe()[['valid_time', 'sdswrf',
                                            't2m', 'si10']]

        # make 'valid_time' an index with 'point', drop 'step'
        df_temp = (df_temp.reset_index().set_index(['valid_time', 'point'])
                   .drop('step', axis=1))

        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')
        # rename wind speed
        df_temp = df_temp.rename(columns={'si10': 'wind_speed'})
        # convert air temperature units
        df_temp['temp_air'] = df_temp['t2m'] - 273.15
    elif priority == 'dynamical':
        if not _has_dynamical_catalog:
            raise ImportError((
                "`dynamical_catalog` is required to use priority='dynamical'."
                " Please install it, e.g., with `pip install "
                "dynamical_catalog`."))
        if hrrr_coursen_window is not None:
            raise ValueError(
                "hrrr_coursen_window option is not"
                " currently available with priority='dynamical'")
        ifs_single = False
        if model == 'hrrr':
            if pd.Timestamp(init_date).hour in {0, 6, 12, 18}:
                dataset_id = 'noaa-hrrr-forecast-48-hour'
            else:
                print('accessing dynamical.org HRRR 18h *virtual* -'
                      ' this will be slower than the HRRR 48h.')
                dataset_id = 'noaa-hrrr-forecast-18-hour-virtual'
        elif model == 'gfs':
            dataset_id = 'noaa-gfs-forecast'
        elif model == 'gefs':
            if pd.Timestamp(init_date).hour != 0:
                raise ValueError('gefs is only available from '
                                 'dynamical.org for 00Z cycles')
            else:
                dataset_id = 'noaa-gefs-forecast-35-day'
        elif model in {'ifs', 'ifs_ens'}:
            if pd.Timestamp(init_date).hour != 0:
                raise ValueError('ifs/ifs_ens is only available from'
                                 'dynamical.org for 00Z cycles')
            else:
                dataset_id = 'ecmwf-ifs-ens-forecast-15-day-0-25-degree'
                if model == 'ifs':
                    ifs_single = True
        elif model == 'aifs':
            dataset_id = 'ecmwf-aifs-single-forecast'
        elif model == 'aifs_ens':
            dataset_id = 'ecmwf-aifs-ens-forecast'

        # adjust 'member' if needed
        # if ifs_single, or if model is an ensemble and a member is provided
        if ifs_single or (member is not None):
            if ifs_single:
                member = 0
                model = 'ifs_ens'  # change to ifs_ens, as dynamical doesn't have ifs single
            if isinstance(member, str) and (member.lower() in
                                            [x.lower() for x in
                                             ['avg', 'mean']]):
                member = 'mean'
            elif isinstance(member, str):
                member = int(''.join(filter(str.isdigit, member)))

        # translate Herbie search strings from model_input_formatter to lists of dynamical catalog variables
        mapping_in = {
            'DSWRF': ['downward_short_wave_radiation_flux_surface'],  # NOAA GHI
            'VBDSF': ['visible_beam_downward_solar_flux_surface'],  # NOAA HRRR DNI
            ':TMP:2 m above': ['temperature_2m'],  # NOAA 2m temp
            '[UV]GRD:10 m above': ['wind_u_10m', 'wind_v_10m'],  # NOAA 10m wind
            'ssrd': ['downward_short_wave_radiation_flux_surface'],  # IFS/AIFS GHI
            ':ssrd': ['downward_short_wave_radiation_flux_surface'],  # IFS/AIFS GHI, with the leading ":"
            '2t:sfc': ['temperature_2m'],  # IFS/AIFS 2m temp
            ':2t:sfc': ['temperature_2m'],  # IFS/AIFS 2m temp, with the leading ":"
            '10[uv]': ['wind_u_10m', 'wind_v_10m'],  # IFS/AIFS 10m wind
            ':10[uv]': ['wind_u_10m', 'wind_v_10m'],  # IFS/AIFS 10m wind, with the leading ":"
            '[UV]GRD:80 m above': ['wind_u_80m', 'wind_v_80m'],  # GFS/GEFS
            '[UV]GRD:100 m above': ['wind_u_100m', 'wind_v_100m'],  # GFS/GEFS
            'PRES:surface': ['pressure_surface'],  # GFS
            ':TMP:80 m above': ['temperature_80m'],  # GFS/GEFS
            'PRES:80 m above': ['pressure_80m'],  # GFS/GEFS
            ':100[uv]': ['wind_u_100m', 'wind_v_100m'],  # IFS/AIFS
            ':sp:': ['pressure_surface'],  # IFS/AIFS
        }

        # TODO: mapping out needs to change depending on resource_type
        if resource_type == 'solar':
            # map dynamical:internal hefty variable names
            mapping_out = {
                'downward_short_wave_radiation_flux_surface': 'sdswrf',
                'visible_beam_downward_solar_flux_surface': 'vbdsf',
                'temperature_2m': 'temp_air',
                'wind_speed_10m': 'wind_speed',
                'ensemble_member': 'number',  # hefty uses "number" to indicate ensemble member
            }
        elif resource_type == 'wind':
            # map dynamical:internal hefty variable names
            mapping_out = {
                'wind_speed_10m': 'wind_speed_10m',
                'wind_speed_80m': 'wind_speed_80m',
                'wind_speed_100m': 'wind_speed_100m',
                'wind_direction_10m': 'wind_direction_10m',
                'wind_direction_80m': 'wind_direction_80m',
                'wind_direction_100m': 'wind_direction_100m',
                'temperature_2m': 'temp_air_2m',
                'temperature_80m': 'temp_air_80m',
                'pressure_surface': 'pressure_0m',
                'pressure_80m': 'pressure_80m',
                'ensemble_member': 'number',  # hefty uses "number" to indicate ensemble member
            }

        # dynamical variables list
        # replace each search string value with a list of dynamical variables
        list1 = [mapping_in.get(a, a) for a in search_string_list]
        # flatten list of lists
        dynamical_var_list = [s for list1 in list1 for s in list1]

        # make a locations dataset
        locations = (pd.DataFrame({
            'latitude': latitude,
            'longitude': longitude
            }).reset_index().rename(columns={'index': 'point'})
            .set_index('point'))
        locations_ds = locations[["latitude", "longitude"]].to_xarray()

        # open dataset
        ds = dynamical_catalog.open(dataset_id=dataset_id, chunks=None)

        # hrrr requires custom coordinates transform
        if dataset_id in {'noaa-hrrr-forecast-48-hour',
                          'noaa-hrrr-forecast-18-hour-virtual'}:
            # from
            # https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/ex_python_plot_zarr.html
            import cartopy.crs as ccrs
            # projection = ccrs.LambertConformal(central_longitude=262.5,
            #                        central_latitude=38.5,
            #                        standard_parallels=(38.5, 38.5),
            #                         globe=ccrs.Globe(semimajor_axis=6371229,
            #                                          semiminor_axis=6371229))
            # xyz = projection.transform_points(src_crs=ccrs.PlateCarree(),
            #                                   x=locations["longitude"],
            #                                   y=locations["latitude"])
            # locations['x'], locations['y'], _ = map(list, zip(*xyz))
            # locations_ds = locations[['x', 'y']].to_xarray()

            # similar to mesowest.utah.edu example,
            # but from
            # https://github.com/dynamical-org/notebooks/blob/main/noaa-hrrr-forecast-18-hour-virtual.ipynb
            crs = ds.spatial_ref.attrs
            hrrr_proj = ccrs.LambertConformal(
                central_longitude=crs["longitude_of_central_meridian"],
                central_latitude=crs["latitude_of_projection_origin"],
                standard_parallels=crs["standard_parallel"],
                globe=ccrs.Globe(
                    semimajor_axis=crs["semi_major_axis"],
                    semiminor_axis=crs["semi_minor_axis"],
                ),
            )
            xyz = hrrr_proj.transform_points(src_crs=ccrs.PlateCarree(),
                                             x=locations["longitude"],
                                             y=locations["latitude"])
            locations['x'], locations['y'], _ = map(list, zip(*xyz))
            locations_ds = locations[['x', 'y']].to_xarray()

            # get dataarray
            da = (
                ds[dynamical_var_list]
                .sel(init_time=pd.Timestamp(init_date))
                .sel(x=locations_ds.x,
                     y=locations_ds.y,
                     method="nearest")
                .sel(lead_time=slice(pd.Timedelta(hours=min(fxx_range)),
                                     pd.Timedelta(hours=max(fxx_range))))
                .load()
            )           
        else:
            locations_ds = locations[["latitude", "longitude"]].to_xarray()
            if (model in {'gefs', 'ifs_ens', 'aifs_ens'} and
                    isinstance(member, int)):
                # get dataarray, only the specified member
                da = (
                    ds[dynamical_var_list]
                    .sel(init_time=pd.Timestamp(init_date))
                    .sel(latitude=locations_ds.latitude,
                         longitude=locations_ds.longitude,
                         method="nearest")
                    .sel(lead_time=slice(pd.Timedelta(hours=min(fxx_range)),
                                         pd.Timedelta(hours=max(fxx_range))))
                    .sel(ensemble_member=member)
                    .load()
                )
            elif ((model in {'gefs', 'ifs_ens', 'aifs_ens'}) and
                  (member == 'mean')):
                da = (
                    ds[dynamical_var_list]
                    .sel(init_time=pd.Timestamp(init_date))
                    .sel(latitude=locations_ds.latitude,
                         longitude=locations_ds.longitude,
                         method="nearest")
                    .sel(lead_time=slice(pd.Timedelta(hours=min(fxx_range)),
                                         pd.Timedelta(hours=max(fxx_range))))
                    .mean(dim='ensemble_member')
                    .load()
                )
                da['number'] = 'mean'
            else:
                # get dataarray
                da = (
                    ds[dynamical_var_list]
                    .sel(init_time=pd.Timestamp(init_date))
                    .sel(latitude=locations_ds.latitude,
                         longitude=locations_ds.longitude,
                         method="nearest")
                    .sel(lead_time=slice(pd.Timedelta(hours=min(fxx_range)),
                                         pd.Timedelta(hours=max(fxx_range))))
                    .load()
                )

        # convert to dataframe
        df = da.to_dataframe().reset_index().set_index('valid_time')

        # calculate wind speed and direction
        if 'wind_u_10m' in df.columns:
            df['wind_speed_10m'] = np.sqrt(df['wind_u_10m']**2 +
                                           df['wind_v_10m']**2)
            df['wind_direction_10m'] = (
                    (270 - np.rad2deg(
                        np.arctan2(df['wind_u_10m'],
                                   df['wind_v_10m']))) % 360
            )
        if 'wind_u_80m' in df.columns:
            df['wind_speed_80m'] = np.sqrt(df['wind_u_80m']**2 +
                                           df['wind_v_80m']**2)
            df['wind_direction_80m'] = (
                    (270 - np.rad2deg(
                        np.arctan2(df['wind_u_80m'],
                                   df['wind_v_80m']))) % 360
            )
        if 'wind_u_100m' in df.columns:
            df['wind_speed_100m'] = np.sqrt(df['wind_u_100m']**2 +
                                            df['wind_v_100m']**2)
            df['wind_direction_100m'] = (
                    (270 - np.rad2deg(
                        np.arctan2(df['wind_u_100m'],
                                   df['wind_v_100m']))) % 360
            )

        # convert temperature units to celsius
        # dynamical.org temperatures are already in celsius...
        # if 'temperature_2m' in df.columns:
        #     df['temperature_2m'] = df['temperature_2m'] - 273.15
        # if 'temperature_80m' in df.columns:
        #     df['temperature_80m'] = df['temperature_80m'] - 273.15

        # rename columns to hefty-friendly variable names
        df = df.rename(columns=mapping_out)

        # add timezone
        df = df.tz_localize('UTC', level='valid_time')

        # make index valid_time and point
        df = df.reset_index().set_index(['valid_time', 'point'])

        # calculate lead time in hours
        df['lead_time'] = df['lead_time'].dt.total_seconds() / 3600

        # filter to columns of interest
        keep_cols = (list(mapping_out.values()) +
                     ['lead_time', 'latitude', 'longitude'])
        df_temp = df[df.columns.intersection(keep_cols)]

        # if full_ens == False:

    return df_temp
