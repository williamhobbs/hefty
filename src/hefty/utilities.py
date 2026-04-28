import pandas as pd
import warnings
import math


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
        'end_hour': [144, 360, 90],
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
    # delays based on https://dynamical.org/status/ as of 2026-04-24
    fcast_sched_dict_ifs_ens_1 = {
        'start_date': ['2024-03-10 12:00',
                       '2024-03-10 12:00',
                       '2024-03-10 18:00'],  # https://herbie.readthedocs.io/en/stable/gallery/ecmwf_models/ecmwf.html#Data-Availability
        'start_hour': [0, 150, 0],
        'end_hour': [144, 240, 90],
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
        'end_hour': [360, 96],
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
        if init_date.hour == 6 or init_date.hour == 18:
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
