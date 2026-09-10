import pandas as pd
import xarray as xr
from herbie import Herbie
import time
from hefty.utilities import model_input_formatter, get_fcast_dataframe


def get_wind_forecast(latitude, longitude, init_date, run_length,
                      lead_time_to_start=0, model='gfs', member='avg',
                      attempts=2, hrrr_hour_middle=True,
                      hrrr_coursen_window=None, priority=None):
    """
    Get a wind resource forecast for one or several sites from one of several
    NWPs. This function uses Herbie [1]_ and pvlib [2]_.

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

    run_length : int
        Length of the forecast in hours - number of hours forecasted

    lead_time_to_start : int, optional
        Number of hours between init_date (initialization) and
        the first forecasted interval. NOAA GFS data goes out
        384 hours, so run_length + lead_time_to_start must be less
        than or equal to 384.

    model : string, default 'gfs'
        Forecast model. Default is NOAA GFS ('gfs'), but can also be ECMWF IFS
        single ('ifs'), ECMWF AIFS single ('aifs'), NOAA HRRR ('hrrr'), or
        NOAA GEFS ('gefs') (a single member from the ensemble). Unlike
        :py:func:`hefty.solar.get_solar_forecast`, ECMWF CAMS is not an option
        because the CAMS version of IFS does not include 80 or 100m wind
        speed.

    member: string or int, default 'avg'
        For models that are ensembles (GEFS is the only current option),
        pass an appropriate single member label. See Herbie documentation for
        details [1]_. Options for GEFS include 'avg' or 'mean' (the ensemble
        mean), 0 or 'c00' (control member), and 1-30 or 'p01'-'p30' for the 30
        individual members.

    attempts : int, optional
        Number of times to try getting forecast data. The function will pause
        for n^2 minutes after each n attempt, e.g., 1 min after the first
        attempt, 4 minutes after the second, etc.

    hrrr_hour_middle : bool, default True
        If model is 'hrrr', setting this False keeps the forecast at the
        native instantaneous top-of-hour format. True (default) shifts
        the forecast to middle of the hour, more closely representing an
        integrated hourly forecast that is centered in the middle of the
        hour.

    hrrr_coursen_window : int or None, default None
        If model is 'hrrr', optional setting that is the x and y window size
        for coarsening the xarray dataset, effectively applying spatial
        smoothing to the HRRR model. The HRRR has a native resolution of
        about 3 km, so a value of 10 results in approx. 30 x 30 km grid.

    priority : list or string
        List of model sources to get the data in the order of download
        priority, or string for a single source. See Herbie docs.
        Typical values would be 'aws' or 'google'.

    Returns
    -------
    data : pandas.DataFrane
        timeseries forecasted wind resource data

    References
    ----------

    .. [1] `Blaylock, B. K. (YEAR). Herbie: Retrieve Numerical Weather
       Prediction Model Data (Version 20xx.x.x) [Computer software].
       <https://doi.org/10.5281/zenodo.4567540>`_
    .. [2] `Anderson, K., et al. “pvlib python: 2023 project update.” Journal
       of Open Source Software, 8(92), 5994, (2023).
       <http://dx.doi.org/10.21105/joss.05994>`_
    """

    # # set clear sky model. could be an input variable at some point
    # model_cs = 'haurwitz'

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    num_sites = len(latitude)
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    fast = False
    resource_type = 'wind'

    # check if init_date is top of hour
    if init_date != init_date.floor('1h'):
        raise ValueError(f'init_date must be on the hour, e.g., '
                         f'{init_date.floor('1h')}, not {init_date}. '
                         'Consider using init_date.floor("1h") or '
                         'similar')

    # get model-specific Herbie inputs
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start, model, resource_type)

    df_temp = get_fcast_dataframe(
        latitude, longitude, date, fxx_range, model,
        search_str, priority, product,
        fast, attempts, resource_type,
        member, hrrr_coursen_window=hrrr_coursen_window)

    # work through sites
    dfs = {}  # empty list of dataframes
    if type(latitude) is float or type(latitude) is int:
        num_sites = 1
    else:
        num_sites = len(latitude)

    for j in range(num_sites):
        df = df_temp[df_temp['point'] == j]
        df = df.drop(['point'], axis=1)  # drop point column, we will add it back later

        if model == 'hrrr' and hrrr_hour_middle is False:
            # keep top of hour instantaneous HRRR convention
            dfs[j] = df
        else:
            # 60min version of data, centered at bottom of the hour
            new_index = pd.date_range(df.index.min(),
                                      df.index.max(),
                                      freq='30min',
                                      name='valid_time')
            df_interp = df.reindex(
                new_index).interpolate(method='time')
            df_60min = df_interp[df_interp.index.minute == 30]

            dfs[j] = df_60min

    # concatenate creating multiindex with keys of the list of point numbers
    # assigned to 'point', reorder indices, and sort by valid_time
    df_60min = (
        pd.concat(dfs, keys=list(range(num_sites)), names=['point'])
        .reorder_levels(["valid_time", "point"])
        .sort_index(level='valid_time')
    )

    df_60min = df_60min.reset_index().set_index('valid_time')

    return df_60min
