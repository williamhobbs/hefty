import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie, FastHerbie
import time


def get_custom_forecast(latitude, longitude, init_date, run_length,
                        lead_time_to_start=0, period=3, model='gfs',
                        product='pgrb2.0p25', search_str=':TMP:2 m above', 
                        member=None, attempts=2, hrrr_hour_middle=True,
                        hrrr_coursen_window=None, priority=None):
    """
    Get a custom forecast for one or several sites from one of several
    NWPs. This function uses Herbie [1]_.

    Parameters
    ----------
    latitude : float or list of floats
        Latitude in decimal degrees. Positive north of equator, negative
        to south.

    longitude : float or list of floats
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.

    init_date : pandas-parsable datetime
        Model initialization datetime.

    run_length : int
        Length of the forecast in hours - number of hours forecasted

    search_str : string
        regex search string for grib files. See [2]_ for more info.

    lead_time_to_start : int, optional
        Number of hours between init_date (initialization) and
        the first forecasted interval. NOAA GFS data goes out
        384 hours, so run_length + lead_time_to_start must be less
        than or equal to 384.

    model : string, default 'gfs'
        Forecast model. Default is NOAA GFS ('gfs'), but can also be
        ECMWF IFS ('ifs'), NOAA HRRR ('hrrr'), or NOAA GEFS ('gefs).

    member: string or int
        For models that are ensembles, pass an appropriate single member label.

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
        timeseries forecasted weather data

    References
    ----------

    .. [1] `Blaylock, B. K. (YEAR). Herbie: Retrieve Numerical Weather
       Prediction Model Data (Version 20xx.x.x) [Computer software].
       <https://doi.org/10.5281/zenodo.4567540>`_
    .. [2] <https://herbie.readthedocs.io/en/latest/user_guide/tutorial/search.html> # noqa
    """

# variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # get model-specific Herbie inputs
    date = init_date
    fxx_max = run_length + lead_time_to_start
    fxx_range = range(lead_time_to_start, fxx_max, period)

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
                        date,
                        model=model,
                        product=product,
                        fxx=fxx,
                        member=member,
                        priority=priority
                        ).xarray(search_str)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    ds = Herbie(
                        date,
                        model=model,
                        product=product,
                        fxx=fxx,
                        member=member,
                        priority=priority
                        ).xarray(search_str, overwrite=True)
            except Exception:
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
            else:
                break
        else:
            raise ValueError('download failed, ran out of attempts')

        # merge - override avoids hight conflict between 2m temp and 10m wind
        ds = xr.merge(ds, compat='override')
        # calculate wind speed from u and v components
        ds = ds.herbie.with_wind('both')

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

    # convert to dataframe
    df_temp = ts.to_dataframe()

    # work through sites
    dfs = {}  # empty list of dataframes
    if type(latitude) is float or type(latitude) is int:
        num_sites = 1
    else:
        num_sites = len(latitude)

    for j in range(num_sites):
        df = df_temp[df_temp.index.get_level_values('point') == j]
        df = df.droplevel('point')

        if model == 'hrrr' and hrrr_hour_middle is False:
            # keep top of hour instantaneous HRRR convention
            dfs[j] = df
        else:
            # 60min version of data, centered at bottom of the hour
            # 1min interpolation, then 60min mean
            df_60min = (
                df
                .resample('1min')
                .interpolate()
                .resample('60min').mean()
            )
            df_60min.index = df_60min.index + pd.Timedelta('30min')
            dfs[j] = df_60min

    # concatenate creating multiindex with keys of the list of point numbers
    # assigned to 'point', reorder indices, and sort by valid_time
    df_60min = (
        pd.concat(dfs, keys=list(range(num_sites)), names=['point'])
        .reorder_levels(["valid_time", "point"])
        .sort_index(level='valid_time')
    )

    # set "point" index as a column
    df_60min = df_60min.reset_index().set_index('valid_time')

    # drop unneeded columns if they exist
    # df_60min = df_60min.drop(['t2m', 'sdswrf'], axis=1, errors='ignore')

    return df_60min
