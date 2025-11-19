import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie, FastHerbie
import pvlib
import time
from hefty.utilities import model_input_formatter


def get_solar_forecast(latitude, longitude, init_date, run_length,
                       lead_time_to_start=0, model='gfs', member=None,
                       attempts=2, hrrr_hour_middle=True,
                       hrrr_coursen_window=None, priority=None):
    """
    Get a solar resource forecast for one or several sites from one of several
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
        Model initialization datetime.

    run_length : int
        Length of the forecast in hours - number of hours forecasted

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
        timeseries forecasted solar resource data

    References
    ----------

    .. [1] `Blaylock, B. K. (YEAR). Herbie: Retrieve Numerical Weather
       Prediction Model Data (Version 20xx.x.x) [Computer software].
       <https://doi.org/10.5281/zenodo.4567540>`_
    .. [2] `Anderson, K., et al. “pvlib python: 2023 project update.” Journal
       of Open Source Software, 8(92), 5994, (2023).
       <http://dx.doi.org/10.21105/joss.05994>`_
    """

    # set clear sky model. could be an input variable at some point
    model_cs = 'haurwitz'

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # get model-specific Herbie inputs
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start, model)

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
    # rename 'ssrd' to 'sdswrf' in ifs
    if model == 'ifs':
        ts = ts.rename({'ssrd': 'sdswrf'})
    # convert to dataframe
    df_temp = ts.to_dataframe()[['sdswrf', 't2m', 'si10']]
    # add timezone
    df_temp = df_temp.tz_localize('UTC', level='valid_time')
    # rename wind speed
    df_temp = df_temp.rename(columns={'si10': 'wind_speed'})
    # convert air temperature units
    df_temp['temp_air'] = df_temp['t2m'] - 273.15

    # work through sites
    dfs = {}  # empty list of dataframes
    if type(latitude) is float or type(latitude) is int:
        num_sites = 1
    else:
        num_sites = len(latitude)

    for j in range(num_sites):
        df = df_temp[df_temp.index.get_level_values('point') == j]
        df = df.droplevel('point')

        loc = pvlib.location.Location(
            latitude=latitude[j],
            longitude=longitude[j],
            tz=df.index.tz
            )

        if model == 'gfs':
            # for gfs ghi: we have to "unmix" the rolling average irradiance
            # that resets every 6 hours
            mixed = df[['sdswrf']].copy()
            mixed['hour'] = mixed.index.hour
            mixed['hour'] = mixed.index.hour
            mixed['hour_of_mixed_period'] = ((mixed['hour'] - 1) % 6) + 1
            mixed['sdswrf_prev'] = mixed['sdswrf'].shift(
                periods=1,
                fill_value=0
                )
            mixed['int_len'] = mixed.index.diff().total_seconds().values / 3600

            # set the first interval length:
            if lead_time_to_start >= 120:
                mixed.loc[mixed.index[0], 'int_len'] = 1
            else:
                mixed.loc[mixed.index[0], 'int_len'] = 3
            unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                        - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                        * mixed['sdswrf_prev']) / mixed['int_len'])
            df['ghi'] = unmixed

        elif model == 'gefs':
            # for gfs ghi: we have to "unmix" the rolling average irradiance
            # that resets every 6 hours
            mixed = df[['sdswrf']].copy()
            mixed['hour'] = mixed.index.hour
            mixed['hour'] = mixed.index.hour
            mixed['hour_of_mixed_period'] = ((mixed['hour'] - 1) % 6) + 1
            mixed['sdswrf_prev'] = mixed['sdswrf'].shift(
                periods=1,
                fill_value=0
                )
            mixed['int_len'] = mixed.index.diff().total_seconds().values / 3600

            # set the first interval length:
            mixed.loc[mixed.index[0], 'int_len'] = 3
            unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                        - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                        * mixed['sdswrf_prev']) / mixed['int_len'])
            df['ghi'] = unmixed

        elif model == 'ifs':
            # for ifs ghi: cumulative J/m^s to average W/m^2 over the interval
            # ending at the valid time. calculate difference in measurement
            # over diff in time to get avg J/s/m^2 = W/m^2
            df['ghi'] = df['sdswrf'].diff() / df.index.diff().seconds.values

        elif model == 'hrrr':
            df['ghi'] = df['sdswrf']

        if model == 'gfs' or model == 'gefs' or model == 'ifs':
            # make 1min interval clear sky data covering our time range
            times = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq='1min',
                tz='UTC')

            cs = loc.get_clearsky(times, model=model_cs)

            # calculate average CS ghi over the intervals from the forecast
            # based on list comprehension example in
            # https://stackoverflow.com/a/55724134/27574852
            ghi = cs['ghi']
            dates = df.index
            ghi_clear = [
                ghi.loc[(ghi.index > dates[i]) & (ghi.index <= dates[i+1])]
                .mean() for i in range(len(dates) - 1)
                ]

            # write to df and calculate clear sky index of ghi
            df['ghi_clear'] = [np.nan] + ghi_clear
            df['ghi_csi'] = df['ghi'] / df['ghi_clear']

            # avoid divide by zero issues
            df.loc[df['ghi'] == 0, 'ghi_csi'] = 0

            # 60min version of data, centered at bottom of the hour
            # 1min interpolation, then 60min mean
            df_60min = (
                df[['temp_air', 'wind_speed']]
                .resample('1min')
                .interpolate()
                .resample('60min').mean()
            )
            # make timestamps center-labeled for instantaneous pvlib modeling
            # later
            df_60min.index = df_60min.index + pd.Timedelta('30min')
            # drop last row, since we don't have data for the last full hour
            # (just an instantaneous end point)
            df_60min = df_60min.iloc[:-1]
            # "backfill" ghi csi
            # merge based on nearest index from 60min version looking forward
            # in 3hr version
            df_60min = pd.merge_asof(
                left=df_60min,
                right=df.ghi_csi,
                on='valid_time',
                direction='forward'
            ).set_index('valid_time')

            # make 60min interval clear sky, centered at bottom of the hour
            times = pd.date_range(
                start=df.index[0]+pd.Timedelta('30m'),
                end=df.index[-1]-pd.Timedelta('30m'),
                freq='60min',
                tz='UTC')
            cs = loc.get_clearsky(times, model=model_cs)

            # calculate ghi from clear sky and backfilled forecasted clear sky
            # index
            df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(times)
            out_erbs = pvlib.irradiance.erbs(
                df_60min.ghi,
                sp.zenith,
                df_60min.index,
            )
            df_60min['dni'] = out_erbs.dni
            df_60min['dhi'] = out_erbs.dhi

            # add clearsky ghi
            df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

            dfs[j] = df_60min

        elif model == 'hrrr':
            if hrrr_hour_middle is True:
                # clear sky index
                times = df.index
                cs = loc.get_clearsky(times, model=model_cs)
                df['csi'] = df['ghi'] / cs['ghi']
                # avoid divide by zero issues
                df.loc[df['ghi'] == 0, 'csi'] = 0

                # make 1min interval clear sky data covering our time range
                times = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq='1min',
                    tz='UTC')

                cs = loc.get_clearsky(times, model=model_cs)
                # calculate 1min interpolated temp_air, wind_speed, csi
                df_01min = (
                    df[['temp_air', 'wind_speed', 'csi']]
                    .resample('1min')
                    .interpolate()
                )
                # add ghi_clear
                df_01min['ghi_clear'] = cs['ghi']
                # calculate hour averages centered labelled at bottom of the
                # hour
                df_60min = df_01min.resample('1h').mean()
                df_60min.index = df_60min.index + pd.Timedelta('30min')
                # calculate new ghi
                df_60min['ghi'] = df_60min['csi'] * df_60min['ghi_clear']

            else:
                df_60min = df.copy()

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(df_60min.index)
            out_erbs = pvlib.irradiance.erbs(
                df_60min.ghi,
                sp.zenith,
                df_60min.index,
            )
            df_60min['dni'] = out_erbs.dni
            df_60min['dhi'] = out_erbs.dhi

            # add clearsky ghi
            cs = loc.get_clearsky(df_60min.index, model=model_cs)
            df_60min['ghi_clear'] = cs['ghi']

            dfs[j] = df_60min.copy()

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
    df_60min = df_60min.drop(['t2m', 'sdswrf'], axis=1, errors='ignore')

    return df_60min


def get_solar_forecast_fast(latitude, longitude, init_date, run_length,
                            lead_time_to_start=0, model='gfs', member=None,
                            attempts=2, hrrr_hour_middle=True,
                            hrrr_coursen_window=None, priority=None):
    """
    Get a solar resource forecast for one or several sites from one of several
    NWPs. This function uses Herbie [1]_ and pvlib [2]_. This version
    uses FastHerbie and may be about 15% faster. It currently only works
    with a single init_date, not a list of dates like FastHerbie can use.

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
        timeseries forecasted solar resource data

    References
    ----------

    .. [1] `Blaylock, B. K. (YEAR). Herbie: Retrieve Numerical Weather
        Prediction Model Data (Version 20xx.x.x) [Computer software].
        <https://doi.org/10.5281/zenodo.4567540>`_
    .. [2] `Anderson, K., et al. “pvlib python: 2023 project update.” Journal
        of Open Source Software, 8(92), 5994, (2023).
        <http://dx.doi.org/10.21105/joss.05994>`_
    """

    # set clear sky model. could be an input variable at some point
    model_cs = 'haurwitz'

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # get model-specific Herbie inputs
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start, model)

    delimiter = '|'
    search_string_list = search_str.split(delimiter)

    i = []
    ds_dict = {}
    FH = FastHerbie([date], model=model, product=product, fxx=fxx_range,
                    member=member, priority=priority)
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
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    FH.download(search_string_list[j])
                    ds_dict[j] = FH.xarray(search_string_list[j],
                                           remove_grib=True,
                                           overwrite=True)
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
        ds = xr.merge(ds_dict.values(), compat='override')
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
    # convert to dataframe
    # rename 'ssrd' to 'sdswrf' in ifs
    if model == 'ifs':
        df_temp = i[-1].to_dataframe()[['valid_time', 'ssrd', 't2m', 'si10']]
        df_temp = df_temp.rename(columns={'ssrd': 'sdswrf'})
    else:
        df_temp = i[-1].to_dataframe()[['valid_time', 'sdswrf', 't2m', 'si10']]

    # make 'valid_time' an index with 'point', drop 'step'
    df_temp = (df_temp.reset_index().set_index(['valid_time', 'point'])
               .drop('step', axis=1))

    # add timezone
    df_temp = df_temp.tz_localize('UTC', level='valid_time')
    # rename wind speed
    df_temp = df_temp.rename(columns={'si10': 'wind_speed'})
    # convert air temperature units
    df_temp['temp_air'] = df_temp['t2m'] - 273.15

    # work through sites
    dfs = {}  # empty list of dataframes
    if type(latitude) is float or type(latitude) is int:
        num_sites = 1
    else:
        num_sites = len(latitude)

    for j in range(num_sites):
        df = df_temp[df_temp.index.get_level_values('point') == j]
        df = df.droplevel('point')

        loc = pvlib.location.Location(
            latitude=latitude[j],
            longitude=longitude[j],
            tz=df.index.tz
            )

        if model == 'gfs':
            # for gfs ghi: we have to "unmix" the rolling average irradiance
            # that resets every 6 hours
            mixed = df[['sdswrf']].copy()
            mixed['hour'] = mixed.index.hour
            mixed['hour'] = mixed.index.hour
            mixed['hour_of_mixed_period'] = ((mixed['hour'] - 1) % 6) + 1
            mixed['sdswrf_prev'] = mixed['sdswrf'].shift(
                periods=1,
                fill_value=0
                )
            mixed['int_len'] = mixed.index.diff().total_seconds().values / 3600

            # set the first interval length:
            if lead_time_to_start >= 120:
                mixed.loc[mixed.index[0], 'int_len'] = 1
            else:
                mixed.loc[mixed.index[0], 'int_len'] = 3
            unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                        - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                        * mixed['sdswrf_prev']) / mixed['int_len'])
            df['ghi'] = unmixed

        elif model == 'gefs':
            # for gfs ghi: we have to "unmix" the rolling average irradiance
            # that resets every 6 hours
            mixed = df[['sdswrf']].copy()
            mixed['hour'] = mixed.index.hour
            mixed['hour'] = mixed.index.hour
            mixed['hour_of_mixed_period'] = ((mixed['hour'] - 1) % 6) + 1
            mixed['sdswrf_prev'] = mixed['sdswrf'].shift(
                periods=1,
                fill_value=0
                )
            mixed['int_len'] = mixed.index.diff().total_seconds().values / 3600

            # set the first interval length:
            mixed.loc[mixed.index[0], 'int_len'] = 3
            unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                        - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                        * mixed['sdswrf_prev']) / mixed['int_len'])
            df['ghi'] = unmixed

        elif model == 'ifs':
            # for ifs ghi: cumulative J/m^s to average W/m^2 over the interval
            # ending at the valid time. calculate difference in measurement
            # over diff in time to get avg J/s/m^2 = W/m^2
            df['ghi'] = df['sdswrf'].diff() / df.index.diff().seconds.values

        elif model == 'hrrr':
            df['ghi'] = df['sdswrf']

        if model == 'gfs' or model == 'gefs' or model == 'ifs':
            # make 1min interval clear sky data covering our time range
            times = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq='1min',
                tz='UTC')

            cs = loc.get_clearsky(times, model=model_cs)

            # calculate average CS ghi over the intervals from the forecast
            # based on list comprehension example in
            # https://stackoverflow.com/a/55724134/27574852
            ghi = cs['ghi']
            dates = df.index
            ghi_clear = [
                ghi.loc[(ghi.index > dates[i]) & (ghi.index <= dates[i+1])]
                .mean() for i in range(len(dates) - 1)
                ]

            # write to df and calculate clear sky index of ghi
            df['ghi_clear'] = [np.nan] + ghi_clear
            df['ghi_csi'] = df['ghi'] / df['ghi_clear']

            # avoid divide by zero issues
            df.loc[df['ghi'] == 0, 'ghi_csi'] = 0

            # 60min version of data, centered at bottom of the hour
            # 1min interpolation, then 60min mean
            df_60min = (
                df[['temp_air', 'wind_speed']]
                .resample('1min')
                .interpolate()
                .resample('60min').mean()
            )
            # make timestamps center-labeled for instantaneous pvlib modeling
            # later
            df_60min.index = df_60min.index + pd.Timedelta('30min')
            # drop last row, since we don't have data for the last full hour
            # (just an instantaneous end point)
            df_60min = df_60min.iloc[:-1]
            # "backfill" ghi csi
            # merge based on nearest index from 60min version looking forward
            # in 3hr version
            df_60min = pd.merge_asof(
                left=df_60min,
                right=df.ghi_csi,
                on='valid_time',
                direction='forward'
            ).set_index('valid_time')

            # make 60min interval clear sky, centered at bottom of the hour
            times = pd.date_range(
                start=df.index[0]+pd.Timedelta('30m'),
                end=df.index[-1]-pd.Timedelta('30m'),
                freq='60min',
                tz='UTC')
            cs = loc.get_clearsky(times, model=model_cs)

            # calculate ghi from clear sky and backfilled forecasted clear sky
            # index
            df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(times)
            out_erbs = pvlib.irradiance.erbs(
                df_60min.ghi,
                sp.zenith,
                df_60min.index,
            )
            df_60min['dni'] = out_erbs.dni
            df_60min['dhi'] = out_erbs.dhi

            # add clearsky ghi
            df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

            dfs[j] = df_60min

        elif model == 'hrrr':
            if hrrr_hour_middle is True:
                # clear sky index
                times = df.index
                cs = loc.get_clearsky(times, model=model_cs)
                df['csi'] = df['ghi'] / cs['ghi']
                # avoid divide by zero issues
                df.loc[df['ghi'] == 0, 'csi'] = 0

                # make 1min interval clear sky data covering our time range
                times = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq='1min',
                    tz='UTC')

                cs = loc.get_clearsky(times, model=model_cs)
                # calculate 1min interpolated temp_air, wind_speed, csi
                df_01min = (
                    df[['temp_air', 'wind_speed', 'csi']]
                    .resample('1min')
                    .interpolate()
                )
                # add ghi_clear
                df_01min['ghi_clear'] = cs['ghi']
                # calculate hour averages centered labelled at bottom of the
                # hour
                df_60min = df_01min.resample('1h').mean()
                df_60min.index = df_60min.index + pd.Timedelta('30min')
                # calculate new ghi
                df_60min['ghi'] = df_60min['csi'] * df_60min['ghi_clear']

            else:
                df_60min = df.copy()

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(df_60min.index)
            out_erbs = pvlib.irradiance.erbs(
                df_60min.ghi,
                sp.zenith,
                df_60min.index,
            )
            df_60min['dni'] = out_erbs.dni
            df_60min['dhi'] = out_erbs.dhi

            # add clearsky ghi
            cs = loc.get_clearsky(df_60min.index, model=model_cs)
            df_60min['ghi_clear'] = cs['ghi']

            dfs[j] = df_60min.copy()

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
    df_60min = df_60min.drop(['t2m', 'sdswrf'], axis=1, errors='ignore')

    return df_60min


def get_solar_forecast_ensemble_subset(
        latitude, longitude, init_date, run_length, lead_time_to_start=0,
        model='ifs', attempts=2, num_members=3, priority=None):
    """
    Get solar resource forecasts for one or several sites using a subset of
    ensemble members. Use `get_solar_forecast_ensemble` for all ensemble
    members, or anything over about 25% of members, as it is about 4x
    faster per member. This function uses Herbie's FastHerbie [1]_ and pvlib
    [2]_. It currently only works with a single init_date, not a list of dates
    like FastHerbie can use. Temperature data comes from the ensemble mean,
    and wind speed is currently just a filler value of 2 m/s to save time.

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

    lead_time_to_start : int, optional
        Number of hours between init_date (initialization) and
        the first forecasted interval.

    model : string, default 'ifs'
        Forecast model. Default and only option is ECMWF IFS ('ifs'). NOAA
        GEFS may be added in the future.

    attempts : int, optional
        Number of times to try getting forecast data. The function will pause
        for n^2 minutes after each n attempt, e.g., 1 min after the first
        attempt, 4 minutes after the second, etc.

    num_members : int, default 3
        Number of ensemble members to get. IFS has 50 members.

    priority : list or string
        List of model sources to get the data in the order of download
        priority, or string for a single source. See Herbie docs.
        Typical values would be 'aws' or 'google'.

    Returns
    -------
    data : pandas.DataFrane
        timeseries forecasted solar resource data

    References
    ----------

    .. [1] `Blaylock, B. K. (YEAR). Herbie: Retrieve Numerical Weather
        Prediction Model Data (Version 20xx.x.x) [Computer software].
        <https://doi.org/10.5281/zenodo.4567540>`_
    .. [2] `Anderson, K., et al. “pvlib python: 2023 project update.” Journal
        of Open Source Software, 8(92), 5994, (2023).
        <http://dx.doi.org/10.21105/joss.05994>`_
    """

    # set clear sky model. could be an input variable at some point
    model_cs = 'haurwitz'

    # check model
    if model.casefold() != ('ifs').casefold():
        raise ValueError('model must be ifs, you entered ' + model)

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    num_sites = len(latitude)

    # get model-specific Herbie inputs, except product and search string,
    # which are unique for the ensemble
    init_date, fxx_range, _, _ = model_input_formatter(
        init_date, run_length, lead_time_to_start, model)

    dfs = []

    # loop through IFS ensemble members and get GHI data
    for number in range(1, num_members+1):
        search_str = ':ssrd:sfc:' + str(number) + ':'
        # try n times based loosely on
        # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
        for attempts_remaining in reversed(range(attempts)):
            attempt_num = attempts - attempts_remaining
            try:
                if attempt_num == 1:
                    # try downloading
                    ds = FastHerbie(DATES=[init_date],
                                    model='ifs',
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority).xarray(search_str)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    ds = FastHerbie(DATES=[init_date],
                                    model='ifs',
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority).xarray(search_str,
                                                          overwrite=True)
            except Exception:
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
            else:
                break
        else:
            raise ValueError('download failed, ran out of attempts')

        # use pick_points for single point or list of points
        ds2 = ds.herbie.pick_points(pd.DataFrame({
                        "latitude": latitude,
                        "longitude": longitude,
                        }))
        # convert to dataframe
        df_temp = (ds2
                   .to_dataframe()
                   .reset_index()
                   .set_index('valid_time')[['point', 'ssrd']])
        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')
        # rename ssrd
        df_temp = df_temp.rename(columns={'ssrd': 'sdswrf'})

        # work through sites (points)
        if type(latitude) is float or type(latitude) is int:
            num_sites = 1
        else:
            num_sites = len(latitude)
        for point in range(num_sites):
            df = df_temp[df_temp['point'] == point].copy()

            loc = pvlib.location.Location(
                latitude=latitude[point],
                longitude=longitude[point],
                tz=df.index.tz
                )

            # convert cumulative J/m^s to average W/m^2
            df['ghi'] = df['sdswrf'].diff() / df.index.diff().seconds.values

            # make 1min interval clear sky data covering our time range
            times = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq='1min',
                tz='UTC')
            cs = loc.get_clearsky(times, model=model_cs)

            # calculate average CS ghi over the intervals from the forecast
            # based on list comprehension example in
            # https://stackoverflow.com/a/55724134/27574852
            ghi = cs['ghi']
            dates = df.index
            ghi_clear = [
                ghi.loc[(ghi.index > dates[i]) & (ghi.index <= dates[i+1])]
                .mean() for i in range(len(dates) - 1)
                ]

            # write to df and calculate clear sky index of ghi
            df['ghi_clear'] = [np.nan] + ghi_clear
            df['ghi_csi'] = df['ghi'] / df['ghi_clear']

            # avoid divide by zero issues
            df.loc[df['ghi'] == 0, 'ghi_csi'] = 0

            # make a dummy column
            df['dummy'] = 0

            # 60min version of data, centered at bottom of the hour
            # 1min interpolation, then 60min mean
            df_60min = (
                df['dummy']
                .resample('1min')
                .interpolate()
                .resample('60min').mean()
            )
            # make timestamps center-labeled for instantaneous pvlib modeling
            # later
            df_60min.index = df_60min.index + pd.Timedelta('30min')
            # drop last row, since we don't have data for the last full hour
            # (just an instantaneous end point)
            df_60min = df_60min.iloc[:-1]
            # "backfill" ghi csi
            # merge based on nearest index from 60min version looking forward
            # in 3hr version
            df_60min = pd.merge_asof(
                left=df_60min,
                right=df.ghi_csi,
                on='valid_time',
                direction='forward'
            ).set_index('valid_time')

            # make 60min interval clear sky, centered at bottom of the hour
            times = pd.date_range(
                start=df.index[0]+pd.Timedelta('30m'),
                end=df.index[-1]-pd.Timedelta('30m'),
                freq='60min',
                tz='UTC')
            cs = loc.get_clearsky(times, model=model_cs)

            # calculate ghi from clear sky and backfilled forecasted clear sky
            # index
            df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(times)
            out_erbs = pvlib.irradiance.erbs(
                df_60min.ghi,
                sp.zenith,
                df_60min.index,
            )
            df_60min['dni'] = out_erbs.dni
            df_60min['dhi'] = out_erbs.dhi

            # add clearsky ghi
            df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

            # add member number and point, drop dummy column
            df_60min['member'] = number
            df_60min['point'] = point
            df_60min = df_60min.drop(columns=['dummy'])

            # append
            dfs.append(df_60min)

    # convert to dataframe
    df_60min_irr = pd.concat(dfs)

    # get deterministic temp_air
    search_str = ':2t:sfc:g:0001:od:cf:enfo'

    # try n times based loosely on
    # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
    for attempts_remaining in reversed(range(attempts)):
        attempt_num = attempts - attempts_remaining
        try:
            if attempt_num == 1:
                # try downloading
                ds = FastHerbie(DATES=[init_date],
                                model='ifs',
                                product='enfo',
                                fxx=fxx_range).xarray(search_str)
            else:
                # after first attempt, set overwrite=True to overwrite
                # partial files
                ds = FastHerbie(DATES=[init_date],
                                model='ifs',
                                product='enfo',
                                fxx=fxx_range).xarray(search_str,
                                                      overwrite=True)
        except Exception:
            if attempts_remaining:
                print('attempt ' + str(attempt_num) + ' failed, pause for '
                      + str((attempt_num)**2) + ' min')
                time.sleep(60*(attempt_num)**2)
        else:
            break
    else:
        raise ValueError('download failed, ran out of attempts')

    # use pick_points for single point or list of points
    ds2 = ds.herbie.pick_points(pd.DataFrame({
                    "latitude": latitude,
                    "longitude": longitude,
                    }))

    # convert to dataframe
    df_temp = (ds2
               .to_dataframe()
               .reset_index()
               .set_index('valid_time')[['point', 't2m']])
    # add timezone
    df_temp = df_temp.tz_localize('UTC', level='valid_time')

    # convert air temperature units
    df_temp['temp_air'] = df_temp['t2m'] - 273.15

    dfs_temp_air = []
    # work through sites (points)
    if type(latitude) is float or type(latitude) is int:
        num_sites = 1
    else:
        num_sites = len(latitude)
    for point in range(num_sites):
        df = df_temp[df_temp['point'] == point].copy()

        # 60min version of data, centered at bottom of the hour
        # 1min interpolation, then 60min mean
        df_60min_temp_air = (
            df[['temp_air']]
            .resample('1min')
            .interpolate()
            .resample('60min').mean()
        )

        # make timestamps center-labeled for instantaneous pvlib modeling
        # later
        df_60min_temp_air.index = df_60min_temp_air.index + \
            pd.Timedelta('30min')
        # drop last row, since we don't have data for the last full hour
        # (just an instantaneous end point)
        df_60min_temp_air = df_60min_temp_air.iloc[:-1]

        # drop unneeded columns if they exist
        df_60min_temp_air = df_60min_temp_air.drop(['t2m'],
                                                   axis=1,
                                                   errors='ignore')

        # add member number and point, drop dummy column
        # df_60min_temp_air['member'] = pd.NA
        df_60min_temp_air['point'] = point

        # append
        dfs_temp_air.append(df_60min_temp_air)

    # concat
    df_60min_temp_air = pd.concat(dfs_temp_air)

    # final merge
    df_60min = pd.merge(df_60min_irr,
                        df_60min_temp_air,
                        on=['valid_time', 'point'])

    # add generic wind
    df_60min['wind_speed'] = 2

    return df_60min


def get_solar_forecast_ensemble(latitude, longitude, init_date, run_length,
                                lead_time_to_start=0, model='ifs',
                                attempts=2, priority=None):
    """
    Get solar resource forecasts for one or several sites using all ensemble
    members. Using `get_solar_forecast_ensemble_subset` may be fast for a
    small subset of ensemble members, e.g., much less that 25% of members.
    This function uses Herbie's FastHerbie [1]_ and pvlib [2]_. It currently
    only works with a single init_date, not a list of dates like FastHerbie
    can use. Temperature data comes from the ensemble mean, and wind speed is
    currently just a filler value of 2 m/s to save time.

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

    lead_time_to_start : int, optional
        Number of hours between init_date (initialization) and
        the first forecasted interval.

    model : string, default 'ifs'
        Forecast model. Can be ECMWF IFS ('ifs'), ECMWF AIFS ('aifs'), or NOAA
        GEFS ('gefs').

    attempts : int, optional
        Number of times to try getting forecast data. The function will pause
        for n^2 minutes after each n attempt, e.g., 1 min after the first
        attempt, 4 minutes after the second, etc.

    priority : list or string
        List of model sources to get the data in the order of download
        priority, or string for a single source. See Herbie docs.
        Typical values would be 'aws' or 'google'.

    Returns
    -------
    data : pandas.DataFrane
        timeseries forecasted solar resource data

    References
    ----------

    .. [1] `Blaylock, B. K. (YEAR). Herbie: Retrieve Numerical Weather
        Prediction Model Data (Version 20xx.x.x) [Computer software].
        <https://doi.org/10.5281/zenodo.4567540>`_
    .. [2] `Anderson, K., et al. “pvlib python: 2023 project update.” Journal
        of Open Source Software, 8(92), 5994, (2023).
        <http://dx.doi.org/10.21105/joss.05994>`_
    """

    # set clear sky model. could be an input variable at some point
    model_cs = 'haurwitz'

    # check model
    if (
        model.casefold() != ('ifs').casefold() and
        model.casefold() != ('aifs').casefold() and
        model.casefold() != ('gefs').casefold()
    ):
        raise ValueError(('model must be ifs, aifs, or gefs, you entered '
                          + model))

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    num_sites = len(latitude)

    # get model-specific Herbie inputs, except product and search string,
    # which are unique for the ensemble
    init_date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start, model)

    # ifs/aifs workflow
    if model == 'ifs' or model == 'aifs':
        # get GHI data for all IFS ensemble members (not the mean)
        # search for ":ssrd:sfc:" and NOT ":ssrd:sfc:g"
        # (the "g" is right after sfc if there is no member number)
        # regex based on https://superuser.com/a/1335688
        search_str = '^(?=.*:ssrd:sfc:)(?:(?!:ssrd:sfc:g).)*$'

        # try n times based loosely on
        # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
        for attempts_remaining in reversed(range(attempts)):
            attempt_num = attempts - attempts_remaining
            try:
                if attempt_num == 1:
                    # try downloading
                    FH = FastHerbie(DATES=[init_date],
                                    model=model,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority)
                    FH.download(search_str)
                    ds = FH.xarray(search_str, remove_grib=False)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    # try downloading
                    FH = FastHerbie(DATES=[init_date],
                                    model=model,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority)
                    FH.download(search_str, overwrite=True)
                    ds = FH.xarray(search_str, remove_grib=False)
            except Exception:
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
            else:
                break
        else:
            raise ValueError('download failed, ran out of attempts')

        # use pick_points for single point or list of points
        ds2 = ds.herbie.pick_points(pd.DataFrame({
                        "latitude": latitude,
                        "longitude": longitude,
                        }))
        # convert to dataframe
        df_temp = (ds2
                   .to_dataframe()
                   .reset_index()
                   .set_index('valid_time')[['number', 'point', 'ssrd',
                                             'time']])
        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')
        # rename ssrd, init_time
        df_temp = df_temp.rename(columns={'ssrd': 'sdswrf',
                                          'time': 'init_time'})

        # work through sites (points) and members
        if type(latitude) is float or type(latitude) is int:
            num_sites = 1
        else:
            num_sites = len(latitude)
        member_list = df_temp['number'].unique()
        dfs = []
        for number in member_list:
            for point in range(num_sites):
                df = df_temp[(df_temp['point'] == point) &
                             (df_temp['number'] == number)].copy()

                loc = pvlib.location.Location(
                    latitude=latitude[point],
                    longitude=longitude[point],
                    tz=df.index.tz
                    )

                # convert cumulative J/m^s to average W/m^2
                df['ghi'] = (df['sdswrf'].diff() /
                             df.index.diff().seconds.values)

                # make 1min interval clear sky data covering our time range
                times = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq='1min',
                    tz='UTC')
                cs = loc.get_clearsky(times, model=model_cs)

                # calculate average CS ghi over the intervals from the forecast
                # based on list comprehension example in
                # https://stackoverflow.com/a/55724134/27574852
                ghi = cs['ghi']
                dates = df.index
                ghi_clear = [
                    ghi.loc[(ghi.index > dates[i]) & (ghi.index <= dates[i+1])]
                    .mean() for i in range(len(dates) - 1)
                    ]

                # write to df and calculate clear sky index of ghi
                df['ghi_clear'] = [np.nan] + ghi_clear
                df['ghi_csi'] = df['ghi'] / df['ghi_clear']

                # avoid divide by zero issues
                df.loc[df['ghi'] == 0, 'ghi_csi'] = 0

                # make a dummy column
                df['dummy'] = 0

                # 60min version of data, centered at bottom of the hour
                # 1min interpolation, then 60min mean
                df_60min = (
                    df['dummy']
                    .resample('1min')
                    .interpolate()
                    .resample('60min').mean()
                )
                # make timestamps center-labeled for instantaneous pvlib
                # modeling later
                df_60min.index = df_60min.index + pd.Timedelta('30min')
                # drop last row, since we don't have data for the last full
                # hour (just an instantaneous end point)
                df_60min = df_60min.iloc[:-1]
                # "backfill" ghi csi
                # merge based on nearest index from 60min version looking
                # forward in 3hr version
                df_60min = pd.merge_asof(
                    left=df_60min,
                    right=df[['ghi_csi', 'init_time']],
                    on='valid_time',
                    direction='forward'
                ).set_index('valid_time')

                # make 60min interval clear sky, centered at bottom of the hour
                times = pd.date_range(
                    start=df.index[0]+pd.Timedelta('30m'),
                    end=df.index[-1]-pd.Timedelta('30m'),
                    freq='60min',
                    tz='UTC')
                cs = loc.get_clearsky(times, model=model_cs)

                # calculate ghi from clear sky and backfilled forecasted clear
                # sky index
                df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

                # dni and dhi using pvlib erbs. could also DIRINT or
                # erbs-driesse
                sp = loc.get_solarposition(times)
                out_erbs = pvlib.irradiance.erbs(
                    df_60min.ghi,
                    sp.zenith,
                    df_60min.index,
                )
                df_60min['dni'] = out_erbs.dni
                df_60min['dhi'] = out_erbs.dhi

                # add clearsky ghi
                df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

                # add member number and point, drop dummy column
                df_60min['member'] = number
                df_60min['point'] = point
                df_60min = df_60min.drop(columns=['dummy'])

                # append
                dfs.append(df_60min)

        # convert to dataframe
        df_60min_irr = pd.concat(dfs)

        # get deterministic temp_air
        search_str = ':2t:sfc:g:0001:od:cf:enfo'

        # try n times based loosely on
        # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
        for attempts_remaining in reversed(range(attempts)):
            attempt_num = attempts - attempts_remaining
            try:
                if attempt_num == 1:
                    # try downloading
                    FH = FastHerbie(DATES=[init_date],
                                    model='ifs',
                                    product='enfo',
                                    fxx=fxx_range)
                    FH.download(search_str)
                    ds = FH.xarray(search_str, remove_grib=False)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    FH = FastHerbie(DATES=[init_date],
                                    model='ifs',
                                    product='enfo',
                                    fxx=fxx_range)
                    FH.download(search_str, overwrite=True)
                    ds = FH.xarray(search_str, remove_grib=False)
            except Exception:
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
            else:
                break
        else:
            raise ValueError('download failed, ran out of attempts')

        # use pick_points for single point or list of points
        ds2 = ds.herbie.pick_points(pd.DataFrame({
                        "latitude": latitude,
                        "longitude": longitude,
                        }))

        # convert to dataframe
        df_temp = (ds2
                   .to_dataframe()
                   .reset_index()
                   .set_index('valid_time')[['point', 't2m']])
        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')

        # convert air temperature units
        df_temp['temp_air'] = df_temp['t2m'] - 273.15

        dfs_temp_air = []
        # work through sites (points)
        if type(latitude) is float or type(latitude) is int:
            num_sites = 1
        else:
            num_sites = len(latitude)
        for point in range(num_sites):
            df = df_temp[df_temp['point'] == point].copy()

            # 60min version of data, centered at bottom of the hour
            # 1min interpolation, then 60min mean
            df_60min_temp_air = (
                df[['temp_air']]
                .resample('1min')
                .interpolate()
                .resample('60min').mean()
            )

            # make timestamps center-labeled for instantaneous pvlib modeling
            # later
            df_60min_temp_air.index = df_60min_temp_air.index + \
                pd.Timedelta('30min')
            # drop last row, since we don't have data for the last full hour
            # (just an instantaneous end point)
            df_60min_temp_air = df_60min_temp_air.iloc[:-1]

            # drop unneeded columns if they exist
            df_60min_temp_air = df_60min_temp_air.drop(['t2m'],
                                                       axis=1,
                                                       errors='ignore')

            # add member number and point, drop dummy column
            # df_60min_temp_air['member'] = pd.NA
            df_60min_temp_air['point'] = point

            # append
            dfs_temp_air.append(df_60min_temp_air)

        # concat
        df_60min_temp_air = pd.concat(dfs_temp_air)

        # final merge
        df_60min = pd.merge(df_60min_irr,
                            df_60min_temp_air,
                            on=['valid_time', 'point'])

        # add generic wind
        df_60min['wind_speed'] = 2

    elif model == 'gefs':
        search_str = 'DSWRF'
        # list of GEFS ensemble members, e.g., 'p01', 'p02', etc.
        num_members = 30
        member_list = [f"p{x:02d}" for x in range(1, num_members+1)]

        dfs = []
        for x in range(0, num_members):
            # try n times based loosely on
            # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
            for attempts_remaining in reversed(range(attempts)):
                attempt_num = attempts - attempts_remaining
                try:
                    if attempt_num == 1:
                        # try downloading
                        FH = FastHerbie(DATES=[init_date],
                                        model=model,
                                        product=product,
                                        fxx=fxx_range,
                                        member=member_list[x])
                        FH.download(search_str)
                        ds = FH.xarray(search_str, remove_grib=False)
                    else:
                        # after first attempt, set overwrite=True to overwrite
                        # partial files
                        FH = FastHerbie(DATES=[init_date],
                                        model=model,
                                        product=product,
                                        fxx=fxx_range,
                                        member=member_list[x])
                        FH.download(search_str, overwrite=True)
                        ds = FH.xarray(search_str, remove_grib=False)
                except Exception:
                    if attempts_remaining:
                        print('attempt ' + str(attempt_num) + ' failed'
                              + ', pause for ' + str((attempt_num)**2)
                              + ' min')
                        time.sleep(60*(attempt_num)**2)
                else:
                    break
            else:
                raise ValueError('download failed, ran out of attempts')

            # use pick_points for single point or list of points
            ds2 = ds.herbie.pick_points(pd.DataFrame({
                            "latitude": latitude,
                            "longitude": longitude,
                            }))
            # convert to dataframe
            df_temp = (ds2
                       .to_dataframe()
                       .reset_index()
                       .set_index('valid_time')[['number',
                                                 'point',
                                                 'sdswrf',
                                                 'time']])
            # add timezone
            df_temp = df_temp.tz_localize('UTC', level='valid_time')
            # rename init_time
            df_temp = df_temp.rename(columns={'time': 'init_time'})

            # work through sites (points) and members
            if type(latitude) is float or type(latitude) is int:
                num_sites = 1
            else:
                num_sites = len(latitude)

            for point in range(num_sites):
                df = df_temp[(df_temp['point'] == point)].copy()

                loc = pvlib.location.Location(
                    latitude=latitude[point],
                    longitude=longitude[point],
                    tz=df.index.tz
                    )

                # for gfs ghi: we have to "unmix" the rolling average
                # irradiance that resets every 6 hours
                mixed = df[['sdswrf']].copy()
                mixed['hour'] = mixed.index.hour
                mixed['hour'] = mixed.index.hour
                mixed['hour_of_mixed_period'] = ((mixed['hour'] - 1) % 6) + 1
                mixed['sdswrf_prev'] = mixed['sdswrf'].shift(
                    periods=1,
                    fill_value=0
                    )
                mixed['int_len'] = (mixed.index.diff()
                                    .total_seconds().values) / 3600

                # set the first interval length:
                mixed.loc[mixed.index[0], 'int_len'] = 3
                unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                           - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                           * mixed['sdswrf_prev']) / mixed['int_len'])
                df['ghi'] = unmixed

                # make 1min interval clear sky data covering our time range
                times = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq='1min',
                    tz='UTC')

                cs = loc.get_clearsky(times, model=model_cs)

                # calculate average CS ghi over the intervals from the forecast
                # based on list comprehension example in
                # https://stackoverflow.com/a/55724134/27574852
                ghi = cs['ghi']
                dates = df.index
                ghi_clear = [
                    ghi.loc[(ghi.index > dates[i]) & (ghi.index <= dates[i+1])]
                    .mean() for i in range(len(dates) - 1)
                    ]

                # write to df and calculate clear sky index of ghi
                df['ghi_clear'] = [np.nan] + ghi_clear
                df['ghi_csi'] = df['ghi'] / df['ghi_clear']

                # avoid divide by zero issues
                df.loc[df['ghi'] == 0, 'ghi_csi'] = 0

                # make a dummy column
                df['dummy'] = 0

                # 60min version of data, centered at bottom of the hour
                # 1min interpolation, then 60min mean
                df_60min = (
                    df[['dummy']]
                    .resample('1min')
                    .interpolate()
                    .resample('60min').mean()
                )
                # make timestamps center-labeled for instantaneous pvlib
                # modeling later
                df_60min.index = df_60min.index + pd.Timedelta('30min')
                # drop last row, since we don't have data for the last full
                # hour (just an instantaneous end point)
                df_60min = df_60min.iloc[:-1]
                # "backfill" ghi csi
                # merge based on nearest index from 60min version looking
                # forward in 3hr version
                df_60min = pd.merge_asof(
                    left=df_60min,
                    right=df[['ghi_csi', 'init_time']],
                    on='valid_time',
                    direction='forward'
                ).set_index('valid_time')

                # make 60min interval clear sky, centered at bottom of the hour
                times = pd.date_range(
                    start=df.index[0]+pd.Timedelta('30m'),
                    end=df.index[-1]-pd.Timedelta('30m'),
                    freq='60min',
                    tz='UTC')
                cs = loc.get_clearsky(times, model=model_cs)

                # calculate ghi from clear sky and backfilled forecasted clear
                # sky index
                df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

                # dni and dhi using pvlib erbs. could also DIRINT or
                # erbs-driesse
                sp = loc.get_solarposition(times)
                out_erbs = pvlib.irradiance.erbs(
                    df_60min.ghi,
                    sp.zenith,
                    df_60min.index,
                )
                df_60min['dni'] = out_erbs.dni
                df_60min['dhi'] = out_erbs.dhi

                # add clearsky ghi
                df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

                # add member number and point, drop dummy column
                df_60min['member'] = ds['number'].values
                df_60min['point'] = point
                df_60min = df_60min.drop(columns=['dummy'])

                # append
                dfs.append(df_60min)

        # convert to dataframe
        df_60min_irr = pd.concat(dfs)

        # get deterministic temp_air
        search_str = ':TMP:2 m above'
        member = 'c00'  # use the control member

        # try n times based loosely on
        # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
        for attempts_remaining in reversed(range(attempts)):
            attempt_num = attempts - attempts_remaining
            try:
                if attempt_num == 1:
                    # try downloading
                    FH = FastHerbie(DATES=[init_date],
                                    model=model,
                                    product=product,
                                    fxx=fxx_range,
                                    member=member)
                    FH.download(search_str)
                    ds = FH.xarray(search_str, remove_grib=False)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    FH = FastHerbie(DATES=[init_date],
                                    model=model,
                                    product=product,
                                    fxx=fxx_range,
                                    member=member)
                    FH.download(search_str, overwrite=True)
                    ds = FH.xarray(search_str, remove_grib=False)
            except Exception:
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
            else:
                break
        else:
            raise ValueError('download failed, ran out of attempts')

        # use pick_points for single point or list of points
        ds2 = ds.herbie.pick_points(pd.DataFrame({
                        "latitude": latitude,
                        "longitude": longitude,
                        }))
        # convert to dataframe
        df_temp = (ds2
                   .to_dataframe()
                   .reset_index()
                   .set_index('valid_time')[['point', 't2m', 'time']])
        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')
        # rename init_time
        df_temp = df_temp.rename(columns={'time': 'init_time'})

        # convert air temperature units
        df_temp['temp_air'] = df_temp['t2m'] - 273.15

        # work through sites (points)
        if type(latitude) is float or type(latitude) is int:
            num_sites = 1
        else:
            num_sites = len(latitude)

        dfs_temp_air = []
        for point in range(num_sites):
            df = df_temp[(df_temp['point'] == point)].copy()

            # 60min version of data, centered at bottom of the hour
            # 1min interpolation, then 60min mean
            df_60min_temp_air = (
                df[['temp_air']]
                .resample('1min')
                .interpolate()
                .resample('60min').mean()
            )

            # make timestamps center-labeled for instantaneous pvlib modeling
            # later
            df_60min_temp_air.index = df_60min_temp_air.index + \
                pd.Timedelta('30min')
            # drop last row, since we don't have data for the last full hour
            # (just an instantaneous end point)
            df_60min_temp_air = df_60min_temp_air.iloc[:-1]

            # drop unneeded columns if they exist
            df_60min_temp_air = df_60min_temp_air.drop(['t2m'],
                                                       axis=1,
                                                       errors='ignore')

            # add member number and point, drop dummy column
            # df_60min_temp_air['member'] = pd.NA
            df_60min_temp_air['point'] = point

            # append
            dfs_temp_air.append(df_60min_temp_air)

        # concat
        df_60min_temp_air = pd.concat(dfs_temp_air)

        # final merge
        df_60min = pd.merge(df_60min_irr,
                            df_60min_temp_air,
                            on=['valid_time', 'point'])

        # add generic wind
        df_60min['wind_speed'] = 2

    return df_60min
