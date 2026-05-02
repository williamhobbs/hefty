import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie, FastHerbie
import pvlib
import time
from hefty.utilities import model_input_formatter

try:
    import cdsapi
except ImportError:
    _has_cdsapi = False
else:
    _has_cdsapi = True
import os
import tomllib


def get_solar_forecast(latitude, longitude, init_date, run_length,
                       lead_time_to_start=0, model='gfs', member='avg',
                       attempts=2, hrrr_hour_middle=True,
                       hrrr_coursen_window=None, priority=None,
                       cams_api_key=None, cams_area=None):
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
        Model initialization datetime. Note that this should be UTC and on the
        hour for the models currently available with hefty, and most models
        don't initialize every hour. See
        :py:func:`hefty.utilities.adjust_forecast_datetimes` for help
        determining appropriate init_date values.

    run_length : int
        Length of the forecast in hours - number of hours forecasted

    lead_time_to_start : int, optional
        Number of hours between init_date (initialization) and the first
        forecasted interval. NOAA GFS data goes out 384 hours, so run_length
        + lead_time_to_start must be less than or equal to 384.

    model : string, default 'gfs'
        Forecast model. Default is NOAA GFS ('gfs'), but can also be ECMWF IFS
        single ('ifs'), ECMWF AIFS single ('aifs'), NOAA HRRR ('hrrr'), or
        NOAA GEFS ('gefs') (a single member from the ensemble). ECMWF CAMS
        ('cams') is an experimental option. It requires cdsapi to be installed
        and a CDS API key to be passed via the 'cams_api_key' parameter.

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

    cams_api_key : string
        Climate Data Store (CDS) API key, which is required for the 'cams'
        model option. See https://ads.atmosphere.copernicus.eu/how-to-api.

    cams_area : list, optional
        List of latitude and logitude coordinates defining the North, West,
        South, and East corners of the area to be covered when using 'cams'.
        For example, [50, -125, 20, -65] approximately covers CONUS.

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
    model_cs = 'simplified_solis'
    model_cs_kwargs = {
        'aod700': 0.05,
        'precipitable_water': 0.5,
    }
    # minimum cosine of zenith, same default used in pvlib.irradiance
    # functions. Could be an input variable at some point
    min_cos_zenith = 0.065

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    num_sites = len(latitude)
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # check if init_date is top of hour
    if init_date != init_date.floor('1h'):
        raise ValueError(f'init_date must be on the hour, e.g., '
                         f'{init_date.floor('1h')}, not {init_date}. '
                         'Consider using init_date.floor("1h") or '
                         'similar')

    # get model-specific Herbie inputs
    date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start, model)

    # get NWP data as dataframe
    if model != 'cams':
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

            # merge - override avoids hight conflict between 2m temp and 10m
            # wind
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

    elif model == 'cams':
        if not _has_cdsapi:
            raise ImportError(('cdsapi is required to use cams, '
                               'e.g., with `pip install cdsapi`.'))
        # directory path for saving cams files
        # check to see if a custom herbie config path has been set
        if os.environ.get('HERBIE_CONFIG_PATH') is not None:
            herbie_config_path = os.environ['HERBIE_CONFIG_PATH']
        else:
            # otherwise, use default herbie config path
            herbie_config_path = os.path.join(
                os.path.expanduser('~'), '.config', 'herbie', 'config.toml')

        with open(herbie_config_path, "rb") as f:
            config_data = tomllib.load(f)
        # use 'cams' subfolder
        cams_dir_path = os.path.join(config_data['default']['save_dir'],
                                     'cams')
        # if cams folder doesn't exist, make it
        if not os.path.exists(cams_dir_path):
            os.makedirs(cams_dir_path)

        if cams_area is None:
            cams_area = [90, -180, -90, 180]

        ts = pd.Timestamp(date)
        date_str = ts.strftime('%Y-%m-%d')
        time_str = ts.strftime('%H:%M')
        filename = (
            f'cams.{ts.strftime('%Y%m%d')}.{ts.strftime('%H')}z.'
            f'{min(fxx_range):0{3}}.{max(fxx_range):0{3}}.'
            f'{cams_area[0]}N_{cams_area[1]}E_{cams_area[2]}E_{cams_area[3]}N'
            f'.grib'
            )
        download_path_file = os.path.join(
            cams_dir_path,
            filename)

        if os.path.exists(download_path_file):  # load file if it exists
            ds = xr.load_dataset(download_path_file)
        else:  # otherwise download file
            request = {
                'variable': ['surface_solar_radiation_downward_clear_sky',
                             'surface_solar_radiation_downwards',
                             '2m_temperature',
                             '10m_u_component_of_wind',
                             '10m_v_component_of_wind',
                             'direct_solar_radiation',
                             'clear_sky_direct_solar_radiation_at_surface',
                             ],
                'date': [date_str],
                'time': [time_str],
                'leadtime_hour': [str(i) for i in fxx_range],
                'type': ['forecast'],
                'data_format': 'grib',
                'area': cams_area  # [N, W, S, E]
            }
            URL = 'https://ads.atmosphere.copernicus.eu/api'
            client = cdsapi.Client(url=URL, key=cams_api_key)
            dataset = "cams-global-atmospheric-composition-forecasts"
            client.retrieve(dataset, request).download(download_path_file)
            ds = xr.load_dataset(download_path_file)

        # convert wind
        ds.herbie.with_wind('speed')
        # rename 'ssrd' to 'sdswrf', 'dsrp' to 'vbdsf'
        ds = ds.rename({'ssrd': 'sdswrf',
                        'dsrp': 'vbdsf'})

        # use pick_points for single point or list of points
        method = 'weighted'
        # point can be at most about 1.5 grid cells away from furthest of 4
        # nearest neighbors (far corner of region). At ~45km max per 0.4 deg
        # cell, that's sqrt(2*(45*1.5)^2) = ~96km
        max_distance = 96
        # use_cached_tree = True # hold off on saving a BallTree for now...
        # tree_name = 'cams_balltree'
        ds_temp = ds.herbie.pick_points(
                        pd.DataFrame(
                            {
                                "latitude": latitude,
                                "longitude": longitude,
                            }
                        ),
                        method=method,
                        max_distance=max_distance,
                        # use_cached_tree=use_cached_tree,
                        # tree_name=tree_name,
                    )

        # convert to dataframe
        df_temp = ds_temp.to_dataframe()
        if method == 'weighted':
            # filter to a single "k"
            df_temp = (df_temp[df_temp.index.isin([0], level='k')].
                       droplevel(level='k'))

        # reset index
        df_temp = df_temp.reset_index().set_index('valid_time')

        # add timezone
        df_temp = df_temp.tz_localize('UTC', level='valid_time')
        # rename wind speed
        df_temp = df_temp.rename(columns={
            'si10': 'wind_speed',
            })
        # convert air temperature units
        df_temp['temp_air'] = df_temp['t2m'] - 273.15

        # keep only select columns
        df_temp = df_temp[['point', 'sdswrf', 'wind_speed', 'temp_air',
                           'ssrdc', 'vbdsf', 'cdir', 'time']].copy()

        # make index valid_time and point
        df_temp = df_temp.reset_index().set_index(['valid_time',
                                                   'point'])

    # work through sites
    dfs = {}  # empty list of dataframes
    for j in range(num_sites):
        df = df_temp[df_temp.index.get_level_values('point') == j]
        df = df.droplevel('point')

        loc = pvlib.location.Location(
            latitude=latitude[j],
            longitude=longitude[j],
            tz=df.index.tz
            )

        if model in {'gfs', 'gefs'}:
            # for gfs and gefs ghi: we have to "unmix" the rolling average
            # irradiance that resets every 6 hours
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
            if model == 'gfs' and lead_time_to_start >= 120:
                mixed.loc[mixed.index[0], 'int_len'] = 1
            else:
                mixed.loc[mixed.index[0], 'int_len'] = 3
            unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                        - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                        * mixed['sdswrf_prev']) / mixed['int_len'])
            df['ghi'] = unmixed

        elif model in {'ifs', 'aifs', 'cams'}:
            # for ifs ghi: cumulative J/m^s to average W/m^2 over the interval
            # ending at the valid time. calculate difference in measurement
            # over diff in time to get avg J/s/m^2 = W/m^2
            df['ghi'] = df['sdswrf'].diff() / df.index.diff().seconds.values

            if model == 'cams':
                df['dni'] = df['vbdsf'].diff() / df.index.diff().seconds.values
                df['ghi_clear_nwp'] = (df['ssrdc'].diff() /
                                       df.index.diff().seconds.values)
                df['direct_horiz_clear'] = (df['cdir'].diff() /
                                            df.index.diff().seconds.values)

        elif model == 'hrrr':
            df['ghi'] = df['sdswrf']
            df['dni'] = df['vbdsf']

        if model in {'gfs', 'gefs', 'ifs', 'aifs'}:
            # make 1min interval clear sky data covering our time range
            times = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq='1min',
                tz='UTC')

            # calculate clear sky ghi with pvlib
            cs = loc.get_clearsky(times, model=model_cs, **model_cs_kwargs)

            # calculate average CS ghi over the intervals from the forecast
            # based on list comprehension example in
            # https://stackoverflow.com/a/55724134/27574852
            ghi = cs['ghi']
            dates = df.index
            ghi_clear = [
                ghi.loc[(ghi.index > dates[i]) & (ghi.index <= dates[i+1])]
                .mean() for i in range(len(dates) - 1)
                ]

            # write to df
            df['ghi_clear'] = [np.nan] + ghi_clear

            # calculate clear sky index of ghi, dni
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
                right=df['ghi_csi'],
                on='valid_time',
                direction='forward'
            ).set_index('valid_time')

            # make 60min interval clear sky, centered at bottom of the hour
            times = pd.date_range(
                start=df.index[0]+pd.Timedelta('30m'),
                end=df.index[-1]-pd.Timedelta('30m'),
                freq='60min',
                tz='UTC')
            cs = loc.get_clearsky(times, model=model_cs, **model_cs_kwargs)

            # calculate ghi from clear sky and backfilled forecasted clear sky
            # index
            df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

            # clip to avoid occasional small negative ghi in GEFS, see GH #35
            df_60min['ghi'] = df_60min['ghi'].clip(lower=0)

            # dni and dhi using pvlib erbs. could also DIRINT or
            # erbs-driesse
            sp = loc.get_solarposition(times)
            out_erbs = pvlib.irradiance.erbs(
                df_60min['ghi'],
                sp['zenith'],
                df_60min.index,
            )
            df_60min['dni'] = out_erbs['dni']
            df_60min['dhi'] = out_erbs['dhi']

            # add clearsky ghi
            df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

            dfs[j] = df_60min

        elif model == 'cams':
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

            # adjust timestamps to center of interval
            df.index = df.index - pd.Timedelta('30min')

            # direct horiz clear to dni_clear
            sp = loc.get_solarposition(df.index)
            cos_zenith = np.maximum(np.cos(np.deg2rad(sp['apparent_zenith'])),
                                    min_cos_zenith)
            df['dni_clear'] = (df['direct_horiz_clear'] / cos_zenith)
            df_60min = df_60min.join(df.drop(['temp_air', 'wind_speed'],
                                             axis=1))

            # calculate dhi from ghi, dni, solar position
            df_60min['dhi'] = (df_60min['ghi'] -
                               (df_60min['dni'] * cos_zenith))

            # clean up dataframe
            df_60min['ghi_clear'] = df_60min['ghi_clear_nwp']
            df_60min = df_60min[['temp_air', 'wind_speed', 'ghi', 'dni', 'dhi',
                                 'ghi_clear', 'dni_clear', 'time',
                                 'direct_horiz_clear']]

            dfs[j] = df_60min.copy()

        elif model == 'hrrr':
            if hrrr_hour_middle is True:
                # clear sky index
                times = df.index
                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)
                df['csi_ghi'] = df['ghi'] / cs['ghi']
                df['csi_dni'] = df['dni'] / cs['dni']
                # avoid divide by zero issues
                df.loc[df['ghi'] == 0, 'csi_ghi'] = 0
                df.loc[df['dni'] == 0, 'csi_dni'] = 0

                # make 1min interval clear sky data covering our time range
                times = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq='1min',
                    tz='UTC')

                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)
                # calculate 1min interpolated temp_air, wind_speed, csi
                df_01min = (
                    df[['temp_air', 'wind_speed', 'csi_ghi', 'csi_dni']]
                    .resample('1min')
                    .interpolate()
                )
                # add ghi_clear
                df_01min['ghi_clear'] = cs['ghi']
                df_01min['dni_clear'] = cs['dni']
                # calculate hour averages centered labelled at bottom of the
                # hour
                df_60min = df_01min.resample('1h').mean()
                df_60min.index = df_60min.index + pd.Timedelta('30min')
                # calculate new ghi
                df_60min['ghi'] = df_60min['csi_ghi'] * df_60min['ghi_clear']
                df_60min['dni'] = df_60min['csi_dni'] * df_60min['dni_clear']

            else:
                df_60min = df.copy()

            # calculate dhi from ghi, dni, solar position
            sp = loc.get_solarposition(df_60min.index)
            cos_zenith = np.maximum(np.cos(np.deg2rad(sp['apparent_zenith'])),
                                    min_cos_zenith)
            df_60min['dhi'] = (df_60min['ghi'] -
                               (df_60min['dni'] * cos_zenith))

            # add clearsky ghi
            cs = loc.get_clearsky(df_60min.index, model=model_cs,
                                  **model_cs_kwargs)
            df_60min['ghi_clear'] = cs['ghi']
            df_60min['dni_clear'] = cs['dni']

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
                            lead_time_to_start=0, model='gfs', member='avg',
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
        NOAA GEFS ('gefs') (a single member from the ensemble). ECMWF CAMS
        ('cams') is NOT available with this function. For CAMS access, see
        :py:func:`hefty.solar.get_solar_forecast`.

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
    model_cs = 'simplified_solis'
    model_cs_kwargs = {
        'aod700': 0.05,
        'precipitable_water': 0.5,
    }
    # minimum cosine of zenith, same default used in pvlib.irradiance
    # functions. Could be an input variable at some point
    min_cos_zenith = 0.065

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    num_sites = len(latitude)
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # check if init_date is top of hour
    if init_date != init_date.floor('1h'):
        raise ValueError(f'init_date must be on the hour, e.g., '
                         f'{init_date.floor('1h')}, not {init_date}. '
                         'Consider using init_date.floor("1h") or '
                         'similar')

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
            except Exception as e:
                print(e)
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
                else:
                    raise ValueError(f'download failed, ran out of attempts '
                                     f'with error: {e}')
            else:
                break

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
    # rename 'ssrd' to 'sdswrf' in ifs/aifs
    if model == 'ifs' or model == 'aifs':
        df_temp = i[-1].to_dataframe()[['valid_time', 'ssrd', 't2m', 'si10']]
        df_temp = df_temp.rename(columns={'ssrd': 'sdswrf'})
    elif model == 'hrrr':
        df_temp = i[-1].to_dataframe()[['valid_time', 'sdswrf', 'vbdsf',
                                        't2m', 'si10']]
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
    for j in range(num_sites):
        df = df_temp[df_temp.index.get_level_values('point') == j]
        df = df.droplevel('point')

        loc = pvlib.location.Location(
            latitude=latitude[j],
            longitude=longitude[j],
            tz=df.index.tz
            )

        if model in {'gfs', 'gefs'}:
            # for gfs and gefs ghi: we have to "unmix" the rolling average
            # irradiance that resets every 6 hours
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
            if model == 'gfs' and lead_time_to_start >= 120:
                mixed.loc[mixed.index[0], 'int_len'] = 1
            else:
                mixed.loc[mixed.index[0], 'int_len'] = 3
            unmixed = ((mixed['hour_of_mixed_period'] * mixed['sdswrf']
                        - (mixed['hour_of_mixed_period'] - mixed['int_len'])
                        * mixed['sdswrf_prev']) / mixed['int_len'])
            df['ghi'] = unmixed

        elif model == 'ifs' or model == 'aifs':
            # for ifs ghi: cumulative J/m^s to average W/m^2 over the interval
            # ending at the valid time. calculate difference in measurement
            # over diff in time to get avg J/s/m^2 = W/m^2
            df['ghi'] = df['sdswrf'].diff() / df.index.diff().seconds.values

        elif model == 'hrrr':
            df['ghi'] = df['sdswrf']
            df['dni'] = df['vbdsf']

        if model in {'gfs', 'gefs', 'ifs', 'aifs'}:
            # make 1min interval clear sky data covering our time range
            times = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq='1min',
                tz='UTC')

            cs = loc.get_clearsky(times, model=model_cs, **model_cs_kwargs)

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
                right=df['ghi_csi'],
                on='valid_time',
                direction='forward'
            ).set_index('valid_time')

            # make 60min interval clear sky, centered at bottom of the hour
            times = pd.date_range(
                start=df.index[0]+pd.Timedelta('30m'),
                end=df.index[-1]-pd.Timedelta('30m'),
                freq='60min',
                tz='UTC')
            cs = loc.get_clearsky(times, model=model_cs, **model_cs_kwargs)

            # calculate ghi from clear sky and backfilled forecasted clear sky
            # index
            df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

            # clip to avoid occasional small negative ghi in GEFS, see GH #35
            df_60min['ghi'] = df_60min['ghi'].clip(lower=0)

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(times)
            out_erbs = pvlib.irradiance.erbs(
                df_60min['ghi'],
                sp['zenith'],
                df_60min.index,
            )
            df_60min['dni'] = out_erbs['dni']
            df_60min['dhi'] = out_erbs['dhi']

            # add clearsky ghi
            df_60min['ghi_clear'] = df_60min['ghi'] / df_60min['ghi_csi']

            dfs[j] = df_60min

        elif model == 'hrrr':
            if hrrr_hour_middle is True:
                # clear sky index
                times = df.index
                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)
                df['csi_ghi'] = df['ghi'] / cs['ghi']
                df['csi_dni'] = df['dni'] / cs['dni']
                # avoid divide by zero issues
                df.loc[df['ghi'] == 0, 'csi_ghi'] = 0
                df.loc[df['dni'] == 0, 'csi_dni'] = 0

                # make 1min interval clear sky data covering our time range
                times = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq='1min',
                    tz='UTC')

                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)
                # calculate 1min interpolated temp_air, wind_speed, csi
                df_01min = (
                    df[['temp_air', 'wind_speed', 'csi_ghi', 'csi_dni']]
                    .resample('1min')
                    .interpolate()
                )
                # add ghi_clear
                df_01min['ghi_clear'] = cs['ghi']
                df_01min['dni_clear'] = cs['dni']
                # calculate hour averages centered labelled at bottom of the
                # hour
                df_60min = df_01min.resample('1h').mean()
                df_60min.index = df_60min.index + pd.Timedelta('30min')
                # calculate new ghi
                df_60min['ghi'] = df_60min['csi_ghi'] * df_60min['ghi_clear']
                df_60min['dni'] = df_60min['csi_dni'] * df_60min['dni_clear']

            else:
                df_60min = df.copy()

            # calculate dhi from ghi, dni, solar position
            sp = loc.get_solarposition(df_60min.index)
            cos_zenith = np.maximum(np.cos(np.deg2rad(sp['apparent_zenith'])),
                                    min_cos_zenith)
            df_60min['dhi'] = (df_60min['ghi'] -
                               (df_60min['dni'] * cos_zenith))

            # add clearsky ghi
            cs = loc.get_clearsky(df_60min.index, model=model_cs,
                                  **model_cs_kwargs)
            df_60min['ghi_clear'] = cs['ghi']
            df_60min['dni_clear'] = cs['dni']

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
        Model initialization datetime. Note that this should be UTC and on the
        hour for the models currently available with hefty, and most models
        don't initialize every hour. See
        :py:func:`hefty.utilities.adjust_forecast_datetimes` for help
        determining appropriate init_date values.

    run_length : int
        Length of the forecast in hours - number of hours forecasted

    lead_time_to_start : int, optional
        Number of hours between init_date (initialization) and
        the first forecasted interval.

    model : string, default 'ifs_ens'
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
    model_cs = 'simplified_solis'
    model_cs_kwargs = {
        'aod700': 0.05,
        'precipitable_water': 0.5,
    }

    # check model
    if model.casefold() != ('ifs_ens').casefold():
        raise ValueError('model must be ifs_ens, you entered ' + model)

    model_herbie = 'ifs'  # this is the model name Herbie uses

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    num_sites = len(latitude)
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # check if init_date is top of hour
    if init_date != init_date.floor('1h'):
        raise ValueError(f'init_date must be on the hour, e.g., '
                         f'{init_date.floor('1h')}, not {init_date}. '
                         'Consider using init_date.floor("1h") or '
                         'similar')

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
                                    model=model_herbie,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority).xarray(search_str)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    ds = FastHerbie(DATES=[init_date],
                                    model=model_herbie,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority).xarray(search_str,
                                                              overwrite=True)
            except Exception as e:
                print(e)
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
                else:
                    raise ValueError(f'download failed, ran out of attempts '
                                     f'with error: {e}')
            else:
                break

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
            cs = loc.get_clearsky(times, model=model_cs, **model_cs_kwargs)

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
            cs = loc.get_clearsky(times, model=model_cs, **model_cs_kwargs)

            # calculate ghi from clear sky and backfilled forecasted clear sky
            # index
            df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

            # clip to avoid occasional small negative ghi in GEFS, see GH #35
            df_60min['ghi'] = df_60min['ghi'].clip(lower=0)

            # dni and dhi using pvlib erbs. could also DIRINT or erbs-driesse
            sp = loc.get_solarposition(times)
            out_erbs = pvlib.irradiance.erbs(
                df_60min['ghi'],
                sp['zenith'],
                df_60min.index,
            )
            df_60min['dni'] = out_erbs['dni']
            df_60min['dhi'] = out_erbs['dhi']

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
                                model=model_herbie,
                                product='enfo',
                                fxx=fxx_range,
                                priority=priority).xarray(search_str)
            else:
                # after first attempt, set overwrite=True to overwrite
                # partial files
                ds = FastHerbie(DATES=[init_date],
                                model=model_herbie,
                                product='enfo',
                                fxx=fxx_range,
                                priority=priority).xarray(search_str,
                                                          overwrite=True)
        except Exception as e:
            print(e)
            if attempts_remaining:
                print('attempt ' + str(attempt_num) + ' failed, pause for '
                      + str((attempt_num)**2) + ' min')
                time.sleep(60*(attempt_num)**2)
            else:
                raise ValueError(f'download failed, ran out of attempts '
                                 f'with error: {e}')
        else:
            break

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
                                lead_time_to_start=0, model='ifs_ens',
                                attempts=2, priority=None):
    """
    Get solar resource forecasts for one or several sites using all ensemble
    members. Using `get_solar_forecast_ensemble_subset` may be fast for a
    small subset of ensemble members, e.g., much less that 25% of members.
    This function uses Herbie's FastHerbie [1]_ and pvlib [2]_. It currently
    only works with a single init_date, not a list of dates like FastHerbie
    can use. Temperature data comes from the ensemble mean for GEFS, the
    control member for IFS (mean is available and could be used), and the
    first member for AIFS (mean and control do not seem to be available). Wind
    speed is currently just a filler value of 2 m/s to save time.

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
        the first forecasted interval.

    model : string, default 'ifs_ens'
        Forecast model. Can be ECMWF IFS Ensemble ('ifs_ens'), ECMWF AIFS
        Ensemble ('aifs_ens'), or NOAA GEFS ('gefs').

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
    model_cs = 'simplified_solis'
    model_cs_kwargs = {
        'aod700': 0.05,
        'precipitable_water': 0.5,
    }

    # check model
    if (
        model.casefold() != ('ifs_ens').casefold() and
        model.casefold() != ('aifs_ens').casefold() and
        model.casefold() != ('gefs').casefold()
    ):
        raise ValueError(('model must be ifs_ens, aifs_ens, or gefs, you '
                          'entered ' + model))

    # model_herbie is the model name Herbie uses
    if model == 'ifs_ens':
        model_herbie = 'ifs'
    elif model == 'aifs_ens':
        model_herbie = 'aifs'
    elif model == 'gefs':
        model_herbie = 'gefs'

    # variable formatting
    # if lat, lon are single values, convert to lists for pickpoints later
    if type(latitude) is float or type(latitude) is int:
        latitude = [latitude]
        longitude = [longitude]
    num_sites = len(latitude)
    # convert init_date to datetime
    init_date = pd.to_datetime(init_date)

    # check if init_date is top of hour
    if init_date != init_date.floor('1h'):
        raise ValueError(f'init_date must be on the hour, e.g., '
                         f'{init_date.floor('1h')}, not {init_date}. '
                         'Consider using init_date.floor("1h") or '
                         'similar')

    # get model-specific Herbie inputs, except product and search string,
    # which are unique for the ensemble
    init_date, fxx_range, product, search_str = model_input_formatter(
        init_date, run_length, lead_time_to_start, model)

    # ifs/aifs workflow
    if model == 'ifs_ens' or model == 'aifs_ens':
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
                                    model=model_herbie,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority)
                    FH.download(search_str)
                    ds = FH.xarray(search_str, remove_grib=False)
                    # check for missing members. if any, raise error
                    # fixes GH #28
                    # see https://github.com/williamhobbs/hefty/issues/28 for
                    # details
                    for data_var in ds.data_vars:
                        # count of valid values in each step/number
                        # combination (slicing along lat/lon plane)
                        c_v = (
                            ds.count(dim=['latitude', 'longitude'])[data_var].
                            values)
                        num_missing_members = (np.count_nonzero(c_v == 0))
                        if num_missing_members > 0:
                            # indices of steps w/ missing members
                            steps_idx = (
                                [i for i, sublist in enumerate(c_v) if 0 in
                                 sublist])
                            # fxx values
                            fxx_vals = ((ds['step'].values[steps_idx] /
                                        np.timedelta64(1, 'h')).
                                        astype(int).tolist())
                            msg = (f'{num_missing_members} members appear to '
                                   f'be missing for init_date {init_date}, fxx'
                                   f' values {fxx_vals}')
                            print(msg)
                            raise ValueError(msg)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    # try downloading
                    FH = FastHerbie(DATES=[init_date],
                                    model=model_herbie,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority)
                    FH.download(search_str, overwrite=True)
                    ds = FH.xarray(search_str, remove_grib=False)
                    # check for missing members again
                    for data_var in ds.data_vars:
                        # count of valid values in each step/number
                        # combination (slicing along lat/lon plane)
                        c_v = (
                            ds.count(dim=['latitude', 'longitude'])[data_var].
                            values)
                        num_missing_members = (np.count_nonzero(c_v == 0))
                        if num_missing_members > 0:
                            # indices of steps w/ missing members
                            steps_idx = (
                                [i for i, sublist in enumerate(c_v) if 0 in
                                 sublist])
                            # fxx values
                            fxx_vals = ((ds['step'].values[steps_idx] /
                                        np.timedelta64(1, 'h')).
                                        astype(int).tolist())
                            msg = (f'{num_missing_members} members appear to '
                                   f'be missing for init_date {init_date}, fxx'
                                   f' values {fxx_vals}')
                            print(msg)
                            raise ValueError(msg)
            except Exception as e:
                print(e)
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
                else:
                    raise ValueError(f'download failed, ran out of attempts '
                                     f'with error: {e}')
            else:
                break

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
                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)

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
                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)

                # calculate ghi from clear sky and backfilled forecasted clear
                # sky index
                df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

                # dni and dhi using pvlib erbs. could also DIRINT or
                # erbs-driesse
                sp = loc.get_solarposition(times)
                out_erbs = pvlib.irradiance.erbs(
                    df_60min['ghi'],
                    sp['zenith'],
                    df_60min.index,
                )
                df_60min['dni'] = out_erbs['dni']
                df_60min['dhi'] = out_erbs['dhi']

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

        if model == 'ifs_ens':
            # get deterministic temp_air using ifs control member
            search_str = ':2t:sfc:g:0001:od:cf:enfo'
            get_control = None
        elif model == 'aifs_ens':
            search_str = ':2t:sfc:'
            # Herbie kwarg to get control member, https://herbie.readthedocs.io/en/stable/gallery/ecmwf_models/ecmwf.html#AIFS-Ensembles
            get_control = True

        # try n times based loosely on
        # https://thingspython.wordpress.com/2021/12/05/how-to-try-something-n-times-in-python/
        for attempts_remaining in reversed(range(attempts)):
            attempt_num = attempts - attempts_remaining
            try:
                if attempt_num == 1:
                    # try downloading
                    FH = FastHerbie(DATES=[init_date],
                                    model=model_herbie,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority,
                                    get_control=get_control,
                                    )
                    FH.download(search_str)
                    ds = FH.xarray(search_str, remove_grib=False)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    FH = FastHerbie(DATES=[init_date],
                                    model=model_herbie,
                                    product='enfo',
                                    fxx=fxx_range,
                                    priority=priority,
                                    get_control=get_control,
                                    )
                    FH.download(search_str, overwrite=True)
                    ds = FH.xarray(search_str, remove_grib=False)
            except Exception as e:
                print(e)
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
                else:
                    raise ValueError(f'download failed, ran out of attempts '
                                     f'with error: {e}')
            else:
                break

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
                                        model=model_herbie,
                                        product=product,
                                        fxx=fxx_range,
                                        member=member_list[x],
                                        priority=priority)
                        FH.download(search_str)
                        ds = FH.xarray(search_str, remove_grib=False)
                    else:
                        # after first attempt, set overwrite=True to overwrite
                        # partial files
                        FH = FastHerbie(DATES=[init_date],
                                        model=model_herbie,
                                        product=product,
                                        fxx=fxx_range,
                                        member=member_list[x],
                                        priority=priority)
                        FH.download(search_str, overwrite=True)
                        ds = FH.xarray(search_str, remove_grib=False)
                except Exception as e:
                    print(e)
                    if attempts_remaining:
                        print('attempt ' + str(attempt_num) + ' failed'
                              + ', pause for ' + str((attempt_num)**2)
                              + ' min')
                        time.sleep(60*(attempt_num)**2)
                    else:
                        raise ValueError(f'download failed, ran out of '
                                         f'attempts with error: {e}')
                else:
                    break

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

                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)

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
                cs = loc.get_clearsky(times, model=model_cs,
                                      **model_cs_kwargs)

                # calculate ghi from clear sky and backfilled forecasted clear
                # sky index
                df_60min['ghi'] = cs['ghi'] * df_60min['ghi_csi']

                # clip to avoid occasional small negative ghi in GEFS, GH #35
                df_60min['ghi'] = df_60min['ghi'].clip(lower=0)

                # dni and dhi using pvlib erbs. could also DIRINT or
                # erbs-driesse
                sp = loc.get_solarposition(times)
                out_erbs = pvlib.irradiance.erbs(
                    df_60min['ghi'],
                    sp['zenith'],
                    df_60min.index,
                )
                df_60min['dni'] = out_erbs['dni']
                df_60min['dhi'] = out_erbs['dhi']

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
                                    model=model_herbie,
                                    product=product,
                                    fxx=fxx_range,
                                    member=member,
                                    priority=priority)
                    FH.download(search_str)
                    ds = FH.xarray(search_str, remove_grib=False)
                else:
                    # after first attempt, set overwrite=True to overwrite
                    # partial files
                    FH = FastHerbie(DATES=[init_date],
                                    model=model_herbie,
                                    product=product,
                                    fxx=fxx_range,
                                    member=member,
                                    priority=priority)
                    FH.download(search_str, overwrite=True)
                    ds = FH.xarray(search_str, remove_grib=False)
            except Exception as e:
                print(e)
                if attempts_remaining:
                    print('attempt ' + str(attempt_num) + ' failed, pause for '
                          + str((attempt_num)**2) + ' min')
                    time.sleep(60*(attempt_num)**2)
                else:
                    raise ValueError(f'download failed, ran out of '
                                     f'attempts with error: {e}')
            else:
                break

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
