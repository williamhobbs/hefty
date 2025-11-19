import numpy as np
import pandas as pd
import pvlib


# from https://github.com/williamhobbs/2024_pvpmc_self_shade
def shade_fractions(fs_array, eff_row_side_num_mods):
    """
    Shade fractions on each _course_ of a rack or tracker.

    Parameters
    ----------
    fs_array : numeric
        Scalar or vector of shade fractions for the entire rack or tracker.
        Zero (0) is unshaded and one (1) is fully shaded.
    eff_row_side_num_mods : int
        Number of courses in the rack as modules. EG: a 2P tracker has 2
        courses.

    Returns
    -------
    Array with the shade fraction on each course.
    """
    fs_course = np.clip([
        fs_array * eff_row_side_num_mods - course
        for course in range(eff_row_side_num_mods)], 0, 1)
    return fs_course


# from https://github.com/williamhobbs/2024_pvpmc_self_shade
def non_linear_shade(n_cells_up, fs, fd):
    """
    Simple non-linear shade model.

    Assume shade loss is linear as direct shade moves from bottom through
    first cell, and then only diffuse for remainder of module up to the top.
    EG: If there are 10 cells, and ``fs`` is 0.05, the bottom cell is only
    half shaded, then the loss would be 50% of the direct irradiance. If
    80% of POA global on module is direct irradiance, IE: ``fd = 0.2``,
    then loss would be 40%. When the direct shade line reaches 10%, one
    cell is completely shaded, then the loss is 80%, and there's only diffuse
    light on the module. Any direct shade above the 1st cell has the same loss.

    Parameters
    ----------
    n_cells_up : int
        Number of cells vertically up
    fs : float
        Fraction of shade on module, 1 is fully shaded
    fd : numeric
        Diffuse fraction

    Returns
    -------
    Array of shade loss same size as ``fd``
    """
    pnorm = np.where(fs < 1/n_cells_up, 1 - (1 - fd)*fs*n_cells_up, fd)
    shade_loss = 1 - pnorm
    return shade_loss


# Partially based on
# https://github.com/williamhobbs/2024_pvpmc_self_shade/tree/iam_and_spectrum
def model_pv_power(
        resource_data,
        latitude,
        longitude,
        mount_type,
        gcr,
        nameplate_dc,
        nameplate_ac,
        dc_loss_fraction,
        gamma_pdc,
        shade_loss_model,
        default_site_transposition_model='perez-driesse',
        backtrack=True,
        backtrack_fraction=1,
        max_tracker_angle=pd.NA,
        axis_tilt=pd.NA,
        axis_azimuth=pd.NA,
        fixed_tilt=pd.NA,
        fixed_azimuth=pd.NA,
        n_cells_up=12,
        row_side_num_mods=pd.NA,
        row_height_center=pd.NA,
        row_pitch=pd.NA,
        collector_width=pd.NA,
        bifacial=False,
        bifaciality_factor=0.8,
        surface_tilt_timeseries=pd.Series([], dtype='float64'),
        surface_azimuth_timeseries=pd.Series([], dtype='float64'),
        use_measured_poa=False,
        use_measured_temp_module=False,
        cell_type='crystalline',
        eta_inv_nom=0.98,
        cross_axis_slope=0,
        gcr_backtrack_setting=pd.NA,
        programmed_gcr_am=pd.NA,
        programmed_gcr_pm=pd.NA,
        slope_aware_backtracking=True,
        programmed_cross_axis_slope=pd.NA,
        altitude=0,
        **kwargs,
):
    """
    Power model for PV plants. Based a bit on
    https://github.com/williamhobbs/2024_pvpmc_self_shade/tree/iam_and_spectrum

    Parameters
    ----------
    resource_data : pandas.DataFrame
        timeseries weather/resource data with the same format as is returned by
        pvlib.iotools.get* functions
    [long list of additional arguments defining a plant based on [1]]
    surface_tilt_timeseries : pandas.DataFrame
        (optional) custom timeseries of the angle between the panel surface and
        the earth surface, accounting for panel rotation. [degrees]
    surface_azimuth_timeseries : pandas.DataFrame
        (optional) custom timeseries of the azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface to
        the earth's surface. [degrees]
    use_measured_poa : bool, default False
        If True, used measure POA data from ``resource_data`` (must have
        column name 'poa').
    use_measured_temp_module: bool, default False
        If True, use measured back of module temperature from ``resource_data``
        (must have column name 'temp_module') in place of modeled cell
        temperature.

    Returns
    -------
    power_ac : pandas.Series
        AC power. Same units as ``dc_capacity_plant`` and
        ``power_plant_ac_max`` (ideally kW).
    resource_data : pandas.DataFrame
        modified version of input ``resource_data`` with modeled poa, module
        temperature, and possibly other parameters added.

    References
    ----------
    .. [1] William Hobbs, pv-plant-specification-rev4.csv,
       https://github.com/williamhobbs/pv-plant-specifications
    """

    # ========================================================================
    # Inputs and Basic Geometry
    # ========================================================================
    # Fill in some necessary variables with defaults if there is no value
    # provided.
    # backtracking settings: default to AM/PM settings, then generic
    # programmed value, then physical gcr
    if pd.isna(gcr_backtrack_setting):
        gcr_backtrack_setting = gcr
    if pd.isna(programmed_gcr_am):
        programmed_gcr_am = gcr_backtrack_setting
    if pd.isna(programmed_gcr_pm):
        programmed_gcr_pm = gcr_backtrack_setting
    if backtrack_fraction == 0:
        backtrack = False  # switch to truetracking to avoid divide by zero

    # slope-aware backtracking settings
    if ((slope_aware_backtracking is True) &
       (pd.isna(programmed_cross_axis_slope))):
        programmed_cross_axis_slope = cross_axis_slope
    elif ((slope_aware_backtracking is False) &
          (pd.isna(programmed_cross_axis_slope))):
        programmed_cross_axis_slope = 0
    elif ((slope_aware_backtracking is False) &
          (pd.notna(programmed_cross_axis_slope))):
        slope_aware_backtracking = True
        print("""You provided a value for programmed_cross_axis_slope
        AND did not set slope_aware_backtracking=True. Slope-aware
        backtracking will be enabled.""")

    # geometry
    if pd.isna(axis_tilt):
        axis_tilt = 0  # default if no value provided
    if pd.isna(axis_azimuth):
        axis_azimuth = 180  # default if no value provided
    if pd.isna(row_side_num_mods):
        row_side_num_mods = 1  # default if no value provided
    if pd.isna(row_height_center):
        row_height_center = 1  # default if no value provided
    if pd.isna(max_tracker_angle):
        max_tracker_angle = 60

    # gcr = collector_width / row_pitch, gcr is a required input, so users can
    # define 1 of the other 2.
    # If all 3 are defined, check to make sure relationship is correct.
    if pd.isna(collector_width) & pd.isna(row_pitch):  # neither provided
        collector_width = 2  # default if no value provided
        row_pitch = collector_width / gcr
    elif pd.isna(row_pitch):
        row_pitch = collector_width / gcr
    elif pd.isna(collector_width):
        collector_width = gcr * row_pitch
    elif gcr != collector_width / row_pitch:
        raise ValueError("""You provided collector_width, row_pitch, and gcr,
            but they are inconsistent. gcr must equal:
            collector_width / row_pitch.
            Please check these values. Note that gcr is required, and only one
            of collector_width and row_pitch are needed to fully define the
            related geometry.""")

    # general geometry calcs
    pitch = collector_width / gcr

    # ========================================================================
    # Time and solar position
    # ========================================================================
    # time and solar position with correct time
    times = resource_data.index
    loc = pvlib.location.Location(latitude=latitude, longitude=longitude,
                                  tz=times.tz, altitude=altitude)
    solar_position = loc.get_solarposition(times)

    # ========================================================================
    # Surface tilt, azimuth, and shaded fraction
    # ========================================================================
    if surface_tilt_timeseries.empty | surface_azimuth_timeseries.empty:
        if mount_type == 'single-axis':
            # modify tracker gcr if needed
            if backtrack is True:
                programmed_gcr = np.where(solar_position.azimuth < 180,
                                          programmed_gcr_am*backtrack_fraction,
                                          programmed_gcr_pm*backtrack_fraction)
            else:
                programmed_gcr = gcr_backtrack_setting

            # change cross_axis_slope used for tracker orientation if needed
            # if slope_aware_backtracking is True:
            #     programmed_cross_axis_slope = cross_axis_slope
            # else:
            #     programmed_cross_axis_slope = 0

            # tracker orientation
            tr = pvlib.tracking.singleaxis(
                solar_position.apparent_zenith,
                solar_position.azimuth,
                gcr=programmed_gcr,
                axis_tilt=axis_tilt,
                axis_azimuth=axis_azimuth,
                cross_axis_tilt=programmed_cross_axis_slope,
                max_angle=max_tracker_angle,
                backtrack=backtrack)

            # calculate shading with slope
            fs_array = pvlib.shading.shaded_fraction1d(
                solar_position.apparent_zenith,
                solar_position.azimuth,
                axis_azimuth=axis_azimuth,
                shaded_row_rotation=tr.tracker_theta,
                collector_width=collector_width, pitch=pitch,
                axis_tilt=axis_tilt,
                cross_axis_slope=cross_axis_slope)

            surface_tilt = tr.surface_tilt.fillna(0)
            surface_azimuth = tr.surface_azimuth.fillna(0)
            resource_data['tracker_theta'] = tr.tracker_theta
        elif mount_type == 'fixed':
            # calculate shading
            # model fixed array as a stuck tracker for azimuth and rotation
            fs_array = pvlib.shading.shaded_fraction1d(
                solar_position.apparent_zenith,
                solar_position.azimuth,
                axis_azimuth=fixed_azimuth - 90,
                shaded_row_rotation=fixed_tilt,
                collector_width=collector_width, pitch=pitch,
                axis_tilt=axis_tilt,
                cross_axis_slope=cross_axis_slope
                )
            surface_tilt = float(fixed_tilt)
            surface_azimuth = float(fixed_azimuth)
            resource_data['tracker_theta'] = np.nan
    else:
        surface_tilt = surface_tilt_timeseries
        surface_azimuth = surface_azimuth_timeseries
        # calculate tracker theta, TODO: double-check this
        tracker_theta = surface_tilt.where((surface_azimuth >= 180),
                                           - surface_tilt)

        fs_array = pvlib.shading.shaded_fraction1d(
            solar_position.apparent_zenith,
            solar_position.azimuth,
            axis_azimuth=axis_azimuth,
            shaded_row_rotation=tracker_theta,
            collector_width=collector_width, pitch=pitch,
            axis_tilt=axis_tilt,
            cross_axis_slope=cross_axis_slope)
        resource_data['tracker_theta'] = tracker_theta
    resource_data['fs_array'] = fs_array

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solar_position.apparent_zenith,
                               solar_position.azimuth)

    # ========================================================================
    # Modeled POA
    # ========================================================================
    if 'dhi' not in resource_data:
        print('calculating dhi')
        # calculate DHI with "complete sum" AKA "closure" equation:
        # DHI = GHI - DNI * cos(zenith)
        resource_data['dhi'] = (resource_data.ghi - resource_data.dni *
                                pvlib.tools.cosd(solar_position.zenith))

    # dni
    dni_extra = pvlib.irradiance.get_extra_radiation(resource_data.index)

    # total irradiance
    total_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_position.apparent_zenith,
        solar_azimuth=solar_position.azimuth,
        dni=resource_data.dni,
        ghi=resource_data.ghi,
        dhi=resource_data.dhi,
        dni_extra=dni_extra,
        albedo=resource_data.albedo,
        model=default_site_transposition_model,
    )

    resource_data['poa_modeled'] = total_irrad['poa_global']

    # ========================================================================
    # Handle measured POA, get diffuse and direct ready
    # ========================================================================
    if use_measured_poa is True:
        poa_total_without_direct_shade = resource_data.poa
        irrad_dirint = pvlib.irradiance.gti_dirint(
            poa_total_without_direct_shade, aoi,
            solar_position.apparent_zenith, solar_position.azimuth,
            times, surface_tilt, surface_azimuth)
        poa_direct_unshaded = irrad_dirint['dni'] # output of gti_dirint
        poa_diffuse_unshaded = irrad_dirint['dhi'] # output of gti_dirint
    else:
        # work backwards to unshaded direct irradiance for the array:
        # poa_direct_unshaded = total_irrad.poa_direct / (1-fs_array)
        # !!! get_total_irradiance doesn't include shade like infinite_sheds,
        # so no correction needed!!!
        poa_direct_unshaded = total_irrad['poa_direct']
        poa_diffuse_unshaded = total_irrad['poa_diffuse']

    # ========================================================================
    # IAM and Spectral correction
    # ========================================================================
    # iam
    # iam = pvlib.iam.physical(aoi, L=0.0032, n_ar=1.29)
    # iam = pvlib.iam.physical(aoi, L=0.0032)
    iam = pvlib.iam.physical(aoi)

    # spectral modifier for cdte
    if cell_type == 'thin-film_cdte':
        airmass = loc.get_airmass(solar_position=solar_position)
        if 'precipitable_water' not in resource_data.columns:
            if (('temp_air' in resource_data.columns) &
               ('relative_humidity' in resource_data.columns)):
                resource_data['precipitable_water'] = \
                    pvlib.atmosphere.gueymard94_pw(
                        temp_air=resource_data.temp_air,
                        relative_humidity=resource_data.relative_humidity)
            else:
                resource_data['precipitable_water'] = 1
        spectral_modifier = pvlib.spectrum.spectral_factor_firstsolar(
            precipitable_water=resource_data.precipitable_water,
            airmass_absolute=airmass.airmass_absolute,
            module_type='cdte',
        )

    # apply iam
    poa_direct_unshaded = poa_direct_unshaded * iam

    # total poa on the front, but without direct shade impacts
    # (would be keeping diffuse impacts from infinite_sheds if we used
    # inifinite_sheds...)
    poa_total_without_direct_shade = (poa_diffuse_unshaded +
                                      poa_direct_unshaded)

    if cell_type == 'thin-film_cdte':
        poa_total_without_direct_shade = (poa_total_without_direct_shade *
                                          spectral_modifier)

    # set zero POA to nan to avoid divide by zero warnings
    # this might not be needed!!!
    poa_total_without_direct_shade = (
        poa_total_without_direct_shade.replace(0, np.nan))

    # ========================================================================
    # Shade losses
    # ========================================================================
    # set the "effective" number of modules on the side of each row
    if shade_loss_model == 'non-linear_simple':
        eff_row_side_num_mods = int(row_side_num_mods)
    elif shade_loss_model == 'non-linear_simple_twin_module':
        # twin modules are treated as effectively two modules with half as
        # many cells each
        eff_row_side_num_mods = int(row_side_num_mods) * 2
        n_cells_up = n_cells_up / 2
    # for linear shade loss, it really doesn't matter how many modules there
    # are on the side of each row, so just run everything once to save time
    elif shade_loss_model == 'linear':
        eff_row_side_num_mods = 1
    else:
        raise ValueError("""shade_loss_model must be one of:
            'non-linear_simple', 'non-linear_simple_twin_module', or 'linear'.
            You entered: '""" + shade_loss_model + """'""")

    # shaded fraction for each course/string going up the row
    fs = shade_fractions(fs_array, eff_row_side_num_mods)
    # total POA *with* direct shade impacts
    poa_total_with_direct_shade = (((1-fs) * poa_direct_unshaded.values) +
                                   total_irrad['poa_diffuse'].values)
    # diffuse fraction
    fd = (total_irrad['poa_diffuse'].values /
          poa_total_without_direct_shade.values)

    # calculate shade loss for each course/string
    if shade_loss_model == 'linear':
        shade_loss = fs * (1 - fd)
    elif (shade_loss_model == 'non-linear_simple' or
          shade_loss_model == 'non-linear_simple_twin_module'):
        shade_loss = non_linear_shade(n_cells_up, fs, fd)

    # adjust irradiance based on modeled shade loss
    poa_front_effective = ((1 - shade_loss) *
                           poa_total_without_direct_shade.values)

    # ========================================================================
    # Temperature
    # ========================================================================
    # cell temperature
    # steady state cell temperature - faiman is much faster than fuentes,
    # simpler than sapm
    t_cell_modeled = np.array([
        pvlib.temperature.faiman(
            poa_total_with_direct_shade[n],
            resource_data['temp_air'],
            resource_data['wind_speed']).values
        for n in range(eff_row_side_num_mods)])

    # apply rolling 10-min avg if interval is less than 10 min
    # based on https://www.epri.com/research/products/000000003002018708
    sample_interval, samples_per_window = pvlib.tools._get_sample_intervals(
        times=resource_data.index, win_length=10)
    if sample_interval < 10:
        N = samples_per_window
        # based on https://stackoverflow.com/a/47490020/27574852
        t_cell_modeled = np.array([
            np.convolve(
                t_cell_modeled[n], np.ones((N,))/N, 'same')
            for n in range(eff_row_side_num_mods)])

    if use_measured_temp_module is True:
        # use measured module temperature - repeat the single timeseries
        # for each course in the array
        t_cell = np.array([
            resource_data['temp_module'].values
            for n in range(eff_row_side_num_mods)])
    else:
        t_cell = t_cell_modeled

    resource_data['t_cell_modeled'] = np.mean(t_cell_modeled, axis=0)

    # ========================================================================
    # Bifacial
    # ========================================================================
    if bifacial is True:
        # transposition models allowed for infinite_sheds:
        if default_site_transposition_model not in ['haydavies', 'isotropic']:
            print('pvlib.bifacial.infinite_sheds does not currently accept'
                  ' the ' + default_site_transposition_model + ' model.')
            print('using haydavies instead.')
            inf_sheds_transposition_model = 'haydavies'
        else:
            inf_sheds_transposition_model = default_site_transposition_model

        # run infinite_sheds to get rear irradiance
        irrad_inf_sh = pvlib.bifacial.infinite_sheds.get_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_position.apparent_zenith,
            solar_azimuth=solar_position.azimuth,
            gcr=gcr,
            height=row_height_center,
            pitch=row_pitch,
            ghi=resource_data.ghi,
            dhi=resource_data.dhi,
            dni=resource_data.dni,
            albedo=resource_data.albedo,
            model=inf_sheds_transposition_model,
            dni_extra=dni_extra,
            bifaciality=bifaciality_factor,
        )

        # now for the rear irradiance
        fs_array_back = irrad_inf_sh['shaded_fraction_back']
        poa_back_direct_unshaded = (
            irrad_inf_sh['poa_back_direct'] / (1-fs_array_back)
        )
        poa_back_total_without_direct_shade = (
            irrad_inf_sh['poa_back_diffuse'] + poa_back_direct_unshaded
        )
        poa_back_total_without_direct_shade.replace(0, np.nan, inplace=True)
        fs_back = shade_fractions(fs_array_back, eff_row_side_num_mods)
        # commented out, not currently used:
        # poa_back_total_with_direct_shade = (
        #     ((1-fs_back) * poa_back_direct_unshaded.values) +
        #     irrad_inf_sh['poa_back_diffuse'].values
        # )
        fd = (irrad_inf_sh['poa_back_diffuse'].values /
              poa_back_total_without_direct_shade.values)
        if shade_loss_model == 'linear':
            # shade_loss = fs * (1 - fd)
            shade_loss = fs_back * (1 - fd)
        elif (shade_loss_model == 'non-linear_simple' or
              shade_loss_model == 'non-linear_simple_twin_module'):
            # shade_loss = non_linear_shade(n_cells_up, fs, fd)
            shade_loss = non_linear_shade(n_cells_up, fs_back, fd)

        # adjust irradiance based on modeled shade loss, include
        # bifaciality_factor
        poa_back_effective = (bifaciality_factor * (1 - shade_loss) *
                              poa_back_total_without_direct_shade.values)

        # combine front and back effective POA
        poa_effective = poa_front_effective + poa_back_effective
        resource_data['poa_effective'] = (
            pd.DataFrame(poa_effective.T, index=times).mean(axis=1))
        resource_data['poa_front_effective'] = (
            pd.DataFrame(poa_front_effective.T, index=times).mean(axis=1))
        resource_data['poa_back_effective'] = (
            pd.DataFrame(poa_back_effective.T, index=times).mean(axis=1))
        resource_data['poa_back_modeled'] = (
            poa_back_total_without_direct_shade.values)
    else:
        poa_effective = poa_front_effective
        resource_data['poa_effective'] = (
            pd.DataFrame(poa_effective.T, index=times).mean(axis=1))
        resource_data['poa_front_effective'] = (
            pd.DataFrame(poa_front_effective.T, index=times).mean(axis=1))
        resource_data['poa_back_effective'] = np.nan
        resource_data['poa_back_modeled'] = np.nan

    # ========================================================================
    # Power
    # ========================================================================
    # PVWatts dc power
    pdc_shaded = pvlib.pvsystem.pvwatts_dc(
        poa_effective, t_cell, nameplate_dc, gamma_pdc)

    # dc power into the inverter after losses
    pdc_inv = pdc_shaded * (1 - dc_loss_fraction)

    # inverter dc input is ac nameplate divided by nominal inverter efficiency
    pdc0 = nameplate_ac/eta_inv_nom

    # average the dc power across n positions up the row
    pdc_inv_total = pd.DataFrame(pdc_inv.T, index=times).mean(axis=1)

    # fill nan with zero
    pdc_inv_total.fillna(0, inplace=True)
    resource_data.fillna(0, inplace=True)

    # ac power with PVWatts inverter model
    power_ac = pvlib.inverter.pvwatts(pdc_inv_total, pdc0, eta_inv_nom)

    return power_ac, resource_data.copy()
