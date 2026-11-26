import pandas as pd

from hefty.pv_model import model_pv_power
from hefty.solar import get_solar_forecast


def model_pv_power_plants(plants, init_date, run_length, albedo=0.2, **kwargs):
    """
    Get a forecast of solar generation for multiple PV plants.

    Downloads one solar resource forecast for all plant locations with
    :py:func:`hefty.solar.get_solar_forecast` and models each plant with
    :py:func:`hefty.pv_model.model_pv_power`, as in
    ``examples/more_solar_examples.ipynb``.

    Parameters
    ----------
    plants : pandas.DataFrame
        One row per plant, indexed by plant name. Columns are
        :py:func:`hefty.pv_model.model_pv_power` arguments and must include
        ``latitude`` and ``longitude``. Missing values fall back to the
        defaults of :py:func:`hefty.pv_model.model_pv_power`.
    init_date : pandas-parsable datetime
        Model initialization datetime (UTC), see
        :py:func:`hefty.solar.get_solar_forecast`.
    run_length : int
        Length of the forecast in hours.
    albedo : numeric, default 0.2
        Ground albedo used for all plants.
    **kwargs
        Passed to :py:func:`hefty.solar.get_solar_forecast`, e.g. ``model``,
        ``lead_time_to_start``, ``priority`` or ``attempts``.

    Returns
    -------
    power_ac : pandas.DataFrame
        AC power with one column per plant, indexed by ``valid_time`` (UTC,
        interval midpoints). Same units as ``nameplate_ac``.
    """
    resource_data = get_solar_forecast(
        latitude=plants.latitude.values,
        longitude=plants.longitude.values,
        init_date=init_date,
        run_length=run_length,
        **kwargs,
    )
    resource_data['albedo'] = albedo

    # Each weather data point must be labelled with the plant name
    resource_data['plant_name'] = plants.index[resource_data['point']]

    pv_generation_list = []
    for plant_name in plants.index:
        plant_parameters = plants.loc[plant_name].dropna().to_dict()
        is_plant = resource_data['plant_name'] == plant_name
        power_ac, weather = model_pv_power(
            resource_data=resource_data[is_plant].copy(),
            **plant_parameters,
        )
        pv_generation_list.append(power_ac.rename(plant_name))
    return pd.concat(pv_generation_list, axis=1)
