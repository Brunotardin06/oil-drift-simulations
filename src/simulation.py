"""Compatibility facade for simulation functions.

This module keeps the previous function-based API used by `simulate.py`
while delegating the implementation to the new service classes.
"""

from src.services.simulation_service import SimulationService

_simulation_service = SimulationService()


def simulate_drift(
    manchas,
    out_filename,
    config,
    skip_animation,
    padding_animation_frame,
    wind_drift_factor=None,
    current_drift_factor=None,
    oil_type=None,
    horizontal_diffusivity=None,
    processes_dispersion=None,
    processes_evaporation=None,
    forcing_source="COPERNICUS",
    current_dataset_path=None,
    wind_dataset_path=None,
    current_dataset_paths=None,
    wind_dataset_paths=None,
    observed_offset_hours=None,
):
    return _simulation_service.simulate_drift(
        manchas=manchas,
        out_filename=out_filename,
        config=config,
        skip_animation=skip_animation,
        padding_animation_frame=padding_animation_frame,
        wind_drift_factor=wind_drift_factor,
        current_drift_factor=current_drift_factor,
        oil_type=oil_type,
        horizontal_diffusivity=horizontal_diffusivity,
        processes_dispersion=processes_dispersion,
        processes_evaporation=processes_evaporation,
        forcing_source=forcing_source,
        current_dataset_path=current_dataset_path,
        wind_dataset_path=wind_dataset_path,
        current_dataset_paths=current_dataset_paths,
        wind_dataset_paths=wind_dataset_paths,
        observed_offset_hours=observed_offset_hours,
    )
