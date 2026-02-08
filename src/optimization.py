"""Compatibility facade for optimization functions.

This module keeps the previous function-based API used by `simulate.py`
while delegating the implementation to the new service/repository classes.
"""

from src.infrastructure.spill_repository import SpillRepository
from src.services.metrics_service import MetricsService
from src.services.optimization_service import OptimizationService

_spill_repository = SpillRepository()
_metrics_service = MetricsService()
_optimization_service = OptimizationService(
    spill_repository=_spill_repository,
    metrics_service=_metrics_service,
)


def build_observed_trajectory(manchas):
    return _spill_repository.build_observed_trajectory(manchas)


def liu_weissberg_skillscore(observed_df, modeled_df):
    return _metrics_service.liu_weissberg_skillscore(observed_df, modeled_df)


def fast_grid_search_wind_drift_factor(
    manchas,
    config,
    observed_trajectory,
    wdf_values,
    particles_per_wdf=1,
    current_drift_factor=None,
    horizontal_diffusivity=None,
    oil_type=None,
    progress=None,
):
    return _optimization_service.fast_grid_search_wind_drift_factor(
        manchas=manchas,
        config=config,
        observed_trajectory=observed_trajectory,
        wdf_values=wdf_values,
        particles_per_wdf=particles_per_wdf,
        current_drift_factor=current_drift_factor,
        horizontal_diffusivity=horizontal_diffusivity,
        oil_type=oil_type,
        progress=progress,
    )


def fast_grid_search_wdf_stokes_current_drift(
    manchas,
    config,
    observed_trajectory,
    wdf_values,
    current_drift_values,
    horizontal_diffusivity_values=None,
    particles_per_wdf=1,
    oil_type=None,
    progress=None,
):
    return _optimization_service.fast_grid_search_wdf_stokes_current_drift(
        manchas=manchas,
        config=config,
        observed_trajectory=observed_trajectory,
        wdf_values=wdf_values,
        current_drift_values=current_drift_values,
        horizontal_diffusivity_values=horizontal_diffusivity_values,
        particles_per_wdf=particles_per_wdf,
        oil_type=oil_type,
        progress=progress,
    )

