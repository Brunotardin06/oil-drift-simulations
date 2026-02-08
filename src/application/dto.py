from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ConfigRequest:
    config_name: str = "main"
    environment: str = "2019"
    simulation_name: str = "sim4validation"
    shp_zip: Optional[str] = None
    min_long: Optional[float] = None
    max_long: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None

    def to_overrides(self) -> List[str]:
        overrides: List[str] = [f"simulation={self.simulation_name}"]
        if self.shp_zip:
            overrides.append(f'paths.plataformas_shp="{self.shp_zip}"')
        if self.min_long is not None:
            overrides.append(f"copernicusmarine.min_long={self.min_long}")
        if self.max_long is not None:
            overrides.append(f"copernicusmarine.max_long={self.max_long}")
        if self.min_lat is not None:
            overrides.append(f"copernicusmarine.min_lat={self.min_lat}")
        if self.max_lat is not None:
            overrides.append(f"copernicusmarine.max_lat={self.max_lat}")
        return overrides


@dataclass(frozen=True)
class ObservedSpillRequest:
    spill_path: Path
    offset_hours: float = 0.0
    start_index: int = 0
    padding_animation_frame: float = 0.1


@dataclass
class ObservedSpillContext:
    manchas: Any
    plot_bounds: Tuple[float, float, float, float]


@dataclass(frozen=True)
class FastOptimizationRequest:
    manchas: Any
    config: Any
    observed_trajectory: Any
    wdf_values: Sequence[float]
    current_drift_values: Sequence[float]
    horizontal_diffusivity_values: Optional[Sequence[float]] = None
    particles_per_wdf: int = 1
    oil_type: Optional[str] = None
    progress: Any = None


@dataclass(frozen=True)
class SimulationRunRequest:
    manchas: Any
    out_filename: str
    config: Any
    skip_animation: bool
    padding_animation_frame: float
    wind_drift_factor: Optional[float] = None
    stokes_drift: Optional[bool] = None
    current_drift_factor: Optional[float] = None
    oil_type: Optional[str] = None
    horizontal_diffusivity: Optional[float] = None
    processes_dispersion: Optional[bool] = None
    processes_evaporation: Optional[bool] = None


@dataclass(frozen=True)
class GifGenerationRequest:
    sim_nc: Path
    shp_zip: str
    out: Path
    extent: str
    datetime_offset_hours: float = 0.0
    real_steps: int = 0


@dataclass(frozen=True)
class ValidationRunRequest:
    config_name: str = "main"
    skip_animation: bool = False
    skip_simulation: bool = False
    skip_plots: bool = False
    evaluation: bool = False
    optimize_wdf: bool = False
    optimize_stokes: bool = False
    optimize_wdf_stokes: bool = False
    optimize_wdf_stokes_cdf: bool = False
    optimize_physics: bool = False
    optimize_wdf_mode: str = "fast"
    fast_particles_per_wdf: int = 1
    wdf_min: float = 0.0
    wdf_max: float = 0.05
    wdf_step: float = 0.0025
    cdf_min: float = 0.5
    cdf_max: float = 1.0
    cdf_step: float = 0.1
    diffusivity_values: Optional[str] = None
    dispersion_values: Optional[str] = None
    evaporation_values: Optional[str] = None
    optimize_cleanup: bool = False
    padding_animation_frame: float = 0.1
    wind_drift_factor: Optional[float] = None
    current_drift_factor: Optional[float] = None
    stokes_drift: Optional[str] = None
    horizontal_diffusivity: Optional[float] = None
    processes_dispersion: Optional[str] = None
    processes_evaporation: Optional[str] = None
    oil_types: Optional[str] = None
    oil_types_file: Optional[str] = None
    environment: str = "2019"
    shp_zip: Optional[str] = None
    min_long: Optional[float] = None
    max_long: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    start_index: int = 0
    optimize_cdf_hd_de: bool = False
