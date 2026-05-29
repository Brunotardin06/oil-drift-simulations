from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class StochasticParameterConfig:
    enabled: bool
    mean: float
    std: float
    min_value: float
    max_value: float
    distribution: str = "normal"
    default_value: Optional[float] = None


@dataclass(frozen=True)
class TemporalLagConfig:
    enabled: bool
    mean: float
    std: float
    min_value: float
    max_value: float
    input_unit: str = "seconds"
    rounding_granularity: str = "seconds"
    distribution: str = "normal"
    default_seconds: float = 0.0


@dataclass(frozen=True)
class StochasticGridConfig:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    spatial_resolution: float
    margin: float = 0.0
    crs: str = "EPSG:4326"


@dataclass(frozen=True)
class StochasticRunConfig:
    run_name: str
    n_simulations: int
    seed: Optional[int]
    cdf: StochasticParameterConfig
    wdf: StochasticParameterConfig
    temporal_lag: TemporalLagConfig
    grid: StochasticGridConfig
    output_root: Path
    execution_mode: str = "deterministic_ensemble"
    number_of_workers: int = 1


@dataclass(frozen=True)
class SampledParameterSet:
    simulation_id: int
    seed: int
    cdf: float
    wdf: float
    temporal_lag_original_value: float
    temporal_lag_input_unit: str
    temporal_lag_rounding_granularity: str
    temporal_lag_seconds: float
    status: str = "pending"
    error_message: str = ""
    output_path: str = ""


@dataclass(frozen=True)
class StochasticRunResult:
    run_name: str
    total_simulations: int
    successful_simulations: int
    failed_simulations: int
    output_path: Path
    sampled_parameters_path: Path
    summary_path: Path
    hit_count_map_path: Optional[Path] = None
    probability_map_path: Optional[Path] = None
    hit_count_final_timestep_map_path: Optional[Path] = None
    probability_final_timestep_map_path: Optional[Path] = None
    hourly_probability_map_paths: Tuple[Path, ...] = ()
