from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class TrajectoryPoint:
    time: datetime
    lon: float
    lat: float


@dataclass
class SimulationParams:
    wind_drift_factor: Optional[float] = None
    current_drift_factor: Optional[float] = None
    stokes_drift: Optional[bool] = None
    horizontal_diffusivity: Optional[float] = None
    processes_dispersion: Optional[bool] = None
    processes_evaporation: Optional[bool] = None
    oil_type: Optional[str] = None


@dataclass(frozen=True)
class OptimizationResult:
    wind_drift_factor: float
    current_drift_factor: float
    stokes_drift: bool
    skillscore: float
    horizontal_diffusivity: Optional[float] = None
    oil_type: Optional[str] = None

