from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from opendrift.models.openoil import OpenOil
from shapely.geometry import Point

# Force a non-interactive backend to avoid Tkinter teardown errors when
# matplotlib figures are created from worker threads (e.g., Flet execution).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.application.dto import (
    ConfigRequest,
    FastOptimizationRequest,
    GifGenerationRequest,
    ObservedSpillContext,
    ObservedSpillRequest,
    SimulationRunRequest,
    ValidationRunRequest,
    ValidationRunResult,
)
from src.infrastructure.environment_repository import EnvironmentRepository
from src.infrastructure.spill_repository import SpillRepository
from src.infrastructure.workspace_repository import WorkspaceRepository
from src.infrastructure.copernicus_gateway import CopernicusGateway
from src.services.optimization_service import OptimizationService
from src.services.output_service import OutputService
from src.services.simulation_service import SimulationService


class _ProgressPrinter:
    def __init__(
        self,
        total,
        every=15,
        label="Progress",
        on_tick: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        self.total = int(total)
        self.every = int(every)
        self.label = label
        self.count = 0
        self.on_tick = on_tick
        self.should_cancel = should_cancel

    def tick(self, n=1):
        if self.should_cancel is not None and self.should_cancel():
            raise RuntimeError("Execution cancelled by user.")
        if self.total <= 0:
            return
        self.count += int(n)
        if self.on_tick is not None:
            self.on_tick(self.count, self.total)
        if self.count % self.every == 0 or self.count >= self.total:
            print(f"{self.label}: {self.count}/{self.total}")


@dataclass
class _ValidationContext:
    config: Any
    offset_hours: float
    environmental_offset_hours: float
    forcing_source: str
    current_dataset_path: Path
    wind_dataset_path: Optional[Path]
    current_dataset_paths: tuple[Path, ...]
    wind_dataset_paths: tuple[Path, ...]
    real_manchas: Any
    plot_bounds: tuple[float, float, float, float]
    observed_trajectory: Any
    environmental_offset_values: Optional[tuple[float, ...]]
    start_index: int
    skip_animation: bool
    skip_plots: bool
    wave_effects_enabled: bool
    wind_drift_factor: Optional[float]
    current_drift_factor: Optional[float]
    processes_dispersion: Optional[bool]
    processes_evaporation: Optional[bool]
    selected_oil_type: Optional[str]


class SimulationController:
    """Orchestrate repositories and services for full simulation workflows."""

    DEFAULT_ENVIRONMENTAL_OFFSET_HOURS = -3.0
    MAX_ENVIRONMENTAL_OFFSET_HOURS = 24.0

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        environment_repository: Optional[EnvironmentRepository] = None,
        workspace_repository: Optional[WorkspaceRepository] = None,
        copernicus_gateway: Optional[CopernicusGateway] = None,
        optimization_service: Optional[OptimizationService] = None,
        simulation_service: Optional[SimulationService] = None,
        output_service: Optional[OutputService] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
        self.environment_repository = environment_repository or EnvironmentRepository()
        self.workspace_repository = workspace_repository or WorkspaceRepository()
        self.copernicus_gateway = copernicus_gateway or CopernicusGateway()
        self.optimization_service = optimization_service or OptimizationService(
            spill_repository=self.spill_repository
        )
        self.simulation_service = simulation_service or SimulationService(
            spill_repository=self.spill_repository
        )
        self.output_service = output_service or OutputService()

    @staticmethod
    def _normalize_forcing_source(value: Optional[str]) -> str:
        source = (value or "COPERNICUS").strip().upper()
        supported = {"COPERNICUS", "NOAA", "REMO"}
        if source not in supported:
            raise ValueError(
                f"Unsupported forcing source '{value}'. Supported values: COPERNICUS, NOAA, REMO."
            )
        return source

    @staticmethod
    def _parse_float_list(value, default):
        if value is None:
            return list(default)
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            return list(default)
        return [float(item) for item in parts]

    @staticmethod
    def _parse_bool_string(value):
        if value is None:
            return None
        parsed = value.strip().lower()
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
        if parsed not in mapping:
            raise ValueError(f"Invalid boolean value: {value}")
        return mapping[parsed]

    @staticmethod
    def _load_oil_types(oil_types, oil_types_file):
        def normalize_name(name):
            return " ".join(name.strip().split()).lower()

        selected = []
        if oil_types_file:
            file_path = Path(oil_types_file)
            if not file_path.exists():
                raise ValueError(f"oil types file not found: {file_path}")
            selected.extend(
                [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            )
        if oil_types:
            selected.extend([item.strip() for item in oil_types.split(",") if item.strip()])

        if not selected:
            return []

        available = OpenOil(loglevel=50).oiltypes
        by_normalized = {normalize_name(name): name for name in available}
        normalized = []
        invalid = []
        for name in selected:
            mapped = by_normalized.get(normalize_name(name))
            if mapped:
                normalized.append(mapped)
                continue
            invalid.append(name)

        if invalid:
            raise ValueError(
                "Unknown oil types: "
                + ", ".join(invalid[:10])
                + (" ..." if len(invalid) > 10 else "")
            )

        return list(dict.fromkeys(normalized))

    @staticmethod
    def _build_sim_filename(
        base_name,
        wind_drift_factor=None,
        wave_effects_enabled=None,
        current_drift_factor=None,
        processes_dispersion=None,
        processes_evaporation=None,
    ):
        sim_filename = base_name
        if wind_drift_factor is not None:
            safe_wdf = f"{wind_drift_factor:.4f}".replace(".", "p")
            sim_filename = f"{sim_filename}_wdf{safe_wdf}"
        if wave_effects_enabled is not None:
            sim_filename = f"{sim_filename}_{'waves' if wave_effects_enabled else 'nowaves'}"
        if current_drift_factor is not None:
            safe_cdf = f"{current_drift_factor:.2f}".replace(".", "p")
            sim_filename = f"{sim_filename}_cdf{safe_cdf}"
        if processes_dispersion is not None:
            sim_filename = f"{sim_filename}_{'disp' if processes_dispersion else 'nodisp'}"
        if processes_evaporation is not None:
            sim_filename = f"{sim_filename}_{'evap' if processes_evaporation else 'noevap'}"
        return sim_filename

    @staticmethod
    def _check_cancelled(should_cancel: Optional[Callable[[], bool]]) -> None:
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Execution cancelled by user.")

    def load_config(self, request: ConfigRequest):
        return self.environment_repository.compose_config(
            config_name=request.config_name,
            environment=request.environment,
            additional_overrides=request.to_overrides(),
        )

    def download_environment_data(
        self,
        environment: str,
        config_name: str = "main",
        force: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
        copernicus_username: Optional[str] = None,
        copernicus_password: Optional[str] = None,
        min_long: Optional[float] = None,
        max_long: Optional[float] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
    ) -> dict:
        config = self.load_config(
            ConfigRequest(
                config_name=config_name,
                environment=environment,
                simulation_name="sim4validation",
                min_long=min_long,
                max_long=max_long,
                min_lat=min_lat,
                max_lat=max_lat,
            )
        )
        return self.copernicus_gateway.download_environment_data(
            config=config,
            force=force,
            log_callback=log_callback,
            username=copernicus_username,
            password=copernicus_password,
        )

    @staticmethod
    def _dataset_coord_range(ds, names: tuple[str, ...]) -> tuple[float, float]:
        for name in names:
            if name in ds.coords or name in ds.variables:
                values = np.asarray(ds[name].values, dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    return float(finite.min()), float(finite.max())
        raise ValueError(f"Could not find coordinate names {names} in dataset.")

    @staticmethod
    def _has_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
        return not (a_max < b_min or a_min > b_max)

    @staticmethod
    def _normalize_path_list(
        singular_path: Optional[str],
        multiple_paths: Optional[list[str]] = None,
    ) -> list[Path]:
        values: list[str] = []
        if multiple_paths:
            values.extend(str(path).strip() for path in multiple_paths if str(path).strip())
        if singular_path and str(singular_path).strip():
            values.append(str(singular_path).strip())
        return list(dict.fromkeys(Path(value) for value in values))

    def _normalize_environmental_offset_values(
        self,
        values: Optional[list[float] | tuple[float, ...]],
    ) -> Optional[tuple[float, ...]]:
        if values is None:
            return None

        normalized: list[float] = []
        for value in values:
            offset = float(value)
            if not np.isfinite(offset):
                raise ValueError("environmental offset values must be finite")
            if abs(offset) > self.MAX_ENVIRONMENTAL_OFFSET_HOURS:
                raise ValueError(
                    f"environmental offset values must be between "
                    f"-{self.MAX_ENVIRONMENTAL_OFFSET_HOURS:g} and {self.MAX_ENVIRONMENTAL_OFFSET_HOURS:g}"
                )
            normalized.append(offset)

        if not normalized:
            return None
        return tuple(dict.fromkeys(normalized))

    def _valid_environmental_offsets(
        self,
        manchas,
        current_dataset_paths: tuple[Path, ...],
        sal_temp_dataset_path: Path,
        wind_dataset_paths: tuple[Path, ...],
        environmental_offset_values: tuple[float, ...],
    ) -> tuple[float, ...]:
        valid_offsets: list[float] = []
        first_error: Optional[Exception] = None
        for offset in environmental_offset_values:
            try:
                self._validate_environment_coverage(
                    manchas,
                    current_dataset_paths=current_dataset_paths,
                    wind_dataset_paths=wind_dataset_paths,
                    sal_temp_dataset_path=sal_temp_dataset_path,
                    environmental_offset_hours=offset,
                )
                valid_offsets.append(offset)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                print(f"Skipping environmental offset {offset:g} h: {exc}")

        if not valid_offsets:
            raise ValueError(
                "No environmental offset has valid forcing coverage."
                + (f" First error: {first_error}" if first_error else "")
            )
        return tuple(valid_offsets)

    def _validate_environment_coverage(
        self,
        manchas,
        current_dataset_paths: tuple[Path, ...],
        sal_temp_dataset_path: Path,
        wind_dataset_paths: tuple[Path, ...] = (),
        environmental_offset_hours: float = 0.0,
    ) -> None:
        obs_min_lon, obs_min_lat, obs_max_lon, obs_max_lat = [float(v) for v in manchas.total_bounds]
        obs_times = pd.to_datetime(manchas["datetime"], errors="coerce").dropna()
        obs_min_time = obs_times.min() if not obs_times.empty else None
        obs_max_time = obs_times.max() if not obs_times.empty else None

        required_groups: list[tuple[str, tuple[Path, ...]]] = [
            ("current", tuple(current_dataset_paths)),
            ("sal_temp", (Path(sal_temp_dataset_path),)),
        ]
        if wind_dataset_paths:
            required_groups.append(("wind", tuple(wind_dataset_paths)))

        for name, dataset_paths in required_groups:
            if not dataset_paths:
                raise ValueError(f"At least one dataset path is required for '{name}'.")

            agg_min_lon = float("inf")
            agg_max_lon = float("-inf")
            agg_min_lat = float("inf")
            agg_max_lat = float("-inf")
            agg_min_time: Optional[pd.Timestamp] = None
            agg_max_time: Optional[pd.Timestamp] = None

            for dataset_path in dataset_paths:
                if not dataset_path.exists():
                    raise ValueError(
                        f"Required dataset not found: {dataset_path}. "
                        "Provide current/wind files and download sal_temp before execution."
                    )
                with xr.open_dataset(dataset_path) as ds:
                    ds_min_lon, ds_max_lon = self._dataset_coord_range(ds, ("longitude", "lon", "x"))
                    ds_min_lat, ds_max_lat = self._dataset_coord_range(ds, ("latitude", "lat", "y"))
                    agg_min_lon = min(agg_min_lon, ds_min_lon)
                    agg_max_lon = max(agg_max_lon, ds_max_lon)
                    agg_min_lat = min(agg_min_lat, ds_min_lat)
                    agg_max_lat = max(agg_max_lat, ds_max_lat)

                    time_name = None
                    for candidate in ("time", "time1"):
                        if candidate in ds.coords or candidate in ds.variables:
                            time_name = candidate
                            break
                    if time_name is not None:
                        ds_times = pd.to_datetime(ds[time_name].values, errors="coerce")
                        ds_times = ds_times[~pd.isna(ds_times)]
                        if len(ds_times):
                            if environmental_offset_hours:
                                ds_times = ds_times + pd.Timedelta(hours=float(environmental_offset_hours))
                            ds_min_time = ds_times.min()
                            ds_max_time = ds_times.max()
                            agg_min_time = ds_min_time if agg_min_time is None else min(agg_min_time, ds_min_time)
                            agg_max_time = ds_max_time if agg_max_time is None else max(agg_max_time, ds_max_time)

            if not self._has_overlap(obs_min_lon, obs_max_lon, agg_min_lon, agg_max_lon) or not self._has_overlap(
                obs_min_lat, obs_max_lat, agg_min_lat, agg_max_lat
            ):
                raise ValueError(
                    f"Observed spill area is outside '{name}' dataset coverage. "
                    f"Observed lon/lat=[{obs_min_lon:.5f},{obs_max_lon:.5f}] / "
                    f"[{obs_min_lat:.5f},{obs_max_lat:.5f}], "
                    f"dataset lon/lat=[{agg_min_lon:.5f},{agg_max_lon:.5f}] / "
                    f"[{agg_min_lat:.5f},{agg_max_lat:.5f}]."
                )

            if obs_min_time is not None and obs_max_time is not None and agg_min_time is not None and agg_max_time is not None:
                if obs_max_time < agg_min_time or obs_min_time > agg_max_time:
                    raise ValueError(
                        f"Observed spill time window is outside '{name}' dataset time coverage. "
                        f"Observed=[{obs_min_time},{obs_max_time}], "
                        f"dataset=[{agg_min_time},{agg_max_time}]."
                    )

    def load_observed_spills(self, request: ObservedSpillRequest) -> ObservedSpillContext:
        manchas = gpd.read_file(Path(request.spill_path)).to_crs(epsg=4326)
        self.spill_repository.ensure_datetime_column(manchas, offset_hours=request.offset_hours)
        manchas.sort_values("datetime", inplace=True)

        unique_times = sorted(manchas["datetime"].unique())
        if request.start_index < 0 or request.start_index >= len(unique_times):
            raise ValueError(f"start-index out of range (0..{len(unique_times)-1})")
        if request.start_index:
            start_time = unique_times[request.start_index]
            manchas = manchas[manchas["datetime"] >= start_time].copy()

        minlon, minlat, maxlon, maxlat = manchas.total_bounds
        pad_lon = (maxlon - minlon) * request.padding_animation_frame
        pad_lat = (maxlat - minlat) * request.padding_animation_frame
        plot_bounds = (
            minlon - pad_lon,
            maxlon + pad_lon,
            minlat - pad_lat,
            maxlat + pad_lat,
        )
        return ObservedSpillContext(manchas=manchas, plot_bounds=plot_bounds)

    def build_observed_trajectory(self, manchas):
        return self.spill_repository.build_observed_trajectory(manchas)

    def optimize_wdf_cdf(
        self,
        manchas,
        config,
        observed_trajectory,
        wdf_values,
        current_drift_values,
        particles_per_wdf=1,
        oil_type=None,
        progress=None,
        should_cancel=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
        environmental_offset_hours=None,
    ):
        return self.optimization_service.fast_grid_search_wdf_cdf(
            manchas=manchas,
            config=config,
            observed_trajectory=observed_trajectory,
            wdf_values=wdf_values,
            current_drift_values=current_drift_values,
            particles_per_wdf=particles_per_wdf,
            oil_type=oil_type,
            progress=progress,
            should_cancel=should_cancel,
            forcing_source=forcing_source,
            current_dataset_path=current_dataset_path,
            wind_dataset_path=wind_dataset_path,
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
            environmental_offset_hours=environmental_offset_hours,
        )

    def optimize_wdf_cdf_request(self, request: FastOptimizationRequest):
        return self.optimize_wdf_cdf(
            manchas=request.manchas,
            config=request.config,
            observed_trajectory=request.observed_trajectory,
            wdf_values=request.wdf_values,
            current_drift_values=request.current_drift_values,
            particles_per_wdf=request.particles_per_wdf,
            oil_type=request.oil_type,
            progress=request.progress,
            should_cancel=getattr(request, "should_cancel", None),
            forcing_source=request.forcing_source,
            current_dataset_path=request.current_dataset_path,
            wind_dataset_path=request.wind_dataset_path,
            current_dataset_paths=request.current_dataset_paths,
            wind_dataset_paths=request.wind_dataset_paths,
            environmental_offset_hours=request.environmental_offset_hours,
        )

    def run_simulation(
        self,
        manchas,
        out_filename,
        config,
        skip_animation,
        padding_animation_frame,
        wind_drift_factor=None,
        current_drift_factor=None,
        oil_type=None,
        processes_dispersion=None,
        processes_evaporation=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        observed_offset_hours=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
        environmental_offset_hours=None,
    ):
        return self.simulation_service.simulate_drift(
            manchas=manchas,
            out_filename=out_filename,
            config=config,
            skip_animation=skip_animation,
            padding_animation_frame=padding_animation_frame,
            wind_drift_factor=wind_drift_factor,
            current_drift_factor=current_drift_factor,
            oil_type=oil_type,
            processes_dispersion=processes_dispersion,
            processes_evaporation=processes_evaporation,
            forcing_source=forcing_source,
            current_dataset_path=current_dataset_path,
            wind_dataset_path=wind_dataset_path,
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
            observed_offset_hours=observed_offset_hours,
            environmental_offset_hours=environmental_offset_hours,
        )

    def run_simulation_request(self, request: SimulationRunRequest):
        return self.run_simulation(
            manchas=request.manchas,
            out_filename=request.out_filename,
            config=request.config,
            skip_animation=request.skip_animation,
            padding_animation_frame=request.padding_animation_frame,
            wind_drift_factor=request.wind_drift_factor,
            current_drift_factor=request.current_drift_factor,
            oil_type=request.oil_type,
            processes_dispersion=request.processes_dispersion,
            processes_evaporation=request.processes_evaporation,
            forcing_source=request.forcing_source,
            current_dataset_path=request.current_dataset_path,
            wind_dataset_path=request.wind_dataset_path,
            current_dataset_paths=request.current_dataset_paths,
            wind_dataset_paths=request.wind_dataset_paths,
            observed_offset_hours=request.observed_offset_hours,
            environmental_offset_hours=request.environmental_offset_hours,
        )

    def generate_comparison_gif(self, **kwargs):
        return self.output_service.generate_comparison_gif(**kwargs)

    def generate_comparison_gif_request(self, request: GifGenerationRequest):
        return self.generate_comparison_gif(
            sim_nc=request.sim_nc,
            shp_zip=request.shp_zip,
            out=request.out,
            extent=request.extent,
            datetime_offset_hours=request.datetime_offset_hours,
            real_steps=request.real_steps,
            start_index=request.start_index,
        )

    def _build_validation_context(self, request: ValidationRunRequest) -> _ValidationContext:
        skip_animation = request.skip_animation
        skip_plots = request.skip_plots
        if request.evaluation:
            skip_animation = True
            skip_plots = True

        config = self.load_config(
            ConfigRequest(
                config_name=request.config_name,
                environment=request.environment,
                simulation_name="sim4validation",
                run_name=request.run_name,
                shp_zip=request.shp_zip,
                min_long=request.min_long,
                max_long=request.max_long,
                min_lat=request.min_lat,
                max_lat=request.max_lat,
            )
        )
        forcing_source = self._normalize_forcing_source(request.forcing_source)
        offset_hours = (
            0.0
            if request.disable_environment_offset
            else float(getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0) or 0.0)
        )
        environmental_offset_hours = (
            self.DEFAULT_ENVIRONMENTAL_OFFSET_HOURS
            if request.environmental_offset_hours is None
            else float(request.environmental_offset_hours)
        )
        if abs(environmental_offset_hours) > self.MAX_ENVIRONMENTAL_OFFSET_HOURS:
            raise ValueError(
                f"environmental-offset-hours must be between "
                f"-{self.MAX_ENVIRONMENTAL_OFFSET_HOURS:g} and {self.MAX_ENVIRONMENTAL_OFFSET_HOURS:g}"
            )
        environmental_offset_values = self._normalize_environmental_offset_values(
            request.environmental_offset_values
        )
        current_paths = self._normalize_path_list(
            request.current_dataset_path,
            list(request.current_dataset_paths) if request.current_dataset_paths else None,
        )
        if not current_paths:
            current_paths = [Path(config.copernicusmarine.specificities.water_dataset_path)]

        wind_paths = self._normalize_path_list(
            request.wind_dataset_path,
            list(request.wind_dataset_paths) if request.wind_dataset_paths else None,
        )
        if not wind_paths:
            config_wind_path = getattr(config.copernicusmarine.specificities, "wind_dataset_path", None)
            if config_wind_path:
                wind_paths = [Path(config_wind_path)]
        sal_temp_dataset_path = Path(config.copernicusmarine.specificities.sal_temp_dataset_path)
        observed_context = self.load_observed_spills(
            ObservedSpillRequest(
                spill_path=Path(config.paths.plataformas_shp),
                offset_hours=offset_hours,
                start_index=request.start_index,
                padding_animation_frame=request.padding_animation_frame,
            )
        )
        if environmental_offset_values is None:
            self._validate_environment_coverage(
                observed_context.manchas,
                current_dataset_paths=tuple(current_paths),
                wind_dataset_paths=tuple(wind_paths),
                sal_temp_dataset_path=sal_temp_dataset_path,
                environmental_offset_hours=environmental_offset_hours,
            )
        else:
            environmental_offset_values = self._valid_environmental_offsets(
                observed_context.manchas,
                current_dataset_paths=tuple(current_paths),
                wind_dataset_paths=tuple(wind_paths),
                sal_temp_dataset_path=sal_temp_dataset_path,
                environmental_offset_values=environmental_offset_values,
            )
            environmental_offset_hours = environmental_offset_values[0]

        selected_oil_types = self._load_oil_types(request.oil_types, request.oil_types_file)
        selected_oil_type = selected_oil_types[0] if selected_oil_types else None

        return _ValidationContext(
            config=config,
            offset_hours=offset_hours,
            environmental_offset_hours=environmental_offset_hours,
            forcing_source=forcing_source,
            current_dataset_path=current_paths[0],
            wind_dataset_path=wind_paths[0] if wind_paths else None,
            current_dataset_paths=tuple(current_paths),
            wind_dataset_paths=tuple(wind_paths),
            real_manchas=observed_context.manchas,
            plot_bounds=observed_context.plot_bounds,
            observed_trajectory=self.build_observed_trajectory(observed_context.manchas),
            environmental_offset_values=environmental_offset_values,
            start_index=int(request.start_index),
            skip_animation=skip_animation,
            skip_plots=skip_plots,
            wave_effects_enabled=False,
            wind_drift_factor=request.wind_drift_factor,
            current_drift_factor=request.current_drift_factor,
            processes_dispersion=self._parse_bool_string(request.processes_dispersion),
            processes_evaporation=self._parse_bool_string(request.processes_evaporation),
            selected_oil_type=selected_oil_type,
        )

    @staticmethod
    def _validate_supported_flags(request: ValidationRunRequest) -> None:
        if (
            request.optimize_wdf
            or request.optimize_physics
        ):
            raise ValueError(
                "This codebase was simplified. Use only --optimize-wdf-cdf "
                "with fast mode."
            )

    def _run_fast_optimization_phase(
        self,
        request: ValidationRunRequest,
        context: _ValidationContext,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> bool:
        if not request.optimize_wdf_cdf:
            return True
        self._check_cancelled(should_cancel)

        if context.wind_drift_factor is not None:
            print("Ignoring --wind-drift-factor because --optimize-wdf-cdf is set.")
        if context.current_drift_factor is not None:
            print("Ignoring --current-drift-factor because --optimize-wdf-cdf is set.")
        if request.wdf_step <= 0:
            raise ValueError("wdf-step must be > 0")
        if request.wdf_max < request.wdf_min:
            raise ValueError("wdf-max must be >= wdf-min")
        if request.cdf_step <= 0:
            raise ValueError("cdf-step must be > 0")
        if request.cdf_max < request.cdf_min:
            raise ValueError("cdf-max must be >= cdf-min")

        wdf_values = np.arange(
            request.wdf_min,
            request.wdf_max + (request.wdf_step / 2),
            request.wdf_step,
        )
        cdf_values = np.arange(
            request.cdf_min,
            request.cdf_max + (request.cdf_step / 2),
            request.cdf_step,
        )
        out_dir = self.workspace_repository.simulation_output_dir(context.config)
        mode = request.optimize_wdf_mode.lower()
        if mode != "fast":
            raise ValueError("Only fast mode is supported in the simplified optimizer.")
        if request.optimize_cleanup:
            print("Note: --optimize-cleanup is ignored in fast mode.")

        environmental_offset_values = (
            context.environmental_offset_values
            if context.environmental_offset_values is not None
            else (float(context.environmental_offset_hours),)
        )
        total_runs = len(environmental_offset_values) * len(cdf_values)
        print(
            f"Will test {len(environmental_offset_values)} environmental offsets x "
            f"{len(cdf_values)} cdf = {total_runs} simulations "
            f"(fast; all WDFs per run, waves disabled)."
        )
        optimization_time_step_minutes = max(
            5.0,
            float(getattr(context.config.simulation, "time_step_minutes", 1.0) or 1.0),
        )
        print(
            f"Optimization OpenDrift timestep: {optimization_time_step_minutes:g} min "
            f"(final simulation keeps {context.config.simulation.time_step_minutes:g} min)."
        )
        progress = _ProgressPrinter(
            total_runs,
            every=15,
            label="Progress",
            on_tick=progress_callback,
            should_cancel=should_cancel,
        )
        results_frames = []
        for environmental_offset_hours in environmental_offset_values:
            self._check_cancelled(should_cancel)
            print(f"Testing environmental offset {environmental_offset_hours:g} h...")
            _, offset_results_df = self.optimize_wdf_cdf(
                manchas=context.real_manchas,
                config=context.config,
                observed_trajectory=context.observed_trajectory,
                wdf_values=wdf_values,
                current_drift_values=cdf_values,
                particles_per_wdf=request.fast_particles_per_wdf,
                oil_type=context.selected_oil_type,
                progress=progress,
                should_cancel=should_cancel,
                forcing_source=context.forcing_source,
                current_dataset_path=str(context.current_dataset_path),
                wind_dataset_path=(str(context.wind_dataset_path) if context.wind_dataset_path else None),
                current_dataset_paths=[str(path) for path in context.current_dataset_paths],
                wind_dataset_paths=[str(path) for path in context.wind_dataset_paths],
                environmental_offset_hours=environmental_offset_hours,
            )
            if not offset_results_df.empty:
                offset_results_df = offset_results_df.copy()
                offset_results_df["environmental_offset_hours"] = float(environmental_offset_hours)
                results_frames.append(offset_results_df)

        results_df = (
            pd.concat(results_frames, ignore_index=True)
            if results_frames
            else pd.DataFrame(
                columns=[
                    "wind_drift_factor",
                    "skillscore",
                    "current_drift_factor",
                    "environmental_offset_hours",
                ]
            )
        )
        best_row = None
        if not results_df.empty and results_df["skillscore"].notna().any():
            best_row = results_df.loc[results_df["skillscore"].idxmax()]

        results_name = "wdf_cdf_optimization_fast"
        self.workspace_repository.write_csv(out_dir / f"{results_name}.csv", results_df)
        if best_row is None or pd.isna(best_row["skillscore"]):
            print("Combined optimization failed: no valid skillscore computed.")
            return False

        context.wind_drift_factor = float(best_row["wind_drift_factor"])
        context.current_drift_factor = float(best_row["current_drift_factor"])
        context.environmental_offset_hours = float(best_row["environmental_offset_hours"])
        if "oil_type" in best_row:
            context.selected_oil_type = str(best_row["oil_type"])

        summary = {
            "wind_drift_factor": context.wind_drift_factor,
            "wave_effects_enabled": False,
            "current_drift_factor": context.current_drift_factor,
            "environmental_offset_hours": context.environmental_offset_hours,
            "skillscore": float(best_row["skillscore"]),
            "wdf_min": float(request.wdf_min),
            "wdf_max": float(request.wdf_max),
            "wdf_step": float(request.wdf_step),
            "cdf_min": float(request.cdf_min),
            "cdf_max": float(request.cdf_max),
            "cdf_step": float(request.cdf_step),
            "environmental_offset_values": [
                float(value) for value in environmental_offset_values
            ],
            "mode": mode,
        }
        if context.selected_oil_type:
            summary["oil_type"] = context.selected_oil_type
        if mode == "fast":
            summary["particles_per_wdf"] = int(request.fast_particles_per_wdf)
        self.workspace_repository.write_json(out_dir / f"{results_name}.json", summary)
        print(
            f"Best wdf/cdf: {context.wind_drift_factor:.4f} "
            f"cdf={context.current_drift_factor:.2f} "
            f"env_offset={context.environmental_offset_hours:+g}h "
            f"oil={context.selected_oil_type or 'default'} "
            f"(skillscore {best_row['skillscore']:.4f})"
        )
        return True

    def _run_simulation_phase(
        self,
        request: ValidationRunRequest,
        context: _ValidationContext,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> tuple[Path, Path]:
        print("Start simulation...")
        self._check_cancelled(should_cancel)

        sim_filename = self._build_sim_filename(
            "sim_2019_P53_TEST_NOWAVES_30WDF_75CDF",
            wind_drift_factor=context.wind_drift_factor,
            wave_effects_enabled=context.wave_effects_enabled,
            current_drift_factor=context.current_drift_factor,
            processes_dispersion=context.processes_dispersion,
            processes_evaporation=context.processes_evaporation,
        )

        out_dir = self.workspace_repository.simulation_output_dir(context.config)
        run_params = {
            "environment": request.environment,
            "simulation_name": context.config.simulation.name,
            "forcing_source": context.forcing_source,
            "start_index": int(request.start_index),
        }
        run_params["observed_start_timestep"] = int(request.start_index)
        run_params["environmental_offset_hours"] = float(context.environmental_offset_hours)
        if context.wind_drift_factor is not None:
            run_params["wind_drift_factor"] = float(context.wind_drift_factor)
        run_params["wave_effects_enabled"] = False
        if context.current_drift_factor is not None:
            run_params["current_drift_factor"] = float(context.current_drift_factor)
        run_params["current_dataset_path"] = str(context.current_dataset_path)
        run_params["current_dataset_paths"] = [str(path) for path in context.current_dataset_paths]
        if context.wind_dataset_path is not None:
            run_params["wind_dataset_path"] = str(context.wind_dataset_path)
        if context.wind_dataset_paths:
            run_params["wind_dataset_paths"] = [str(path) for path in context.wind_dataset_paths]
        run_params["observed_offset_hours"] = float(context.offset_hours)
        if context.selected_oil_type:
            run_params["oil_type"] = context.selected_oil_type
        if context.processes_dispersion is not None:
            run_params["processes_dispersion"] = bool(context.processes_dispersion)
        if context.processes_evaporation is not None:
            run_params["processes_evaporation"] = bool(context.processes_evaporation)
        self.workspace_repository.write_json(out_dir / f"{sim_filename}.json", run_params)

        if not request.skip_simulation:
            self._check_cancelled(should_cancel)
            self.run_simulation_request(
                SimulationRunRequest(
                    manchas=context.real_manchas,
                    out_filename=sim_filename,
                    config=context.config,
                    skip_animation=context.skip_animation,
                    padding_animation_frame=request.padding_animation_frame,
                    wind_drift_factor=context.wind_drift_factor,
                    current_drift_factor=context.current_drift_factor,
                    oil_type=context.selected_oil_type,
                    processes_dispersion=context.processes_dispersion,
                    processes_evaporation=context.processes_evaporation,
                    forcing_source=context.forcing_source,
                    current_dataset_path=str(context.current_dataset_path),
                    wind_dataset_path=(str(context.wind_dataset_path) if context.wind_dataset_path else None),
                    current_dataset_paths=[str(path) for path in context.current_dataset_paths],
                    wind_dataset_paths=[str(path) for path in context.wind_dataset_paths],
                    observed_offset_hours=float(context.offset_hours),
                    environmental_offset_hours=float(context.environmental_offset_hours),
                )
            )
            print(f"The results have been generated in {out_dir}")
        else:
            print(f"The results are probably already present in {out_dir}")

        return out_dir / f"{sim_filename}.nc", out_dir

    def _run_visualization_phase(
        self,
        context: _ValidationContext,
        sim_path: Path,
        out_dir: Path,
        should_cancel: Optional[Callable[[], bool]] = None,
        show_plots: bool = True,
    ) -> tuple[Optional[Path], Optional[Path]]:
        self._check_cancelled(should_cancel)
        compare_gif: Optional[Path] = None
        frames_dir: Optional[Path] = None
        if not context.skip_animation:
            compare_gif = out_dir / f"{sim_path.stem}_compare.gif"
            try:
                real_steps = int(context.real_manchas["datetime"].nunique())
                self.generate_comparison_gif_request(
                    GifGenerationRequest(
                        sim_nc=sim_path,
                        shp_zip=context.config.paths.plataformas_shp,
                        out=compare_gif,
                        extent=",".join(f"{value:.6f}" for value in context.plot_bounds),
                        datetime_offset_hours=context.offset_hours,
                        real_steps=real_steps,
                        start_index=int(context.start_index),
                    )
                )
            except Exception as exc:
                print(f"Failed to generate comparison GIF: {exc}")

        if context.skip_plots:
            return compare_gif, frames_dir

        ds_result = xr.open_dataset(sim_path, engine="netcdf4")
        snapshot_datetimes = list(context.observed_trajectory["time"])
        frames_dir = out_dir / f"{sim_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for dt_snapshot in snapshot_datetimes:
            self._check_cancelled(should_cancel)
            sim_times = pd.to_datetime(ds_result["time"].values)
            idx_time = (abs(sim_times - dt_snapshot)).argmin()
            lons = ds_result["lon"].isel(time=idx_time).values
            lats = ds_result["lat"].isel(time=idx_time).values
            gdf_sim = gpd.GeoDataFrame(
                {"lon": lons, "lat": lats},
                geometry=[Point(xy) for xy in zip(lons, lats)],
                crs="EPSG:4326",
            )
            manchas_at_time = context.real_manchas[
                context.real_manchas["datetime"] == dt_snapshot
            ]
            _, ax = plt.subplots(figsize=(8, 6))
            gdf_sim.plot(ax=ax, color="blue", markersize=5, label="Simulated particles")
            manchas_at_time.plot(ax=ax, color="red", alpha=0.5, label="Observed spill")
            plt.title(f"Oil spill comparison at {dt_snapshot}")
            plt.legend()
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True)
            timestamp_label = pd.to_datetime(dt_snapshot).strftime("%Y%m%d_%H%M%S")
            frame_file = frames_dir / f"step_{timestamp_label}.png"
            plt.savefig(frame_file, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close()
        return compare_gif, frames_dir

    def run_validation(
        self,
        request: ValidationRunRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        show_plots: bool = True,
    ) -> Optional[ValidationRunResult]:
        self._check_cancelled(should_cancel)
        context = self._build_validation_context(request)
        self._validate_supported_flags(request)
        if not self._run_fast_optimization_phase(
            request,
            context,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
        ):
            return None
        sim_path, out_dir = self._run_simulation_phase(request, context, should_cancel=should_cancel)
        compare_gif, frames_dir = self._run_visualization_phase(
            context,
            sim_path,
            out_dir,
            should_cancel=should_cancel,
            show_plots=show_plots,
        )
        artifact_paths = tuple(sorted((path for path in out_dir.iterdir() if path.is_file()), key=lambda p: p.name))
        return ValidationRunResult(
            run_name=context.config.simulation.name,
            out_dir=out_dir,
            sim_path=sim_path,
            wind_drift_factor=context.wind_drift_factor,
            wave_effects_enabled=False,
            current_drift_factor=context.current_drift_factor,
            environmental_offset_hours=context.environmental_offset_hours,
            oil_type=context.selected_oil_type,
            comparison_gif=compare_gif,
            frames_dir=frames_dir,
            artifact_paths=artifact_paths,
        )
