from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from opendrift.models.openoil import OpenOil
from shapely.geometry import Point

from src.application.dto import (
    ConfigRequest,
    FastOptimizationRequest,
    GifGenerationRequest,
    ObservedSpillContext,
    ObservedSpillRequest,
    SimulationRunRequest,
    ValidationRunRequest,
)
from src.infrastructure.environment_repository import EnvironmentRepository
from src.infrastructure.spill_repository import SpillRepository
from src.infrastructure.workspace_repository import WorkspaceRepository
from src.services.optimization_service import OptimizationService
from src.services.output_service import OutputService
from src.services.simulation_service import SimulationService


class _ProgressPrinter:
    def __init__(self, total, every=15, label="Progress"):
        self.total = int(total)
        self.every = int(every)
        self.label = label
        self.count = 0

    def tick(self, n=1):
        if self.total <= 0:
            return
        self.count += int(n)
        if self.count % self.every == 0 or self.count >= self.total:
            print(f"{self.label}: {self.count}/{self.total}")


@dataclass
class _ValidationContext:
    config: Any
    offset_hours: float
    real_manchas: Any
    plot_bounds: tuple[float, float, float, float]
    observed_trajectory: Any
    skip_animation: bool
    skip_plots: bool
    stokes_override: Optional[bool]
    stokes_drift: Optional[bool]
    wind_drift_factor: Optional[float]
    current_drift_factor: Optional[float]
    horizontal_diffusivity: Optional[float]
    processes_dispersion: Optional[bool]
    processes_evaporation: Optional[bool]
    selected_oil_type: Optional[str]


class SimulationController:
    """Orchestrate repositories and services for full simulation workflows."""

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        environment_repository: Optional[EnvironmentRepository] = None,
        workspace_repository: Optional[WorkspaceRepository] = None,
        optimization_service: Optional[OptimizationService] = None,
        simulation_service: Optional[SimulationService] = None,
        output_service: Optional[OutputService] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
        self.environment_repository = environment_repository or EnvironmentRepository()
        self.workspace_repository = workspace_repository or WorkspaceRepository()
        self.optimization_service = optimization_service or OptimizationService(
            spill_repository=self.spill_repository
        )
        self.simulation_service = simulation_service or SimulationService(
            spill_repository=self.spill_repository
        )
        self.output_service = output_service or OutputService()

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
        stokes_drift=None,
        current_drift_factor=None,
        horizontal_diffusivity=None,
        processes_dispersion=None,
        processes_evaporation=None,
    ):
        sim_filename = base_name
        if wind_drift_factor is not None:
            safe_wdf = f"{wind_drift_factor:.4f}".replace(".", "p")
            sim_filename = f"{sim_filename}_wdf{safe_wdf}"
        if stokes_drift is not None:
            sim_filename = f"{sim_filename}_{'stokes' if stokes_drift else 'nostokes'}"
        if current_drift_factor is not None:
            safe_cdf = f"{current_drift_factor:.2f}".replace(".", "p")
            sim_filename = f"{sim_filename}_cdf{safe_cdf}"
        if horizontal_diffusivity is not None:
            safe_hdiff = f"{horizontal_diffusivity:.2f}".replace(".", "p")
            sim_filename = f"{sim_filename}_hd{safe_hdiff}"
        if processes_dispersion is not None:
            sim_filename = f"{sim_filename}_{'disp' if processes_dispersion else 'nodisp'}"
        if processes_evaporation is not None:
            sim_filename = f"{sim_filename}_{'evap' if processes_evaporation else 'noevap'}"
        return sim_filename

    def load_config(self, request: ConfigRequest):
        return self.environment_repository.compose_config(
            config_name=request.config_name,
            environment=request.environment,
            additional_overrides=request.to_overrides(),
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

    def optimize_wdf_stokes_current_drift(
        self,
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
        return self.optimization_service.fast_grid_search_wdf_stokes_current_drift(
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

    def optimize_wdf_stokes_current_drift_request(self, request: FastOptimizationRequest):
        return self.optimize_wdf_stokes_current_drift(
            manchas=request.manchas,
            config=request.config,
            observed_trajectory=request.observed_trajectory,
            wdf_values=request.wdf_values,
            current_drift_values=request.current_drift_values,
            horizontal_diffusivity_values=request.horizontal_diffusivity_values,
            particles_per_wdf=request.particles_per_wdf,
            oil_type=request.oil_type,
            progress=request.progress,
        )

    def run_simulation(
        self,
        manchas,
        out_filename,
        config,
        skip_animation,
        padding_animation_frame,
        wind_drift_factor=None,
        stokes_drift=None,
        current_drift_factor=None,
        oil_type=None,
        horizontal_diffusivity=None,
        processes_dispersion=None,
        processes_evaporation=None,
    ):
        return self.simulation_service.simulate_drift(
            manchas=manchas,
            out_filename=out_filename,
            config=config,
            skip_animation=skip_animation,
            padding_animation_frame=padding_animation_frame,
            wind_drift_factor=wind_drift_factor,
            stokes_drift=stokes_drift,
            current_drift_factor=current_drift_factor,
            oil_type=oil_type,
            horizontal_diffusivity=horizontal_diffusivity,
            processes_dispersion=processes_dispersion,
            processes_evaporation=processes_evaporation,
        )

    def run_simulation_request(self, request: SimulationRunRequest):
        return self.run_simulation(
            manchas=request.manchas,
            out_filename=request.out_filename,
            config=request.config,
            skip_animation=request.skip_animation,
            padding_animation_frame=request.padding_animation_frame,
            wind_drift_factor=request.wind_drift_factor,
            stokes_drift=request.stokes_drift,
            current_drift_factor=request.current_drift_factor,
            oil_type=request.oil_type,
            horizontal_diffusivity=request.horizontal_diffusivity,
            processes_dispersion=request.processes_dispersion,
            processes_evaporation=request.processes_evaporation,
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
                shp_zip=request.shp_zip,
                min_long=request.min_long,
                max_long=request.max_long,
                min_lat=request.min_lat,
                max_lat=request.max_lat,
            )
        )
        offset_hours = float(
            getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0) or 0.0
        )
        observed_context = self.load_observed_spills(
            ObservedSpillRequest(
                spill_path=Path(config.paths.plataformas_shp),
                offset_hours=offset_hours,
                start_index=request.start_index,
                padding_animation_frame=request.padding_animation_frame,
            )
        )

        selected_oil_types = self._load_oil_types(request.oil_types, request.oil_types_file)
        selected_oil_type = selected_oil_types[0] if selected_oil_types else None

        horizontal_diffusivity = (
            float(request.horizontal_diffusivity)
            if request.horizontal_diffusivity is not None
            else None
        )

        return _ValidationContext(
            config=config,
            offset_hours=offset_hours,
            real_manchas=observed_context.manchas,
            plot_bounds=observed_context.plot_bounds,
            observed_trajectory=self.build_observed_trajectory(observed_context.manchas),
            skip_animation=skip_animation,
            skip_plots=skip_plots,
            stokes_override=self._parse_bool_string(request.stokes_drift),
            stokes_drift=None,
            wind_drift_factor=request.wind_drift_factor,
            current_drift_factor=request.current_drift_factor,
            horizontal_diffusivity=horizontal_diffusivity,
            processes_dispersion=self._parse_bool_string(request.processes_dispersion),
            processes_evaporation=self._parse_bool_string(request.processes_evaporation),
            selected_oil_type=selected_oil_type,
        )

    @staticmethod
    def _validate_supported_flags(request: ValidationRunRequest) -> None:
        if (
            request.optimize_wdf
            or request.optimize_stokes
            or request.optimize_wdf_stokes
            or request.optimize_physics
            or request.optimize_cdf_hd_de
        ):
            raise ValueError(
                "This codebase was simplified. Use only --optimize-wdf-stokes-cdf "
                "with fast mode (CDF/HD grid + fast WDF)."
            )

    def _run_fast_optimization_phase(
        self, request: ValidationRunRequest, context: _ValidationContext
    ) -> bool:
        if not request.optimize_wdf_stokes_cdf:
            return True

        if context.stokes_override is not None:
            print(f"Fixing stokes_drift to {context.stokes_override} for optimization.")
        if context.wind_drift_factor is not None:
            print("Ignoring --wind-drift-factor because --optimize-wdf-stokes-cdf is set.")
        if context.current_drift_factor is not None:
            print("Ignoring --current-drift-factor because --optimize-wdf-stokes-cdf is set.")
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
        if request.diffusivity_values is not None:
            hdiff_values = self._parse_float_list(request.diffusivity_values, [0.0])
        elif context.horizontal_diffusivity is not None:
            hdiff_values = [float(context.horizontal_diffusivity)]
        else:
            hdiff_values = [0.0]

        out_dir = self.workspace_repository.simulation_output_dir(context.config)
        mode = request.optimize_wdf_mode.lower()
        if mode != "fast":
            raise ValueError("Only fast mode is supported in the simplified optimizer.")
        if request.optimize_cleanup:
            print("Note: --optimize-cleanup is ignored in fast mode.")

        total_runs = len(cdf_values) * len(hdiff_values)
        print(
            f"Will test {len(cdf_values)} cdf x {len(hdiff_values)} hd "
            f"= {total_runs} simulations (fast; all WDFs per run, stokes fixed=False)."
        )
        progress = _ProgressPrinter(total_runs, every=15, label="Progress")
        best_row, results_df = self.optimize_wdf_stokes_current_drift_request(
            FastOptimizationRequest(
                manchas=context.real_manchas,
                config=context.config,
                observed_trajectory=context.observed_trajectory,
                wdf_values=wdf_values,
                current_drift_values=cdf_values,
                horizontal_diffusivity_values=hdiff_values,
                particles_per_wdf=request.fast_particles_per_wdf,
                oil_type=context.selected_oil_type,
                progress=progress,
            )
        )

        results_name = "wdf_cdf_hd_optimization_fast"
        self.workspace_repository.write_csv(out_dir / f"{results_name}.csv", results_df)
        if best_row is None or pd.isna(best_row["skillscore"]):
            print("Combined optimization failed: no valid skillscore computed.")
            return False

        context.wind_drift_factor = float(best_row["wind_drift_factor"])
        context.stokes_drift = False
        context.current_drift_factor = float(best_row["current_drift_factor"])
        if "horizontal_diffusivity" in best_row and pd.notna(best_row["horizontal_diffusivity"]):
            context.horizontal_diffusivity = float(best_row["horizontal_diffusivity"])
        if "oil_type" in best_row:
            context.selected_oil_type = str(best_row["oil_type"])

        summary = {
            "wind_drift_factor": context.wind_drift_factor,
            "stokes_drift": context.stokes_drift,
            "current_drift_factor": context.current_drift_factor,
            "skillscore": float(best_row["skillscore"]),
            "wdf_min": float(request.wdf_min),
            "wdf_max": float(request.wdf_max),
            "wdf_step": float(request.wdf_step),
            "cdf_min": float(request.cdf_min),
            "cdf_max": float(request.cdf_max),
            "cdf_step": float(request.cdf_step),
            "hdiff_values": hdiff_values,
            "mode": mode,
        }
        if context.horizontal_diffusivity is not None:
            summary["horizontal_diffusivity"] = float(context.horizontal_diffusivity)
        if context.selected_oil_type:
            summary["oil_type"] = context.selected_oil_type
        if mode == "fast":
            summary["particles_per_wdf"] = int(request.fast_particles_per_wdf)
        self.workspace_repository.write_json(out_dir / f"{results_name}.json", summary)
        print(
            f"Best wdf/stokes/cdf: {context.wind_drift_factor:.4f} "
            f"stokes={context.stokes_drift} "
            f"cdf={context.current_drift_factor:.2f} "
            f"hd={context.horizontal_diffusivity if context.horizontal_diffusivity is not None else 0.0:.2f} "
            f"oil={context.selected_oil_type or 'default'} "
            f"(skillscore {best_row['skillscore']:.4f})"
        )
        return True

    def _run_simulation_phase(
        self, request: ValidationRunRequest, context: _ValidationContext
    ) -> tuple[Path, Path]:
        print("Start simulation...")
        if (
            context.stokes_override is not None
            and not request.optimize_stokes
            and not request.optimize_wdf_stokes
            and not request.optimize_wdf_stokes_cdf
        ):
            context.stokes_drift = context.stokes_override

        sim_filename = self._build_sim_filename(
            "sim_2019_P53_TEST_NOSTOKES_30WDF_75CDF",
            wind_drift_factor=context.wind_drift_factor,
            stokes_drift=context.stokes_drift,
            current_drift_factor=context.current_drift_factor,
            horizontal_diffusivity=context.horizontal_diffusivity,
            processes_dispersion=context.processes_dispersion,
            processes_evaporation=context.processes_evaporation,
        )

        out_dir = self.workspace_repository.simulation_output_dir(context.config)
        run_params = {
            "environment": request.environment,
            "simulation_name": context.config.simulation.name,
        }
        if request.start_index:
            run_params["start_index"] = int(request.start_index)
        if context.wind_drift_factor is not None:
            run_params["wind_drift_factor"] = float(context.wind_drift_factor)
        if context.stokes_drift is not None:
            run_params["stokes_drift"] = bool(context.stokes_drift)
        if context.current_drift_factor is not None:
            run_params["current_drift_factor"] = float(context.current_drift_factor)
        if context.selected_oil_type:
            run_params["oil_type"] = context.selected_oil_type
        if context.horizontal_diffusivity is not None:
            run_params["horizontal_diffusivity"] = float(context.horizontal_diffusivity)
        if context.processes_dispersion is not None:
            run_params["processes_dispersion"] = bool(context.processes_dispersion)
        if context.processes_evaporation is not None:
            run_params["processes_evaporation"] = bool(context.processes_evaporation)
        self.workspace_repository.write_json(out_dir / f"{sim_filename}.json", run_params)

        if not request.skip_simulation:
            self.run_simulation_request(
                SimulationRunRequest(
                    manchas=context.real_manchas,
                    out_filename=sim_filename,
                    config=context.config,
                    skip_animation=context.skip_animation,
                    padding_animation_frame=request.padding_animation_frame,
                    wind_drift_factor=context.wind_drift_factor,
                    stokes_drift=context.stokes_drift,
                    current_drift_factor=context.current_drift_factor,
                    oil_type=context.selected_oil_type,
                    horizontal_diffusivity=context.horizontal_diffusivity,
                    processes_dispersion=context.processes_dispersion,
                    processes_evaporation=context.processes_evaporation,
                )
            )
            print(f"The results have been generated in {out_dir}")
        else:
            print(f"The results are probably already present in {out_dir}")

        return out_dir / f"{sim_filename}.nc", out_dir

    def _run_visualization_phase(
        self, context: _ValidationContext, sim_path: Path, out_dir: Path
    ) -> None:
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
                    )
                )
            except Exception as exc:
                print(f"Failed to generate comparison GIF: {exc}")

        if context.skip_plots:
            return

        ds_result = xr.open_dataset(sim_path, engine="netcdf4")
        snapshot_datetimes = list(context.observed_trajectory["time"])
        for dt_snapshot in snapshot_datetimes:
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
            plt.show()

    def run_validation(self, request: ValidationRunRequest):
        context = self._build_validation_context(request)
        self._validate_supported_flags(request)
        if not self._run_fast_optimization_phase(request, context):
            return
        sim_path, out_dir = self._run_simulation_phase(request, context)
        self._run_visualization_phase(context, sim_path, out_dir)
