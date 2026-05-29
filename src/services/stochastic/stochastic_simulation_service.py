from __future__ import annotations

import copy
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from src.services.conversion.drift_raster_converter import DriftRasterConverter
from src.services.simulation_service import SimulationService
from src.services.stochastic.ensemble_aggregator import EnsembleAggregationResult, EnsembleAggregator
from src.services.stochastic.parameter_sampler import ParameterSampler
from src.services.stochastic.stochastic_models import (
    SampledParameterSet,
    StochasticRunConfig,
    StochasticRunResult,
)
from src.services.stochastic.stochastic_output_service import StochasticOutputService


class StochasticSimulationService:
    """Run a Monte Carlo ensemble as repeated deterministic OpenDrift simulations."""

    def __init__(
        self,
        simulation_service: Optional[SimulationService] = None,
        parameter_sampler: Optional[ParameterSampler] = None,
        converter: Optional[DriftRasterConverter] = None,
        aggregator: Optional[EnsembleAggregator] = None,
        output_service: Optional[StochasticOutputService] = None,
    ) -> None:
        self.simulation_service = simulation_service or SimulationService()
        self.parameter_sampler = parameter_sampler or ParameterSampler()
        self.converter = converter or DriftRasterConverter()
        self.aggregator = aggregator or EnsembleAggregator(self.converter)
        self.output_service = output_service or StochasticOutputService()

    def run(
        self,
        manchas,
        base_config,
        stochastic_config: StochasticRunConfig,
        padding_animation_frame: float,
        forcing_source: str = "COPERNICUS",
        current_dataset_paths: Optional[Sequence[str]] = None,
        wind_dataset_paths: Optional[Sequence[str]] = None,
        observed_offset_hours: Optional[float] = None,
        environmental_offset_hours: Optional[float] = None,
        oil_type: Optional[str] = None,
        processes_dispersion: Optional[bool] = None,
        processes_evaporation: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> StochasticRunResult:
        start = time.monotonic()
        paths = self.output_service.prepare_run_directory(stochastic_config)
        logs_path = paths["logs_path"]

        def log(message: str) -> None:
            print(message)
            self.output_service.append_log(logs_path, message)
            if log_callback is not None:
                log_callback(message)

        def check_cancelled() -> None:
            if should_cancel is not None and should_cancel():
                raise RuntimeError("Execution cancelled by user.")

        self.parameter_sampler.validate_run_config(stochastic_config)
        grid = self.converter.fixed_grid_from_config(stochastic_config.grid)
        samples = self.parameter_sampler.sample(stochastic_config)

        self.output_service.write_config(stochastic_config, paths["config_path"])
        self.output_service.write_samples(samples, paths["sampled_parameters_path"])
        log(
            f"Starting stochastic run '{stochastic_config.run_name}' with "
            f"{stochastic_config.n_simulations} deterministic simulations."
        )
        log(
            f"Fixed grid: lon=[{grid.lon_min:.6f},{grid.lon_max:.6f}] "
            f"lat=[{grid.lat_min:.6f},{grid.lat_max:.6f}] "
            f"resolution={grid.spatial_resolution:g} crs={grid.crs}"
        )

        updated_samples: list[SampledParameterSet] = []
        successful_paths: list[Path] = []
        hit_count: Optional[np.ndarray] = None
        hit_count_final_timestep: Optional[np.ndarray] = None
        hourly_hit_counts: list[np.ndarray] = []
        hourly_time_indices: list[int] = []
        hourly_time_labels: list[str] = []
        valid_simulations = 0

        for sample in samples:
            check_cancelled()
            run_name = f"run_{sample.simulation_id:04d}"
            member_config = self._build_member_config(
                base_config=base_config,
                simulation_data_root=paths["individual_runs_dir"],
                simulation_name=run_name,
            )
            output_dir = paths["individual_runs_dir"] / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            out_filename = "simulation"
            output_path = output_dir / f"{out_filename}.nc"

            log(
                f"[{sample.simulation_id + 1}/{len(samples)}] "
                f"cdf={sample.cdf:.6g} wdf={sample.wdf:.6g} "
                f"tau={sample.temporal_lag_seconds:.0f}s seed={sample.seed}"
            )

            try:
                self.simulation_service.simulate_drift(
                    manchas=manchas.copy(),
                    out_filename=out_filename,
                    config=member_config,
                    skip_animation=True,
                    padding_animation_frame=padding_animation_frame,
                    wind_drift_factor=sample.wdf,
                    current_drift_factor=sample.cdf,
                    oil_type=oil_type,
                    processes_dispersion=processes_dispersion,
                    processes_evaporation=processes_evaporation,
                    forcing_source=forcing_source,
                    current_dataset_paths=current_dataset_paths,
                    wind_dataset_paths=wind_dataset_paths,
                    observed_offset_hours=observed_offset_hours,
                    environmental_offset_hours=environmental_offset_hours,
                    temporal_lag_seconds=sample.temporal_lag_seconds,
                )

                if not output_path.exists():
                    raise FileNotFoundError(f"Simulation output was not created: {output_path}")

                binary = self.converter.convert_simulation_to_binary_array(output_path, grid)
                hourly_binaries = self.converter.convert_simulation_to_binary_arrays_by_time(
                    output_path, grid, start_time_index=1
                )
                if hourly_binaries:
                    binary_final_timestep = hourly_binaries[-1][2]
                else:
                    binary_final_timestep = self.converter.convert_simulation_to_binary_array(
                        output_path,
                        grid,
                        time_index=-1,
                    )
                if hit_count is None:
                    hit_count = np.zeros(binary.shape, dtype=np.uint32)
                if hit_count_final_timestep is None:
                    hit_count_final_timestep = np.zeros(binary_final_timestep.shape, dtype=np.uint32)
                hit_count += binary.astype(np.uint32)
                hit_count_final_timestep += binary_final_timestep.astype(np.uint32)

                if hourly_binaries:
                    if not hourly_hit_counts:
                        hourly_time_indices = [time_index for time_index, _, _ in hourly_binaries]
                        hourly_time_labels = [time_label for _, time_label, _ in hourly_binaries]
                        hourly_hit_counts = [
                            np.zeros(binary_by_time.shape, dtype=np.uint32)
                            for _, _, binary_by_time in hourly_binaries
                        ]
                    elif len(hourly_binaries) != len(hourly_hit_counts):
                        raise ValueError(
                            "Simulation output time count differs from previous valid members. "
                            f"Expected {len(hourly_hit_counts)} timesteps after initial, "
                            f"got {len(hourly_binaries)}."
                        )

                    for hour_index, (_, _, binary_by_time) in enumerate(hourly_binaries):
                        if binary_by_time.shape != hourly_hit_counts[hour_index].shape:
                            raise ValueError(
                                "Hourly raster shape differs from previous valid members. "
                                f"Expected {hourly_hit_counts[hour_index].shape}, "
                                f"got {binary_by_time.shape}."
                            )
                        hourly_hit_counts[hour_index] += binary_by_time.astype(np.uint32)

                valid_simulations += 1
                successful_paths.append(output_path)
                updated_samples.append(
                    replace(sample, status="success", output_path=str(output_path))
                )
            except Exception as exc:
                log(f"Simulation {sample.simulation_id} failed: {exc}")
                updated_samples.append(
                    replace(sample, status="failed", error_message=str(exc), output_path=str(output_path))
                )

            self.output_service.write_samples(updated_samples + samples[len(updated_samples):], paths["sampled_parameters_path"])
            if progress_callback is not None:
                progress_callback(len(updated_samples), len(samples))

        check_cancelled()
        hit_count_map_path = None
        probability_map_path = None
        hit_count_final_timestep_map_path = None
        probability_final_timestep_map_path = None
        hourly_probability_map_paths: list[Path] = []

        if valid_simulations:
            probability_map = hit_count.astype(np.float32) / float(valid_simulations)
            aggregation = self.aggregator.save_maps(
                EnsembleAggregationResult(
                    valid_simulations=valid_simulations,
                    hit_count=hit_count,
                    probability_map=probability_map,
                ),
                paths["aggregated_dir"],
                grid,
            )
            hit_count_map_path = aggregation.hit_count_map_path
            probability_map_path = aggregation.probability_map_path
            probability_final_timestep_map = (
                hit_count_final_timestep.astype(np.float32) / float(valid_simulations)
            )
            final_timestep_aggregation = self.aggregator.save_maps(
                EnsembleAggregationResult(
                    valid_simulations=valid_simulations,
                    hit_count=hit_count_final_timestep,
                    probability_map=probability_final_timestep_map,
                ),
                paths["aggregated_dir"],
                grid,
                hit_count_filename="hit_count_final_timestep_map.tif",
                probability_filename="probability_final_timestep_map.tif",
            )
            hit_count_final_timestep_map_path = final_timestep_aggregation.hit_count_map_path
            probability_final_timestep_map_path = final_timestep_aggregation.probability_map_path
            hourly_probability_map_paths = self._save_hourly_probability_rasters(
                hourly_hit_counts=hourly_hit_counts,
                hourly_time_indices=hourly_time_indices,
                hourly_time_labels=hourly_time_labels,
                valid_simulations=valid_simulations,
                output_dir=paths["aggregated_dir"] / "hourly_probability_rasters",
                grid=grid,
            )
            log(
                f"Aggregated {valid_simulations} valid simulations "
                f"into raster, final-timestep, and {len(hourly_probability_map_paths)} hourly maps."
            )
        else:
            log("No valid simulations available for aggregation.")

        failed_simulations = len(samples) - valid_simulations
        summary = {
            "run_name": stochastic_config.run_name,
            "total_simulations": len(samples),
            "successful_simulations": valid_simulations,
            "failed_simulations": failed_simulations,
            "output_path": str(paths["run_dir"]),
            "sampled_parameters_path": str(paths["sampled_parameters_path"]),
            "summary_path": str(paths["summary_path"]),
            "hit_count_map_path": str(hit_count_map_path) if hit_count_map_path else None,
            "probability_map_path": str(probability_map_path) if probability_map_path else None,
            "hit_count_final_timestep_map_path": (
                str(hit_count_final_timestep_map_path)
                if hit_count_final_timestep_map_path
                else None
            ),
            "probability_final_timestep_map_path": (
                str(probability_final_timestep_map_path)
                if probability_final_timestep_map_path
                else None
            ),
            "hourly_probability_map_paths": [
                str(path) for path in hourly_probability_map_paths
            ],
            "individual_output_paths": [str(path) for path in successful_paths],
            "elapsed_seconds": time.monotonic() - start,
            "stochastic_config": asdict(stochastic_config),
        }
        self.output_service.write_summary(summary, paths["summary_path"])
        log(
            f"Finished stochastic run: {valid_simulations} success, "
            f"{failed_simulations} failed, output={paths['run_dir']}"
        )

        return StochasticRunResult(
            run_name=stochastic_config.run_name,
            total_simulations=len(samples),
            successful_simulations=valid_simulations,
            failed_simulations=failed_simulations,
            output_path=paths["run_dir"],
            sampled_parameters_path=paths["sampled_parameters_path"],
            summary_path=paths["summary_path"],
            hit_count_map_path=hit_count_map_path,
            probability_map_path=probability_map_path,
            hit_count_final_timestep_map_path=hit_count_final_timestep_map_path,
            probability_final_timestep_map_path=probability_final_timestep_map_path,
            hourly_probability_map_paths=tuple(hourly_probability_map_paths),
        )

    @staticmethod
    def _build_member_config(base_config, simulation_data_root: Path, simulation_name: str):
        member_config = copy.deepcopy(base_config)
        member_config.paths.simulation_data = str(simulation_data_root)
        member_config.simulation.name = simulation_name
        return member_config

    def _save_hourly_probability_rasters(
        self,
        hourly_hit_counts: list[np.ndarray],
        hourly_time_indices: list[int],
        hourly_time_labels: list[str],
        valid_simulations: int,
        output_dir: Path,
        grid,
    ) -> list[Path]:
        output_paths: list[Path] = []
        for position, hit_count_by_time in enumerate(hourly_hit_counts):
            time_index = hourly_time_indices[position]
            time_label = self._safe_filename_label(hourly_time_labels[position])
            probability = hit_count_by_time.astype(np.float32) / float(valid_simulations)
            output_path = output_dir / f"probability_timestep_{time_index:03d}_{time_label}.tif"
            self.converter.save_geotiff(probability, output_path, grid, dtype=np.float32)
            output_paths.append(output_path)
        return output_paths

    @staticmethod
    def _safe_filename_label(value: str) -> str:
        label = "".join(char if char.isalnum() else "_" for char in str(value))
        return "_".join(part for part in label.split("_") if part) or "time"
