from pathlib import Path
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

from src.infrastructure.forcing_dataset_adapter import ForcingDatasetAdapter
from src.infrastructure.spill_repository import SpillRepository
from src.services.metrics_service import MetricsService


class OptimizationService:
    """Run optimization loops for model parameters."""
    CURRENT_TIME_OFFSET_HOURS = -3
    WIND_TIME_OFFSET_HOURS = -3
    SAL_TEMP_TIME_OFFSET_HOURS = -3

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        metrics_service: Optional[MetricsService] = None,
        forcing_dataset_adapter: Optional[ForcingDatasetAdapter] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
        self.metrics_service = metrics_service or MetricsService()
        self.forcing_dataset_adapter = forcing_dataset_adapter or ForcingDatasetAdapter()

    @classmethod
    def _build_reader(
        cls,
        dataset_path: Path,
        current_offset: bool = False,
        wind_offset: bool = False,
        sal_temp_offset: bool = False,
    ):
        reader = reader_netCDF_CF_generic.Reader(dataset_path)
        if current_offset:
            reader.shift_start_time(
                reader.start_time + timedelta(hours=cls.CURRENT_TIME_OFFSET_HOURS)
            )
        if wind_offset:
            reader.shift_start_time(
                reader.start_time + timedelta(hours=cls.WIND_TIME_OFFSET_HOURS)
            )
        if sal_temp_offset:
            reader.shift_start_time(
                reader.start_time + timedelta(hours=cls.SAL_TEMP_TIME_OFFSET_HOURS)
            )
        return reader

    def fast_grid_search_wind_drift_factor(
        self,
        manchas,
        config,
        observed_trajectory,
        wdf_values,
        particles_per_wdf=1,
        current_drift_factor=None,
        horizontal_diffusivity=None,
        oil_type=None,
        progress=None,
        should_cancel=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
    ):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Execution cancelled by user.")
        if "datetime" not in manchas.columns:
            raise ValueError("Expected 'datetime' column in manchas")
        if particles_per_wdf < 1:
            raise ValueError("particles_per_wdf must be >= 1")

        wdf_values = np.array(wdf_values, dtype=float)
        wdf_values = wdf_values[np.isfinite(wdf_values)]
        if wdf_values.size == 0:
            raise ValueError("wdf_values must contain at least one finite value")

        # Use the first observed timestep as seed to keep behavior consistent
        # with the main simulation service and support short (2-step) cases.
        shape_inicial = manchas.iloc[0]
        seed_time_start = shape_inicial["datetime"]
        shape_final = manchas.loc[manchas["datetime"].idxmax()]
        end_time = shape_final["datetime"]

        obs = observed_trajectory[
            (observed_trajectory["time"] >= seed_time_start)
            & (observed_trajectory["time"] <= end_time)
        ].copy()
        if obs.empty or len(obs) < 2:
            raise ValueError("Observed trajectory has insufficient timestamps within the simulation window")

        lat_col, lon_col = self.spill_repository.find_lat_lon_columns(manchas)
        if lat_col and lon_col:
            lat_val = pd.to_numeric(shape_inicial[lat_col], errors="coerce")
            lon_val = pd.to_numeric(shape_inicial[lon_col], errors="coerce")
            if pd.isna(lat_val) or pd.isna(lon_val):
                raise ValueError("Invalid lat/lon values in shapefile for the initial timestep")
            start_lat = float(lat_val)
            start_lon = float(lon_val)
        elif hasattr(shape_inicial, "geometry") and shape_inicial.geometry.geom_type == "Point":
            start_lat = float(shape_inicial.geometry.y)
            start_lon = float(shape_inicial.geometry.x)
        else:
            raise ValueError(
                "Latitude/Longitude columns not found (e.g., Latitude/Longitude or lat/lon) "
                "and geometry is not Point. Provide lat/lon fields in the shapefile."
            )

        if current_dataset_paths:
            current_paths = [Path(path) for path in current_dataset_paths]
        elif current_dataset_path:
            current_paths = [Path(current_dataset_path)]
        else:
            current_paths = [Path(config.copernicusmarine.specificities.water_dataset_path)]

        if wind_dataset_paths:
            wind_paths = [Path(path) for path in wind_dataset_paths]
        elif wind_dataset_path:
            wind_paths = [Path(wind_dataset_path)]
        else:
            wind_paths = []
        forcing_source = (forcing_source or "COPERNICUS").strip().upper()
        current_paths = [
            self.forcing_dataset_adapter.prepare_path(path, forcing_source, "current")
            for path in current_paths
        ]
        wind_paths = [
            self.forcing_dataset_adapter.prepare_path(path, forcing_source, "wind")
            for path in wind_paths
        ]

        sal_temp_path = Path(config.copernicusmarine.specificities.sal_temp_dataset_path)

        model = OpenOil(loglevel=50)
        current_readers = [self._build_reader(path, current_offset=True) for path in current_paths]
        wind_readers = [self._build_reader(path, wind_offset=True) for path in wind_paths]
        # Prioritize newer forecast runs when files overlap in time.
        current_readers.sort(key=lambda reader: reader.start_time, reverse=True)
        wind_readers.sort(key=lambda reader: reader.start_time, reverse=True)
        readers = current_readers + wind_readers
        readers.append(self._build_reader(sal_temp_path, sal_temp_offset=True))
        model.add_reader(readers)
        model.set_config("drift:advection_scheme", "runge-kutta4")
        model.set_config("drift:stokes_drift", False)
        if current_drift_factor is not None:
            model.set_config("seed:current_drift_factor", float(current_drift_factor))
        if horizontal_diffusivity is not None:
            model.set_config("drift:horizontal_diffusivity", float(horizontal_diffusivity))

        wdf_array = np.repeat(wdf_values, particles_per_wdf)
        lon_array = np.full_like(wdf_array, start_lon, dtype=float)
        lat_array = np.full_like(wdf_array, start_lat, dtype=float)

        model.seed_elements(
            lon=lon_array,
            lat=lat_array,
            time=seed_time_start,
            wind_drift_factor=wdf_array,
            oil_type=oil_type or getattr(config.simulation, "oil_type", "SOCKEYE SWEET"),
        )

        model.prepare_run()
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Execution cancelled by user.")
        model.run(
            end_time=end_time,
            time_step=config.simulation.time_step_minutes * 60,
            time_step_output=config.simulation.output_time_step_minutes * 60,
        )
        if progress is not None:
            progress.tick()

        ds_result = model.result
        sim_times = pd.to_datetime(ds_result["time"].values)
        time_indices = [int((abs(sim_times - dt)).argmin()) for dt in obs["time"]]
        wdf_per_traj = ds_result["wind_drift_factor"].isel(time=0).values

        results = []
        for wdf in wdf_values:
            mask = np.isclose(wdf_per_traj, wdf, rtol=0, atol=1e-6)
            if not np.any(mask):
                results.append({"wind_drift_factor": float(wdf), "skillscore": float("nan")})
                continue

            rows = []
            for dt, idx in zip(obs["time"], time_indices):
                lons = ds_result["lon"].isel(time=idx).values[mask]
                lats = ds_result["lat"].isel(time=idx).values[mask]
                lons = np.ma.filled(lons, np.nan).astype(float).ravel()
                lats = np.ma.filled(lats, np.nan).astype(float).ravel()
                valid = np.isfinite(lons) & np.isfinite(lats)
                if not valid.any():
                    rows.append({"time": pd.to_datetime(dt), "lon": np.nan, "lat": np.nan})
                    continue
                rows.append(
                    {
                        "time": pd.to_datetime(dt),
                        "lon": float(np.nanmean(lons[valid])),
                        "lat": float(np.nanmean(lats[valid])),
                    }
                )

            sim_traj = pd.DataFrame(rows)
            score = self.metrics_service.liu_weissberg_skillscore(obs, sim_traj)
            results.append({"wind_drift_factor": float(wdf), "skillscore": float(score)})

        results_df = pd.DataFrame(results)
        best_row = None
        if not results_df.empty and results_df["skillscore"].notna().any():
            best_row = results_df.loc[results_df["skillscore"].idxmax()]
        return best_row, results_df

    def fast_grid_search_wdf_cdf_hd(
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
        should_cancel=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
    ):
        if isinstance(current_drift_values, (int, float, np.floating, np.integer)):
            current_drift_values = [float(current_drift_values)]
        current_drift_values = [float(value) for value in current_drift_values]
        if not current_drift_values:
            raise ValueError("current_drift_values must contain at least one value")

        if horizontal_diffusivity_values is None:
            horizontal_diffusivity_values = [None]
        elif isinstance(horizontal_diffusivity_values, (int, float, np.floating, np.integer)):
            horizontal_diffusivity_values = [float(horizontal_diffusivity_values)]
        else:
            horizontal_diffusivity_values = [float(value) for value in horizontal_diffusivity_values]
        if not horizontal_diffusivity_values:
            raise ValueError("horizontal_diffusivity_values must contain at least one value")

        results = []
        for current_drift_factor in current_drift_values:
            for horizontal_diffusivity in horizontal_diffusivity_values:
                if should_cancel is not None and should_cancel():
                    raise RuntimeError("Execution cancelled by user.")
                _, dataframe = self.fast_grid_search_wind_drift_factor(
                    manchas,
                    config,
                    observed_trajectory,
                    wdf_values,
                    particles_per_wdf=particles_per_wdf,
                    current_drift_factor=current_drift_factor,
                    horizontal_diffusivity=horizontal_diffusivity,
                    oil_type=oil_type,
                    progress=progress,
                    should_cancel=should_cancel,
                    forcing_source=forcing_source,
                    current_dataset_path=current_dataset_path,
                    wind_dataset_path=wind_dataset_path,
                    current_dataset_paths=current_dataset_paths,
                    wind_dataset_paths=wind_dataset_paths,
                )
                if dataframe.empty:
                    continue

                dataframe = dataframe.copy()
                dataframe["current_drift_factor"] = float(current_drift_factor)
                if horizontal_diffusivity is not None:
                    dataframe["horizontal_diffusivity"] = float(horizontal_diffusivity)
                results.append(dataframe)

        results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        best_row = None
        if not results_df.empty and results_df["skillscore"].notna().any():
            best_row = results_df.loc[results_df["skillscore"].idxmax()]
        return best_row, results_df
