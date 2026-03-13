import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import geopandas as gpd
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

from src.infrastructure.forcing_dataset_adapter import ForcingDatasetAdapter
from src.infrastructure.spill_repository import SpillRepository
from utils.aux_func import generate_random_points_in_polygon


class SimulationService:
    """Execute OpenDrift simulations using domain parameters."""
    CURRENT_TIME_OFFSET_HOURS = -3
    WIND_TIME_OFFSET_HOURS = -3
    SAL_TEMP_TIME_OFFSET_HOURS = -3

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        forcing_dataset_adapter: Optional[ForcingDatasetAdapter] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
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

    def simulate_drift(
        self,
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
        offset_hours = observed_offset_hours
        if offset_hours is None:
            offset_hours = float(
                getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0) or 0.0
            )
        self.spill_repository.ensure_datetime_column(manchas, offset_hours=offset_hours)
        manchas.sort_values("datetime", inplace=True)

        seed_time_start = manchas["datetime"].iloc[0]
        initial_subset = manchas[manchas["datetime"] == seed_time_start]
        mancha_inicial_geo = (
            initial_subset.geometry.union_all()
            if hasattr(initial_subset.geometry, "union_all")
            else initial_subset.geometry.unary_union
        )
        end_time = manchas["datetime"].max()

        print("First arrival datetime:", seed_time_start)
        print("Most recent spill datetime:", end_time)

        points = generate_random_points_in_polygon(
            mancha_inicial_geo, config.simulation.num_seed_elements
        )
        elements = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

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

        minlon, minlat, maxlon, maxlat = manchas.total_bounds
        padding_lon = (maxlon - minlon) * padding_animation_frame
        padding_lat = (maxlat - minlat) * padding_animation_frame
        min_lon = minlon - padding_lon
        max_lon = maxlon + padding_lon
        min_lat = minlat - padding_lat
        max_lat = maxlat + padding_lat

        print("Seeding elements...")
        model.set_config("drift:advection_scheme", "runge-kutta4")
        # Wave effects are disabled in this project configuration.
        model.set_config("drift:stokes_drift", False)

        default_wdf = 0.015
        wdf = wind_drift_factor
        if wdf is None:
            wdf = getattr(config.simulation, "wind_drift_factor", default_wdf)
        model.set_config("seed:wind_drift_factor", wdf)

        if current_drift_factor is None:
            current_drift_factor = getattr(config.simulation, "current_drift_factor", None)
        if current_drift_factor is not None:
            model.set_config("seed:current_drift_factor", float(current_drift_factor))
        if horizontal_diffusivity is not None:
            model.set_config("drift:horizontal_diffusivity", float(horizontal_diffusivity))
        if processes_dispersion is not None:
            model.set_config("processes:dispersion", bool(processes_dispersion))
        if processes_evaporation is not None:
            model.set_config("processes:evaporation", bool(processes_evaporation))

        selected_oil = oil_type or getattr(config.simulation, "oil_type", "SOCKEYE SWEET")
        model.seed_from_geopandas(
            geodataframe=elements,
            time=seed_time_start,
            oil_type=selected_oil,
        )

        _ = seed_time_start + timedelta(days=config.simulation.duration_days)

        if os.path.exists(Path(config.paths.simulation_data) / config.simulation.name):
            output_filename = (
                Path(config.paths.simulation_data)
                / config.simulation.name
                / f"{out_filename}.nc"
            )
            print("Running...")
            model.prepare_run()
            model.run(
                end_time=end_time,
                time_step=config.simulation.time_step_minutes * 60,
                time_step_output=config.simulation.output_time_step_minutes * 60,
                outfile=str(output_filename.absolute()),
            )
            model.elements
        else:
            print("Erro o.run(): o path do arquivo de saída provavelmente não é correto.")

        print(f"Simulation {out_filename} completed successfully.")
        if skip_animation:
            return model

        animation_filename = (
            Path(config.paths.simulation_data)
            / config.simulation.name
            / f"{out_filename}.gif"
        )
        model.animation(
            filename=str(animation_filename.absolute()),
            background=["x_sea_water_velocity", "y_sea_water_velocity"],
            corners=[min_lon, max_lon, min_lat, max_lat],
            vmin=-1,
            vmax=1,
            fast=True,
            fps=6,
        )
        return model
