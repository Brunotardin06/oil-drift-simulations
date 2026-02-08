import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import geopandas as gpd
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

from src.infrastructure.spill_repository import SpillRepository
from utils.aux_func import generate_random_points_in_polygon


class SimulationService:
    """Execute OpenDrift simulations using domain parameters."""

    def __init__(self, spill_repository: Optional[SpillRepository] = None) -> None:
        self.spill_repository = spill_repository or SpillRepository()

    def simulate_drift(
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

        model = OpenOil(loglevel=50)
        readers = [
            reader_netCDF_CF_generic.Reader(
                Path(config.copernicusmarine.specificities.water_dataset_path)
            ),
            reader_netCDF_CF_generic.Reader(
                Path(config.copernicusmarine.specificities.wind_dataset_path)
            ),
            reader_netCDF_CF_generic.Reader(
                Path(config.copernicusmarine.specificities.wave_dataset_path)
            ),
            reader_netCDF_CF_generic.Reader(
                Path(config.copernicusmarine.specificities.sal_temp_dataset_path)
            ),
        ]
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
        if stokes_drift is None:
            stokes_drift = getattr(config.simulation, "stokes_drift", True)
        model.set_config("drift:stokes_drift", bool(stokes_drift))

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
        model.set_config("wave_entrainment:entrainment_rate", "Li et al. (2017)")
        model.set_config(
            "wave_entrainment:droplet_size_distribution",
            "Johansen et al. (2015)",
        )

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

