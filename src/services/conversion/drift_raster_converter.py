from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from src.services.stochastic.stochastic_models import StochasticGridConfig


@dataclass(frozen=True)
class FixedGrid:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    spatial_resolution: float
    crs: str = "EPSG:4326"

    @property
    def width(self) -> int:
        return int(np.ceil((self.lon_max - self.lon_min) / self.spatial_resolution))

    @property
    def height(self) -> int:
        return int(np.ceil((self.lat_max - self.lat_min) / self.spatial_resolution))

    @property
    def transform(self):
        return from_origin(
            self.lon_min,
            self.lat_max,
            self.spatial_resolution,
            self.spatial_resolution,
        )

    def validate(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError("grid lon_min must be < lon_max")
        if self.lat_min >= self.lat_max:
            raise ValueError("grid lat_min must be < lat_max")
        if self.spatial_resolution <= 0:
            raise ValueError("grid spatial_resolution must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")


class DriftRasterConverter:
    """Convert OpenDrift particle positions to fixed-grid arrays."""

    @staticmethod
    def fixed_grid_from_config(config: StochasticGridConfig) -> FixedGrid:
        margin = float(config.margin or 0.0)
        grid = FixedGrid(
            lon_min=float(config.lon_min) - margin,
            lon_max=float(config.lon_max) + margin,
            lat_min=float(config.lat_min) - margin,
            lat_max=float(config.lat_max) + margin,
            spatial_resolution=float(config.spatial_resolution),
            crs=config.crs,
        )
        grid.validate()
        return grid

    def convert_simulation_to_binary_array(
        self,
        simulation_path: Path,
        grid: FixedGrid,
        time_index: Optional[int] = None,
    ) -> np.ndarray:
        count = self.convert_simulation_to_count_array(simulation_path, grid, time_index=time_index)
        return (count > 0).astype(np.uint8)

    def convert_simulation_to_count_array(
        self,
        simulation_path: Path,
        grid: FixedGrid,
        time_index: Optional[int] = None,
    ) -> np.ndarray:
        grid.validate()
        simulation_path = Path(simulation_path)
        with xr.open_dataset(simulation_path) as dataset:
            lon = dataset["lon"]
            lat = dataset["lat"]
            if time_index is not None:
                lon_values = lon.isel(time=time_index).values
                lat_values = lat.isel(time=time_index).values
            else:
                lon_values = lon.values
                lat_values = lat.values

        return self.rasterize_points(lon_values, lat_values, grid)

    def rasterize_points(
        self,
        lon_values,
        lat_values,
        grid: FixedGrid,
    ) -> np.ndarray:
        lon_array = np.ma.filled(np.asarray(lon_values), np.nan).astype(float).ravel()
        lat_array = np.ma.filled(np.asarray(lat_values), np.nan).astype(float).ravel()
        valid = (
            np.isfinite(lon_array)
            & np.isfinite(lat_array)
            & (lon_array >= grid.lon_min)
            & (lon_array <= grid.lon_max)
            & (lat_array >= grid.lat_min)
            & (lat_array <= grid.lat_max)
        )
        rows = np.floor((grid.lat_max - lat_array[valid]) / grid.spatial_resolution).astype(int)
        cols = np.floor((lon_array[valid] - grid.lon_min) / grid.spatial_resolution).astype(int)
        rows = np.clip(rows, 0, grid.height - 1)
        cols = np.clip(cols, 0, grid.width - 1)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)

        count = np.zeros((grid.height, grid.width), dtype=np.uint32)
        if inside.any():
            np.add.at(count, (rows[inside], cols[inside]), 1)
        return count

    @staticmethod
    def save_geotiff(array: np.ndarray, output_path: Path, grid: FixedGrid, dtype=None) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raster = np.asarray(array)
        if dtype is not None:
            raster = raster.astype(dtype)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=grid.crs,
            transform=grid.transform,
            compress="lzw",
        ) as dataset:
            dataset.write(raster, 1)
