from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


class ForcingDatasetAdapter:
    """Pre-process external forcing files for compatibility with OpenDrift readers."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = cache_dir or (project_root / "data" / "cache" / "forcing")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[tuple[str, str, str], Path] = {}

    def prepare_path(self, dataset_path: Path, forcing_source: str, dataset_kind: str) -> Path:
        source = (forcing_source or "COPERNICUS").strip().upper()
        kind = dataset_kind.strip().lower()
        path = Path(dataset_path)
        if source != "REMO":
            return path

        cache_key = (str(path.resolve()), source, kind)
        memoized = self._memory_cache.get(cache_key)
        if memoized is not None and memoized.exists():
            return memoized

        adapted_path = self._prepare_remo_path(path, kind)
        self._memory_cache[cache_key] = adapted_path
        return adapted_path

    def _prepare_remo_path(self, dataset_path: Path, dataset_kind: str) -> Path:
        if dataset_kind not in {"current", "wind"}:
            return dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Forcing dataset not found: {dataset_path}")

        file_stat = dataset_path.stat()
        cache_token = "|".join(
            [
                str(dataset_path.resolve()),
                str(file_stat.st_mtime_ns),
                str(file_stat.st_size),
                dataset_kind,
            ]
        )
        digest = hashlib.sha1(cache_token.encode("utf-8")).hexdigest()[:16]
        target = self.cache_dir / f"{dataset_path.stem}.{dataset_kind}.remo.{digest}.nc"
        if target.exists():
            return target

        with xr.open_dataset(dataset_path) as ds:
            adapted = self._adapt_remo_dataset(ds, dataset_kind)
            if adapted is None:
                return dataset_path
            adapted.load()
            tmp_path = target.with_suffix(".tmp.nc")
            adapted.to_netcdf(tmp_path)
            tmp_path.replace(target)
            adapted.close()
        return target

    def _adapt_remo_dataset(self, dataset: xr.Dataset, dataset_kind: str) -> Optional[xr.Dataset]:
        ds = dataset.copy()
        changed = False

        rename_map: dict[str, str] = {}
        if "time1" in ds.dims and "time" not in ds.dims:
            rename_map["time1"] = "time"
        if rename_map:
            ds = ds.rename(rename_map)
            changed = True

        if "depth" in ds.dims and ds.sizes.get("depth", 0) == 1:
            ds = ds.squeeze(dim="depth", drop=True)
            changed = True
        elif "depth" in ds.coords and ds["depth"].size == 1:
            ds = ds.drop_vars("depth")
            changed = True

        if dataset_kind == "current":
            regridded = self._regularize_lat_lon_grid(ds)
            if regridded is not ds:
                ds = regridded
                changed = True

        if not changed:
            ds.close()
            return None
        return ds

    @staticmethod
    def _is_uniform_1d(values: np.ndarray) -> bool:
        if values.size < 3:
            return True
        diffs = np.diff(values.astype(float))
        if diffs.size == 0:
            return True
        tolerance = max(1e-8, abs(float(diffs[0])) * 1e-3)
        return bool(np.allclose(diffs, diffs[0], rtol=0.0, atol=tolerance))

    def _regularize_lat_lon_grid(self, dataset: xr.Dataset) -> xr.Dataset:
        lat_name = self._find_coord_name(dataset, ("latitude", "lat"))
        lon_name = self._find_coord_name(dataset, ("longitude", "lon"))
        if lat_name is None or lon_name is None:
            return dataset
        if dataset[lat_name].ndim != 1 or dataset[lon_name].ndim != 1:
            return dataset

        lat_values = np.asarray(dataset[lat_name].values, dtype=float)
        lon_values = np.asarray(dataset[lon_name].values, dtype=float)

        target_lat = lat_values
        target_lon = lon_values
        needs_lat = lat_values.size > 2 and not self._is_uniform_1d(lat_values)
        needs_lon = lon_values.size > 2 and not self._is_uniform_1d(lon_values)
        if needs_lat:
            target_lat = np.linspace(float(lat_values[0]), float(lat_values[-1]), lat_values.size)
        if needs_lon:
            target_lon = np.linspace(float(lon_values[0]), float(lon_values[-1]), lon_values.size)
        if not needs_lat and not needs_lon:
            return dataset

        interp_indexers = {}
        if needs_lat:
            interp_indexers[lat_name] = target_lat
        if needs_lon:
            interp_indexers[lon_name] = target_lon
        return dataset.interp(interp_indexers)

    @staticmethod
    def _find_coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> Optional[str]:
        for candidate in candidates:
            if candidate in dataset.coords or candidate in dataset.variables:
                return candidate
        return None
