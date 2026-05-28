from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.services.conversion.drift_raster_converter import DriftRasterConverter, FixedGrid


@dataclass(frozen=True)
class EnsembleAggregationResult:
    valid_simulations: int
    hit_count: np.ndarray
    probability_map: np.ndarray
    hit_count_map_path: Optional[Path] = None
    probability_map_path: Optional[Path] = None


class EnsembleAggregator:
    """Aggregate deterministic OpenDrift members into probability rasters."""

    def __init__(self, converter: Optional[DriftRasterConverter] = None) -> None:
        self.converter = converter or DriftRasterConverter()

    def aggregate_binary_arrays(self, binary_arrays: Iterable[np.ndarray]) -> EnsembleAggregationResult:
        hit_count: Optional[np.ndarray] = None
        valid_simulations = 0

        for array in binary_arrays:
            binary = (np.asarray(array) > 0).astype(np.uint8)
            if hit_count is None:
                hit_count = np.zeros(binary.shape, dtype=np.uint32)
            if binary.shape != hit_count.shape:
                raise ValueError(
                    f"All binary arrays must share the same shape. "
                    f"Expected {hit_count.shape}, got {binary.shape}."
                )
            hit_count += binary.astype(np.uint32)
            valid_simulations += 1

        if hit_count is None:
            raise ValueError("At least one valid simulation is required for aggregation.")

        probability_map = hit_count.astype(np.float32) / float(valid_simulations)
        return EnsembleAggregationResult(
            valid_simulations=valid_simulations,
            hit_count=hit_count,
            probability_map=probability_map,
        )

    def aggregate_simulation_files(
        self,
        simulation_paths: Iterable[Path],
        grid: FixedGrid,
        time_index: Optional[int] = None,
    ) -> EnsembleAggregationResult:
        arrays = (
            self.converter.convert_simulation_to_binary_array(path, grid, time_index=time_index)
            for path in simulation_paths
        )
        return self.aggregate_binary_arrays(arrays)

    def save_maps(
        self,
        result: EnsembleAggregationResult,
        output_dir: Path,
        grid: FixedGrid,
    ) -> EnsembleAggregationResult:
        output_dir = Path(output_dir)
        hit_count_path = output_dir / "hit_count_map.tif"
        probability_path = output_dir / "probability_map.tif"

        self.converter.save_geotiff(result.hit_count, hit_count_path, grid, dtype=np.uint32)
        self.converter.save_geotiff(result.probability_map, probability_path, grid, dtype=np.float32)

        return EnsembleAggregationResult(
            valid_simulations=result.valid_simulations,
            hit_count=result.hit_count,
            probability_map=result.probability_map,
            hit_count_map_path=hit_count_path,
            probability_map_path=probability_path,
        )
