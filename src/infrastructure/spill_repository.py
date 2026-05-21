from datetime import datetime
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


class SpillRepository:
    """Read and normalize observed spill data."""

    def find_lat_lon_columns(self, dataframe: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        columns = {column.lower(): column for column in dataframe.columns}
        for lat_key in ("latitude", "lat"):
            for lon_key in ("longitude", "lon"):
                if lat_key in columns and lon_key in columns:
                    return columns[lat_key], columns[lon_key]
        if "y" in columns and "x" in columns:
            return columns["y"], columns["x"]
        return None, None

    def lat_lon_from_group(
        self,
        group: pd.DataFrame,
        lat_col: Optional[str],
        lon_col: Optional[str],
    ) -> Tuple[float, float]:
        if lat_col and lon_col:
            lat_values = pd.to_numeric(group[lat_col], errors="coerce").dropna()
            lon_values = pd.to_numeric(group[lon_col], errors="coerce").dropna()
            if lat_values.empty or lon_values.empty:
                return np.nan, np.nan
            return float(lat_values.mean()), float(lon_values.mean())

        if hasattr(group, "geometry") and not group.geometry.empty:
            geometry = group.geometry
            if (geometry.geom_type == "Point").all():
                return float(geometry.y.mean()), float(geometry.x.mean())
            representative_points = geometry.representative_point()
            return float(representative_points.y.mean()), float(representative_points.x.mean())

        raise ValueError(
            "Latitude/Longitude columns not found (e.g., Latitude/Longitude or lat/lon) "
            "and geometry is missing. Provide lat/lon fields in the shapefile."
        )

    def centroid_lat_lon_from_group(self, group: pd.DataFrame) -> Tuple[float, float]:
        if not hasattr(group, "geometry") or group.geometry.empty:
            raise ValueError("Geometry is missing. Provide polygon geometry in the shapefile.")

        geometry = group.geometry.dropna()
        if geometry.empty:
            return np.nan, np.nan

        crs = getattr(group, "crs", None) or "EPSG:4326"
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)
        if gdf.crs is not None and gdf.crs.is_geographic:
            projected_crs = gdf.estimate_utm_crs()
            if projected_crs is None:
                projected_crs = "EPSG:3857"
            gdf_projected = gdf.to_crs(projected_crs)
        else:
            gdf_projected = gdf

        union_geometry = (
            gdf_projected.geometry.union_all()
            if hasattr(gdf_projected.geometry, "union_all")
            else gdf_projected.geometry.unary_union
        )
        centroid = union_geometry.centroid
        centroid_wgs84 = gpd.GeoSeries([centroid], crs=gdf_projected.crs).to_crs(epsg=4326).iloc[0]
        return float(centroid_wgs84.y), float(centroid_wgs84.x)

    def ensure_datetime_column(self, manchas: pd.DataFrame, offset_hours: float = 0.0) -> pd.DataFrame:
        if "datetime" in manchas.columns:
            return manchas

        columns = set(manchas.columns)
        if "DATA_HORA1" in columns and "TEMPO_ENTR" in columns:
            manchas["date"] = pd.to_datetime(manchas["DATA_HORA1"], format="%d/%m/%Y")
            manchas["time"] = pd.to_datetime(manchas["TEMPO_ENTR"], format="%H:%M")
            manchas["datetime"] = manchas.apply(
                lambda row: datetime.combine(row["date"].date(), row["time"].time()),
                axis=1,
            )
        elif "Data/Hora" in columns:
            manchas["datetime"] = pd.to_datetime(
                manchas["Data/Hora"],
                dayfirst=True,
                errors="raise",
            )
        else:
            raise ValueError(
                "Missing datetime fields. Expected DATA_HORA1/TEMPO_ENTR or Data/Hora."
            )

        if offset_hours:
            manchas["datetime"] = manchas["datetime"] + pd.Timedelta(hours=float(offset_hours))
        return manchas

    def build_observed_trajectory(self, manchas: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in manchas.columns:
            raise ValueError("Expected 'datetime' column in manchas")

        grouped = manchas.groupby("datetime", sort=True)
        rows = []
        for dt, group in grouped:
            lat, lon = self.centroid_lat_lon_from_group(group)
            rows.append({"time": pd.to_datetime(dt), "lon": lon, "lat": lat})
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
