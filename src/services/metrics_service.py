import numpy as np
import pandas as pd


class MetricsService:
    """Skill score and distance metrics used by optimization."""

    @staticmethod
    def haversine_m(lon1, lat1, lon2, lat2):
        radius_m = 6371000.0
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        a_value = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        )
        c_value = 2 * np.arctan2(np.sqrt(a_value), np.sqrt(1 - a_value))
        return radius_m * c_value

    def liu_weissberg_skillscore(self, observed_df: pd.DataFrame, modeled_df: pd.DataFrame) -> float:
        merged = observed_df.merge(
            modeled_df,
            on="time",
            suffixes=("_obs", "_mod"),
        ).sort_values("time")
        merged = merged.dropna(subset=["lon_obs", "lat_obs", "lon_mod", "lat_mod"])
        if len(merged) < 2:
            return np.nan

        obs_lon = merged["lon_obs"].to_numpy()
        obs_lat = merged["lat_obs"].to_numpy()
        mod_lon = merged["lon_mod"].to_numpy()
        mod_lat = merged["lat_mod"].to_numpy()

        separation = self.haversine_m(obs_lon, obs_lat, mod_lon, mod_lat)
        path_length = self.haversine_m(obs_lon[1:], obs_lat[1:], obs_lon[:-1], obs_lat[:-1])
        denominator = np.nansum(path_length)
        if denominator <= 0:
            return np.nan
        return 1.0 - (np.nansum(separation[1:]) / denominator)

