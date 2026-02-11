from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

import copernicusmarine as cm


class CopernicusGateway:
    """Download Copernicus datasets defined by environment config."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        login_file: Optional[Path] = None,
    ) -> None:
        self._username = username
        self._password = password
        self._login_file = Path(login_file) if login_file else None

    def _resolve_credentials(self) -> tuple[str, str]:
        if self._username and self._password:
            return self._username, self._password

        env_user = os.getenv("COPERNICUS_USERNAME")
        env_pwd = os.getenv("COPERNICUS_PASSWORD")
        if env_user and env_pwd:
            return env_user, env_pwd

        user_names = [
            value
            for value in [os.getenv("USERNAME"), os.getenv("username"), Path.home().name]
            if value
        ]
        user_names = list(dict.fromkeys(user_names))

        env_loginfile = os.getenv("COPERNICUS_LOGIN_FILE")
        login_candidates: list[Path] = []
        if self._login_file:
            login_candidates.append(self._login_file)
        if env_loginfile:
            login_candidates.append(Path(env_loginfile))
        project_root = Path(__file__).resolve().parents[2]
        for user_name in user_names:
            login_candidates.append(project_root / f"copernicus_login_{user_name}.json")
            login_candidates.append(Path.home() / f"copernicus_login_{user_name}.json")
        login_candidates.append(project_root / "copernicus_login.json")
        login_candidates.append(Path.home() / "copernicus_login.json")
        login_candidates = list(dict.fromkeys(login_candidates))

        login_path = next((path for path in login_candidates if path.exists()), None)
        if login_path is None:
            searched = "\n".join(f"- {path}" for path in login_candidates)
            raise FileNotFoundError(
                "Copernicus login file not found. Searched:\n" + searched
            )

        payload = json.loads(login_path.read_text(encoding="utf-8"))
        username = payload.get("user")
        password = payload.get("pwd")
        if not username or not password:
            raise ValueError(f"Invalid Copernicus login file: {login_path}")
        return username, password

    @staticmethod
    def _dataset_specs(config) -> list[dict]:
        specifics = config.copernicusmarine.specificities
        return [
            {
                "name": "water",
                "dataset_id": specifics.water_dataset_id,
                "dataset_path": Path(specifics.water_dataset_path),
                "variables": ["uo", "vo"],
            },
            {
                "name": "wind",
                "dataset_id": specifics.wind_dataset_id,
                "dataset_path": Path(specifics.wind_dataset_path),
                "variables": ["eastward_wind", "northward_wind"],
            },
            {
                "name": "wave",
                "dataset_id": specifics.wave_dataset_id,
                "dataset_path": Path(specifics.wave_dataset_path),
                "variables": ["VSDX", "VSDY", "VHM0", "VTM02", "VTPK"],
            },
            {
                "name": "sal_temp",
                "dataset_id": specifics.sal_temp_dataset_id,
                "dataset_path": Path(specifics.sal_temp_dataset_path),
                "variables": ["so", "thetao"],
            },
        ]

    def download_environment_data(
        self,
        config,
        force: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> dict:
        def log(message: str) -> None:
            if log_callback is not None:
                log_callback(message)

        if (username and not password) or (password and not username):
            raise ValueError("Provide both Copernicus username and password.")
        if not username and not password:
            username, password = self._resolve_credentials()
        log("Trying Copernicus login...")
        if not cm.login(username=username, password=password, force_overwrite=True):
            raise RuntimeError("Copernicus login failed.")
        log("Copernicus login successful.")

        outcomes: list[dict] = []
        specs = self._dataset_specs(config)
        total = len(specs)
        min_lon = float(config.copernicusmarine.min_long)
        max_lon = float(config.copernicusmarine.max_long)
        min_lat = float(config.copernicusmarine.min_lat)
        max_lat = float(config.copernicusmarine.max_lat)
        start_dt = str(config.copernicusmarine.specificities.start_datetime)
        end_dt = str(config.copernicusmarine.specificities.end_datetime)
        for index, spec in enumerate(specs, start=1):
            dataset_path: Path = spec["dataset_path"]
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            if dataset_path.exists() and not force:
                log(f"{index}/{total} {spec['name']}: file already exists, skipping.")
                outcomes.append(
                    {
                        "name": spec["name"],
                        "path": str(dataset_path),
                        "status": "skipped",
                    }
                )
                continue

            log(f"{index}/{total} {spec['name']}: downloading...")
            log(
                f"{index}/{total} {spec['name']}: "
                f"dataset_id={spec['dataset_id']}, "
                f"lon=[{min_lon},{max_lon}], lat=[{min_lat},{max_lat}], "
                f"time=[{start_dt},{end_dt}]"
            )
            try:
                cm.subset(
                    dataset_id=spec["dataset_id"],
                    minimum_longitude=min_lon,
                    maximum_longitude=max_lon,
                    minimum_latitude=min_lat,
                    maximum_latitude=max_lat,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    output_directory=dataset_path.parent,
                    output_filename=dataset_path.name,
                    variables=spec["variables"],
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "overlap" in error_text and min_lon < 0 and max_lon < 0:
                    wrap_min_lon = min_lon + 360.0
                    wrap_max_lon = max_lon + 360.0
                    log(
                        f"{index}/{total} {spec['name']}: retrying with wrapped longitudes "
                        f"[{wrap_min_lon},{wrap_max_lon}]"
                    )
                    cm.subset(
                        dataset_id=spec["dataset_id"],
                        minimum_longitude=wrap_min_lon,
                        maximum_longitude=wrap_max_lon,
                        minimum_latitude=min_lat,
                        maximum_latitude=max_lat,
                        start_datetime=start_dt,
                        end_datetime=end_dt,
                        output_directory=dataset_path.parent,
                        output_filename=dataset_path.name,
                        variables=spec["variables"],
                    )
                else:
                    raise RuntimeError(
                        f"Failed subset for dataset '{spec['name']}' "
                        f"(id={spec['dataset_id']})."
                    ) from exc
            log(f"{index}/{total} {spec['name']}: done.")
            outcomes.append(
                {
                    "name": spec["name"],
                    "path": str(dataset_path),
                    "status": "downloaded",
                }
            )

        return {"datasets": outcomes, "force": force}
