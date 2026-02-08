import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class WorkspaceRepository:
    """Store and retrieve run artifacts."""

    def simulation_output_dir(self, config) -> Path:
        output_dir = Path(config.paths.simulation_data) / config.simulation.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def write_json(self, destination: Path, payload: Dict[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_csv(self, destination: Path, dataframe: pd.DataFrame) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(destination, index=False)

