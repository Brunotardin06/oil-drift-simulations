from typing import List, Optional

from hydra import compose, initialize_config_dir
from pathlib import Path


class EnvironmentRepository:
    """Load Hydra configs for simulations."""

    def compose_config(
        self,
        config_name: str = "main",
        environment: Optional[str] = None,
        additional_overrides: Optional[List[str]] = None,
    ):
        overrides = list(additional_overrides or [])
        if environment is not None:
            overrides.append(f'environment="{environment}"')

        config_dir = Path(__file__).resolve().parents[2] / "conf"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name=config_name, overrides=overrides)
