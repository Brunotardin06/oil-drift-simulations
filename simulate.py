import click

from src.application.dto import ValidationRunRequest
from src.application.simulation_controller import SimulationController


@click.command()
@click.option("--config-name", default="main")
@click.option("--skip-animation", is_flag=True, default=False)
@click.option("--skip-simulation", is_flag=False, default=False)
@click.option("--skip-plots", is_flag=True, default=False)
@click.option("--evaluation", is_flag=True, default=False)
@click.option("--optimize-wdf", is_flag=True, default=False)
@click.option("--optimize-stokes", is_flag=True, default=False)
@click.option("--optimize-wdf-stokes", is_flag=True, default=False)
@click.option("--optimize-wdf-stokes-cdf", is_flag=True, default=False)
@click.option("--optimize-physics", is_flag=True, default=False)
@click.option(
    "--optimize-wdf-mode",
    type=click.Choice(["robust", "fast"], case_sensitive=False),
    default="fast",
)
@click.option("--fast-particles-per-wdf", type=int, default=1)
@click.option("--wdf-min", type=float, default=0.0)
@click.option("--wdf-max", type=float, default=0.05)
@click.option("--wdf-step", type=float, default=0.0025)
@click.option("--cdf-min", type=float, default=0.5)
@click.option("--cdf-max", type=float, default=1.0)
@click.option("--cdf-step", type=float, default=0.1)
@click.option("--diffusivity-values", type=str, default=None)
@click.option("--dispersion-values", type=str, default=None)
@click.option("--evaporation-values", type=str, default=None)
@click.option("--optimize-cleanup", is_flag=True, default=False)
@click.option("--padding-animation-frame", type=float, default=0.1)
@click.option("--wind-drift-factor", type=float, default=None)
@click.option("--current-drift-factor", type=float, default=None)
@click.option(
    "--stokes-drift",
    type=click.Choice(["true", "false"], case_sensitive=False),
    default=None,
)
@click.option("--horizontal-diffusivity", type=float, default=None)
@click.option(
    "--processes-dispersion",
    type=click.Choice(["true", "false"], case_sensitive=False),
    default=None,
)
@click.option(
    "--processes-evaporation",
    type=click.Choice(["true", "false"], case_sensitive=False),
    default=None,
)
@click.option("--oil-types", type=str, default=None)
@click.option("--oil-types-file", type=str, default=None)
@click.option("--environment", type=str, default="2019")
@click.option("--shp-zip", type=str, default=None)
@click.option("--min-long", type=float, default=None)
@click.option("--max-long", type=float, default=None)
@click.option("--min-lat", type=float, default=None)
@click.option("--max-lat", type=float, default=None)
@click.option("--start-index", type=int, default=0)
@click.option("--optimize-cdf-hd-de", is_flag=True, default=False)
def simulate_validation(**kwargs):
    controller = SimulationController()
    request = ValidationRunRequest(**kwargs)
    controller.run_validation(request)


if __name__ == "__main__":
    simulate_validation()

