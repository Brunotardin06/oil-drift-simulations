from src.application.simulation_controller import SimulationController


class CliAdapter:
    """Thin adapter for CLI entrypoints."""

    def __init__(self, controller=None):
        self.controller = controller or SimulationController()

