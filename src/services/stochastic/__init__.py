"""Stochastic simulation services."""

from src.services.stochastic.parameter_sampler import ParameterSampler
from src.services.stochastic.stochastic_models import (
    SampledParameterSet,
    StochasticGridConfig,
    StochasticParameterConfig,
    StochasticRunConfig,
    StochasticRunResult,
    TemporalLagConfig,
)

__all__ = [
    "EnsembleAggregationResult",
    "EnsembleAggregator",
    "ParameterSampler",
    "SampledParameterSet",
    "StochasticGridConfig",
    "StochasticParameterConfig",
    "StochasticOutputService",
    "StochasticRunConfig",
    "StochasticRunResult",
    "StochasticSimulationService",
    "TemporalLagConfig",
]


def __getattr__(name):
    if name in {"EnsembleAggregationResult", "EnsembleAggregator"}:
        from src.services.stochastic.ensemble_aggregator import (
            EnsembleAggregationResult,
            EnsembleAggregator,
        )

        return {
            "EnsembleAggregationResult": EnsembleAggregationResult,
            "EnsembleAggregator": EnsembleAggregator,
        }[name]
    if name == "StochasticOutputService":
        from src.services.stochastic.stochastic_output_service import StochasticOutputService

        return StochasticOutputService
    if name == "StochasticSimulationService":
        from src.services.stochastic.stochastic_simulation_service import StochasticSimulationService

        return StochasticSimulationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
