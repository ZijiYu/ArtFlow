"""Dynamic agent pipeline for Chinese painting appreciation."""

from .meta_loader import load_context_meta, merge_meta
from .models import PipelineConfig, PipelineResult
from .pipeline import DynamicAgentPipeline

__all__ = [
    "DynamicAgentPipeline",
    "PipelineConfig",
    "PipelineResult",
    "load_context_meta",
    "merge_meta",
    "ClosedLoopConfig",
    "ClosedLoopCoordinator",
    "ClosedLoopResult",
]


def __getattr__(name: str):
    if name in {"ClosedLoopConfig", "ClosedLoopCoordinator", "ClosedLoopResult"}:
        from .closed_loop import ClosedLoopConfig, ClosedLoopCoordinator, ClosedLoopResult

        mapping = {
            "ClosedLoopConfig": ClosedLoopConfig,
            "ClosedLoopCoordinator": ClosedLoopCoordinator,
            "ClosedLoopResult": ClosedLoopResult,
        }
        return mapping[name]
    raise AttributeError(name)
