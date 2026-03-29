from .config import PipelineConfig
from .models import OntologyLink, SlotRecord, TermCandidate

__all__ = [
    "DownstreamPromptRunner",
    "OntologyLink",
    "PerceptionPipeline",
    "PipelineConfig",
    "SlotRecord",
    "TermCandidate",
]


def __getattr__(name: str):
    if name == "DownstreamPromptRunner":
        from .downstream import DownstreamPromptRunner

        return DownstreamPromptRunner
    if name == "PerceptionPipeline":
        from .pipeline import PerceptionPipeline

        return PerceptionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
