"""setfit-cefr: SetFit-based CEFR classification with reproducible config-first CLI."""
from setfit_cefr.config import (
    Config,
    ConfigValidationError,
    DataConfig,
    ModelConfig,
    ReportingConfig,
    RuntimeConfig,
    TrainingConfig,
    set_file_existence_checks,
)
from setfit_cefr.notebook_config import (
    InputConfig,
    NotebookConfig,
    OutputConfig,
    RunConfig,
)

__all__ = [
    "Config",
    "ConfigValidationError",
    "DataConfig",
    "InputConfig",
    "ModelConfig",
    "NotebookConfig",
    "OutputConfig",
    "ReportingConfig",
    "RunConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "set_file_existence_checks",
]

__version__ = "0.1.0"
