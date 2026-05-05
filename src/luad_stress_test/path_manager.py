"""Centralized path management for the LUAD project.

Provides static methods to access all project paths (data, experiments, predictions)
with support for environment-based configuration and automatic directory creation.
"""

from pathlib import Path
import os
from typing import Optional


class PathManager:
    """Centralized path manager for accessing all project paths."""

    _PROJECT_ROOT: Optional[Path] = None
    _DATA_ROOT: Optional[Path] = None

    @classmethod
    def _load_config(cls) -> None:
        """Load configuration from environment variables or defaults."""
        if cls._PROJECT_ROOT is not None:
            return

        # Find project root from this file's location (path_manager.py is in src/luad/)
        cls._PROJECT_ROOT = Path(__file__).parent.parent.parent

        # Load custom data root if specified
        data_root_env = os.getenv("LUAD_DATA_ROOT")
        if data_root_env:
            cls._DATA_ROOT = Path(data_root_env)
        else:
            cls._DATA_ROOT = cls._PROJECT_ROOT

    @classmethod
    def project_root(cls) -> Path:
        """Return the project root path."""
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        return cls._PROJECT_ROOT

    @classmethod
    def data_root(cls) -> Path:
        """Return the data root path (can be overridden by LUAD_DATA_ROOT env var)."""
        cls._load_config()
        assert cls._DATA_ROOT is not None
        return cls._DATA_ROOT

    @classmethod
    def data_raw(cls, subdir: Optional[str] = None) -> Path:
        """Return path to raw data directory.

        Args:
            subdir: Optional subdirectory within raw data (e.g., "dhmc")

        Returns:
            Path object pointing to data/raw or data/raw/{subdir}
        """
        cls._load_config()
        assert cls._DATA_ROOT is not None
        path = cls._DATA_ROOT / "data" / "raw"
        if subdir:
            path = path / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def data_processed(cls, subdir: Optional[str] = None) -> Path:
        """Return path to processed data directory.

        Args:
            subdir: Optional subdirectory within processed data (e.g., "dhmc/patches")

        Returns:
            Path object pointing to data/processed or data/processed/{subdir}
        """
        cls._load_config()
        assert cls._DATA_ROOT is not None
        path = cls._DATA_ROOT / "data" / "processed"
        if subdir:
            path = path / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def dhmc_metadata(cls) -> Path:
        """Return path to DHMC metadata CSV file."""
        cls._load_config()
        assert cls._DATA_ROOT is not None
        metadata_path = cls._DATA_ROOT / "data" / "raw" / "dhmc" / "metadata.csv"
        return metadata_path

    @classmethod
    def experiment_dir(cls, model_name: str) -> Path:
        """Return path to experiment directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "resnet", "vit")

        Returns:
            Path object pointing to experiments/luad/{model_name}
        """
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls._PROJECT_ROOT / "experiments" / "luad" / model_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def experiment_checkpoint(cls, model_name: str) -> Path:
        """Return path to checkpoint directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "resnet", "vit")

        Returns:
            Path object pointing to experiments/luad/{model_name}/checkpoints
        """
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls._PROJECT_ROOT / "experiments" / "luad" / model_name / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def predictions_dir(
        cls, model_name: Optional[str] = None, subdir: Optional[str] = None
    ) -> Path:
        """Return path to predictions directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "resnet", "vit")
            subdir: Optional subdirectory (e.g., dataset name)

        Returns:
            Path object pointing to predictions/{model_name} or predictions/{model_name}/{subdir}
        """
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls._PROJECT_ROOT / "predictions"
        if model_name:
            path = path / model_name
        if subdir:
            path = path / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
