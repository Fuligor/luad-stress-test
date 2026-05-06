"""Centralized path management for the LUAD project.

Provides static methods to access all project paths (data, experiments, predictions)
with support for environment-based configuration and automatic directory creation.
"""

from pathlib import Path
import os
from typing import Optional

from luad_stress_test.utils import ArtifactType, DatasetName


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
    def __artifact_to_name(cls, artifact: ArtifactType | None) -> str:
        """Convert artifact type to directory name."""
        return artifact if artifact else "base"

    @classmethod
    def anorak_raw(cls) -> Path:
        """Return path to Anorak raw data directory."""
        return cls.data_root() / "data" / "raw" / "anorak"

    @classmethod
    def anorak_processed(cls, artifact_name: ArtifactType | None = None) -> Path:
        """Return path to Anorak processed data directory."""
        path = (
            cls.data_root()
            / "data"
            / "processed"
            / "anorak"
            / cls.__artifact_to_name(artifact_name)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def luad_c_raw(cls) -> Path:
        """Return path to LUAD-C raw data directory."""
        return cls.data_root() / "data" / "raw" / "tcga"

    @classmethod
    def luad_c_images(cls) -> Path:
        """Return path to LUAD-C raw images directory."""
        return cls.luad_c_raw() / "images"

    @classmethod
    def luad_c_annotations(cls) -> Path:
        """Return path to LUAD-C raw annotations directory."""
        return cls.luad_c_raw() / "annotations"

    @classmethod
    def luad_c_processed(cls, artifact_name: ArtifactType | None = None) -> Path:
        """Return path to LUAD-C processed data directory."""
        path = (
            cls.data_root()
            / "data"
            / "processed"
            / "luad-c"
            / cls.__artifact_to_name(artifact_name)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def dhmc_raw(cls) -> Path:
        """Return path to DHMC raw data directory."""
        return cls.data_root() / "data" / "raw" / "dhmc"

    @classmethod
    def dhmc_processed(cls, artifact_name: ArtifactType | None = None) -> Path:
        """Return path to DHMC processed data directory."""
        path = (
            cls.data_root()
            / "data"
            / "processed"
            / "dhmc"
            / cls.__artifact_to_name(artifact_name)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def dhmc_metadata(cls) -> Path:
        """Return path to DHMC metadata CSV file."""
        cls._load_config()
        assert cls._DATA_ROOT is not None
        metadata_path = cls.dhmc_raw() / "metadata.csv"
        return metadata_path

    @classmethod
    def processed_path(
        cls, dataset_name: DatasetName, artifact_name: ArtifactType | None = None
    ) -> Path:
        """Return path to processed data directory for a given dataset and artifact."""
        match dataset_name:
            case "anorak":
                return cls.anorak_processed(artifact_name)
            case "luad-c":
                return cls.luad_c_processed(artifact_name)
            case "dhmc":
                return cls.dhmc_processed(artifact_name)
            case _:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

    @classmethod
    def model_dir(cls) -> Path:
        """Return path to model directory."""
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls._PROJECT_ROOT / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def model_checkpoint(cls, model_name: str) -> Path:
        """Return path to checkpoint directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "resnet", "vit")

        Returns:
            Path object pointing to experiments/luad/{model_name}/checkpoints
        """
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls.model_dir() / model_name / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def results_dir(cls) -> Path:
        """Return path to results directory for a specific model."""
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls._PROJECT_ROOT / "predictions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def results_file(cls) -> Path:
        """Return path to results CSV file."""
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = cls.results_dir() / "results.csv"
        return path

    @classmethod
    def predictions_dir(
        cls,
        dataset_name: DatasetName,
        artifact_name: ArtifactType | None,
        model_name: str,
    ) -> Path:
        cls._load_config()
        assert cls._PROJECT_ROOT is not None
        path = (
            cls.results_dir()
            / model_name
            / dataset_name
            / cls.__artifact_to_name(artifact_name)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def prediction_smooth_file(
        cls,
        dataset_name: DatasetName,
        artifact_name: ArtifactType | None,
        model_name: str,
    ) -> Path:
        return (
            cls.predictions_dir(dataset_name, artifact_name, model_name)
            / "predictions_smooth.csv"
        )

    @classmethod
    def predominant_pattern_file(
        cls,
        dataset_name: DatasetName,
        artifact_name: ArtifactType | None,
        model_name: str,
    ) -> Path:
        return (
            cls.predictions_dir(dataset_name, artifact_name, model_name)
            / "predominant_patterns.csv"
        )
