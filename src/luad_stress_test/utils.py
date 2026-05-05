from enum import IntEnum
from pathlib import Path
from typing import Literal

from loguru import logger

from luad_stress_test.models.classification import ClassificationModel
from luad_stress_test.path_manager import PathManager

ArtifactType = Literal["blur", "dust", "fold", "marker", "prec", "stitch"]
DatasetName = Literal["dhmc", "anorak", "tcga"]


class Label(IntEnum):
    NTU = 0
    ACC = 1
    CRB = 2
    LEP = 3
    MIP = 4
    PAP = 5
    SOL = 6


def load_model(
    model_name: str,
    checkpoint_name: str | None = None,
) -> ClassificationModel:
    checkpoint_dir = PathManager.experiment_checkpoint(model_name)
    checkpoint_path: Path

    if checkpoint_name is not None:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_path} not found")
    else:
        checkpoint_paths = list(checkpoint_dir.glob("*.ckpt"))
        if len(checkpoint_paths) == 0:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        if len(checkpoint_paths) > 1:
            raise ValueError(
                f"Multiple checkpoints found in {checkpoint_dir}. "
                f"Please specify the checkpoint to use."
            )

        checkpoint_path = checkpoint_paths[0]

    logger.info(f"Loading model from checkpoint {checkpoint_path}")

    return ClassificationModel.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=checkpoint_path
    )
