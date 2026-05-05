from pathlib import Path
from typing import TypedDict, Iterable

from dotenv import find_dotenv, load_dotenv
from loguru import logger
from jsonargparse import auto_cli
from lightning import Trainer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2  # type: ignore

from luad_stress_test.dataset import AbstractPatchDataset, PatchDataset, DHMCDataset
from luad_stress_test.utils import ArtifactType, DatasetName, Label, load_model
from luad_stress_test.prediction_writer import PredictionWriter
from luad_stress_test.path_manager import PathManager


class PredictionDict(TypedDict):
    batch_indices: torch.Tensor
    logits: torch.Tensor
    features: torch.Tensor | None


def predict_on_patches(
    model_name: str,
    dataset_path: Path | Iterable[Path],
    checkpoint_name: str | None = None,
    write_embedings: bool = False,
    dataset_cls: type[AbstractPatchDataset] = PatchDataset,
    collection_name: str | None = None,
) -> None:
    model = load_model(model_name, checkpoint_name)

    if isinstance(dataset_path, Path):
        dataset_path = [dataset_path]

    for path in dataset_path:
        prediction_writer = PredictionWriter(
            PathManager.predictions_dir(
                model_name,
                f"{collection_name}/{path.stem}" if collection_name else path.stem,
            ),
            write_embedings=write_embedings,
        )
        trainer = Trainer(
            precision="16-mixed",
            callbacks=[prediction_writer],
        )
        dataset = dataset_cls(path, transform=v2.CenterCrop(333))
        logger.info(f"Predicting on {len(dataset)} patches in {path.stem}")
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
        trainer.predict(model, dataloader)
        logger.info(f"Prediction finished for {path.stem}")

        postprocess_predictions(
            prediction_writer.output_dir / "predictions.pt", dataset
        )


def postprocess_predictions(
    predictions_path: Path, dataset: AbstractPatchDataset
) -> None:
    predictions: PredictionDict = torch.load(predictions_path)

    logger.info(f"Postprocessing predictions from {predictions_path}")

    patch_names = [
        dataset.patch_name(batch_index) for batch_index in predictions["batch_indices"]
    ]
    patch_labels = torch.tensor(
        [
            dataset.patch_label(batch_index).value
            for batch_index in predictions["batch_indices"]
        ]
    )

    logger.debug(f"Saving predictions to {predictions_path.with_suffix('.csv')}")
    logits = predictions["logits"]
    labels_columns = [Label(i).name for i in range(logits.shape[-1])] + [
        "predicted_label",
        "expected_label",
    ]
    prediction = torch.cat(
        [
            logits.softmax(dim=1),
            logits.argmax(dim=1, keepdim=True),
            patch_labels.unsqueeze(1),
        ],
        dim=1,
    )
    prediction_df = pd.DataFrame(
        prediction.numpy(), columns=labels_columns, index=patch_names
    )
    prediction_df["predicted_label"] = prediction_df["predicted_label"].apply(
        lambda x: Label(x).name
    )
    prediction_df["expected_label"] = prediction_df["expected_label"].apply(
        lambda x: Label(x).name
    )
    prediction_df.to_csv(predictions_path.with_suffix(".csv"))
    logger.debug(f"Predictions saved to {predictions_path.with_suffix('.csv')}")

    if "features" in predictions and predictions["features"] is not None:
        logger.debug(
            f"Saving embedings to {predictions_path.with_name('embedings.parquet')}"
        )
        pd.DataFrame(
            predictions["features"].tolist(),
            columns=[f"feature_{i}" for i in range(predictions["features"].shape[1])],
            index=patch_names,
        ).to_parquet(predictions_path.with_name("embedings.parquet"))
        logger.debug(
            f"Embedings saved to {predictions_path.with_name('embedings.parquet')}"
        )

    logger.debug(f"Removing temporary predictions file {predictions_path}")
    predictions_path.unlink()
    logger.debug(f"Temporary predictions file {predictions_path} removed")
    logger.info(f"Postprocessing finished for {predictions_path}")


def prediction_wrapper(
    model_name: str,
    dataset_name: DatasetName,
    artifact_name: ArtifactType | None = None,
) -> None:
    artifact_dir = artifact_name if artifact_name else "base"
    data_path = PathManager.data_processed(f"{dataset_name}/{artifact_dir}")
    dataset_cls: type[AbstractPatchDataset]
    dataset_path: Path | Iterable[Path]

    if dataset_name == "dhmc":
        dataset_cls = DHMCDataset
        dataset_path = data_path.iterdir()
    else:
        dataset_cls = PatchDataset
        dataset_path = data_path

    predict_on_patches(
        model_name,
        dataset_path=dataset_path,
        write_embedings=dataset_name != "dhmc",
        dataset_cls=dataset_cls,
    )


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    auto_cli(predict_on_patches)
