import json
from pathlib import Path

from luad_stress_test.wsi.tissue_image.utils import parser_template
from luad_stress_test.wsi.utils import TileManifest
from jsonargparse import auto_cli
from loguru import logger
import numpy as np
import pandas as pd
from scipy.signal import convolve2d

from luad_stress_test.utils import Label
from luad_stress_test.path_manager import PathManager


def smooth_predictions(
    model_id: str,
    artifact: str | None = None,
    sample_ratio: float | None = None,
    weighted: bool = False,
):
    patches_dir = PathManager.data_processed("dhmc/patches")
    base_results_dir = PathManager.predictions_dir(model_id, "DHMC-base")

    if artifact is not None:
        artifact_results_dir = PathManager.predictions_dir(model_id, f"DHMC-{artifact}")

    patterns = [Label(i).name for i in range(len(Label))]

    for slide_path in sorted(patches_dir.iterdir()):
        if not slide_path.is_dir():
            continue

        logger.info(f"Smoothing predictions for {slide_path.stem}")

        with open(slide_path / "manifest.json", encoding="utf-8") as f:
            patch_manifest: TileManifest = json.load(f)

        prediction_array = np.zeros(
            (
                patch_manifest["slide"]["grid"]["rows"],
                patch_manifest["slide"]["grid"]["columns"],
                len(Label),
            ),
            dtype=np.float32,
        )
        # prediction_array[..., Label.NTU.value] = 1.0

        base_slide_prediction_dir = base_results_dir / slide_path.stem
        predictions = pd.read_csv(
            base_slide_prediction_dir / "predictions.csv", index_col=0
        )

        if artifact is not None:
            artifact_slide_prediction_dir = artifact_results_dir / slide_path.stem
            artifact_predictions = pd.read_csv(
                artifact_slide_prediction_dir / "predictions.csv", index_col=0
            )

            if sample_ratio is not None:
                artifact_predictions = artifact_predictions.sample(
                    frac=sample_ratio, random_state=42
                )

            for index, row in artifact_predictions.iterrows():
                predictions.loc[index] = row

        if weighted:
            tissue_ratios = pd.read_csv(slide_path / "tissue_ratio.csv", index_col=0)
            predictions = predictions.merge(
                tissue_ratios, left_index=True, right_index=True, how="left"
            )

        for index, row in predictions.iterrows():
            patch_name = Path(str(index)).stem
            patch_metadata = parser_template(
                patch_name, patch_manifest["patch"]["name"]
            )

            prediction_array[
                int(patch_metadata["row"]),
                int(patch_metadata["column"]),
            ] = row[patterns].to_numpy() * (row.tissue_ratio if weighted else 1)

        kernel = np.ones((3, 3))
        smooth = [
            convolve2d(prediction_array[..., layer], kernel, mode="same")
            for layer in range(prediction_array.shape[-1])
        ]
        smooth_array = np.array(smooth)

        results = predictions.copy()

        for index in predictions.index:
            patch_name = Path(str(index)).stem
            patch_metadata = parser_template(
                patch_name, patch_manifest["patch"]["name"]
            )
            smooth_preds = smooth_array[
                :, int(patch_metadata["row"]), int(patch_metadata["column"])
            ]

            results.loc[index, patterns] = smooth_preds / smooth_preds.sum()
            results.loc[index, "predicted_label"] = Label(
                int(np.argmax(smooth_preds))
            ).name

        output_dir = (
            base_results_dir if artifact is None else artifact_results_dir
        ) / slide_path.stem
        output_name = (
            f"predictions_smooth_{artifact}_{sample_ratio}.csv"
            if sample_ratio and artifact
            else "predictions_smooth.csv"
        )
        results.to_csv(output_dir / output_name)


if __name__ == "__main__":
    auto_cli(smooth_predictions)
