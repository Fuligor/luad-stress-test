from jsonargparse import auto_cli
from loguru import logger
import pandas as pd

from luad_stress_test.utils import ArtifactType, Label
from luad_stress_test.path_manager import PathManager


def assess_predominant_pattern(
    model_id: str, artifact: ArtifactType | None = None
) -> None:
    results_dir = PathManager.predictions_dir(
        "dhmc",
        artifact,
        model_id,
    )
    results = []

    for slide_path in sorted(results_dir.iterdir()):
        if not slide_path.is_dir():
            continue

        predictions = pd.read_csv(
            PathManager.prediction_smooth_file("dhmc", artifact, model_id), index_col=0
        )
        filtered = predictions[predictions["predicted_label"] != Label.NTU.name]

        if filtered.empty:
            logger.info(
                "{}: No tumor detected, skipping",
                slide_path.stem,
            )

            results.append(
                {
                    "slide": slide_path.stem,
                    "predominant_pattern": "NTU",
                    "expected_pattern": predictions["expected_label"].iloc[0],
                }
            )

            continue

        grouped = filtered.groupby("predicted_label")
        counts = grouped["tissue_ratio"].sum()

        logger.info(
            "{}: Predominant pattern is {} with {:.2f}% of tissue (expected {})",
            slide_path.stem,
            counts.idxmax(),
            counts.max() / counts.sum() * 100,
            predictions["expected_label"].iloc[0],
        )

        results.append(
            {
                "slide": slide_path.stem,
                "predominant_pattern": counts.index[int(counts.argmax())],
                "expected_pattern": predictions["expected_label"].iloc[0],
            }
        )

    pd.DataFrame(results).to_csv(
        PathManager.predominant_pattern_file("dhmc", artifact, model_id), index=False
    )


if __name__ == "__main__":
    auto_cli(assess_predominant_pattern)
