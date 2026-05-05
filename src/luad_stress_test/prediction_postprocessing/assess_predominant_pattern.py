from jsonargparse import auto_cli
from loguru import logger
import pandas as pd

from luad_stress_test.utils import Label
from luad_stress_test.path_manager import PathManager


def assess_predominant_pattern(
    model_id: str, artifact: str | None = None, sample_ratio: float | None = None
) -> None:
    base_results_dir = PathManager.predictions_dir(
        model_id, "DHMC-base" if artifact is None else f"DHMC-{artifact}"
    )
    results = []

    for slide_path in sorted(base_results_dir.iterdir()):
        if not slide_path.is_dir():
            continue

        output_name = (
            f"predictions_smooth_{artifact}_{sample_ratio}.csv"
            if sample_ratio and artifact
            else "predictions_smooth.csv"
        )

        predictions = pd.read_csv(slide_path / output_name, index_col=0)
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
        base_results_dir / f"predominant_patterns_{sample_ratio}.csv", index=False
    )


if __name__ == "__main__":
    auto_cli(assess_predominant_pattern)
