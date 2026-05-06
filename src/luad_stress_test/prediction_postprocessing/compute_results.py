from typing import get_args

from jsonargparse import auto_cli
from loguru import logger
import pandas as pd
from sklearn.metrics import cohen_kappa_score  # type: ignore

from luad_stress_test.utils import ArtifactType, Label
from luad_stress_test.path_manager import PathManager


def compute_results() -> None:
    result = []

    for model_path in sorted(PathManager.model_dir().iterdir()):
        if not model_path.is_dir():
            continue

        model_id = model_path.stem

        logger.info(f"Computing results for model {model_id}")
        base_predictions = pd.read_csv(
            PathManager.predominant_pattern_file("dhmc", None, model_id),
            index_col=0,
        )
        base_correct_predictions = (
            base_predictions["predominant_pattern"]
            == base_predictions["expected_pattern"]
        )
        base_incorrect_predictions = ~base_correct_predictions

        for artifact in (None,) + get_args(ArtifactType):
            try:
                predominant_pattern = pd.read_csv(
                    PathManager.predominant_pattern_file("dhmc", artifact, model_id),
                    index_col=0,
                )

                kappa = cohen_kappa_score(
                    predominant_pattern["predominant_pattern"],
                    predominant_pattern["expected_pattern"],
                    labels=[Label(i).name for i in range(1, len(Label))],
                )

                correct_predictions = (
                    predominant_pattern["predominant_pattern"]
                    == predominant_pattern["expected_pattern"]
                )
                incorrect_predictions = ~correct_predictions

                correct_into_incorrect = (
                    base_correct_predictions & incorrect_predictions
                ).sum()
                incorrect_into_correct = (
                    base_incorrect_predictions & correct_predictions
                ).sum()
                incorrect_into_incorrect = (
                    base_incorrect_predictions
                    & incorrect_predictions
                    & (
                        predominant_pattern["predominant_pattern"]
                        != base_predictions["predominant_pattern"]
                    )
                ).sum()
                unchanged = (
                    base_predictions["predominant_pattern"]
                    == predominant_pattern["predominant_pattern"]
                ).sum()
                correct_unchanged = (
                    (
                        base_predictions["predominant_pattern"]
                        == predominant_pattern["predominant_pattern"]
                    )
                    & correct_predictions
                ).sum()

                result.append(
                    {
                        "model_id": model_id,
                        "artifact": artifact,
                        "kappa": kappa,
                        "correct -> incorrect": correct_into_incorrect,
                        "incorrect -> correct": incorrect_into_correct,
                        "incorrect -> incorrect": incorrect_into_incorrect,
                        "unchanged": unchanged,
                        "correct unchanged": correct_unchanged,
                    }
                )

                logger.info(
                    f"Cohen's kappa for predominant patterns in {model_id} with artifact {artifact}: {kappa:.4f}"
                )
            except FileNotFoundError:
                logger.warning(
                    f"Predominant patterns file not found for {model_id} with artifact {artifact}, skipping kappa calculation"
                )

    pd.DataFrame(result).to_csv(PathManager.results_file(), index=False)


if __name__ == "__main__":
    auto_cli(compute_results)
