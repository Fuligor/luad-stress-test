from jsonargparse import auto_cli
from loguru import logger
import pandas as pd
from sklearn.metrics import cohen_kappa_score  # type: ignore

from luad_stress_test.utils import Label
from luad_stress_test.path_manager import PathManager


def compute_results() -> None:
    predictions_dir = PathManager.predictions_dir()
    result = []

    for model_path in sorted(predictions_dir.iterdir()):
        if not model_path.is_dir():
            continue

        model_id = model_path.stem

        if model_id == "virchow2-base":
            continue

        logger.info(f"Computing results for model {model_id}")
        base_predictions = pd.read_csv(
            predictions_dir / f"{model_id}/DHMC-base" / "predominant_patterns_0.0.csv",
            index_col=0,
        )
        base_correct_predictions = (
            base_predictions["predominant_pattern"]
            == base_predictions["expected_pattern"]
        )
        base_incorrect_predictions = ~base_correct_predictions

        if model_id == "virchow2-rev1":
            model_id = "virchow2-base"

        for artifact_dir in sorted(model_path.glob("DHMC-*")):
            artifact = artifact_dir.stem.split("-")[1]

            for predominant_pattern_path in sorted(
                artifact_dir.glob("predominant_patterns_*.csv")
            ):
                sample_ratio_str = predominant_pattern_path.stem.split("_")[-1]
                sample_ratio = float(sample_ratio_str)

                try:
                    predominant_pattern = pd.read_csv(
                        predominant_pattern_path, index_col=0
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

                    logger.info(
                        f"Cohen's kappa for predominant patterns in {model_id} with artifact {artifact} and sample ratio {sample_ratio}: {kappa:.4f}"
                    )
                except FileNotFoundError:
                    logger.warning(
                        f"Predominant patterns file not found for {model_id} with artifact {artifact} and sample ratio {sample_ratio}, skipping kappa calculation"
                    )

                result.append(
                    {
                        "model_id": model_id,
                        "artifact": artifact,
                        "sample_ratio": sample_ratio if artifact != "base" else None,
                        "kappa": kappa,
                        "correct -> incorrect": correct_into_incorrect,
                        "incorrect -> correct": incorrect_into_correct,
                        "incorrect -> incorrect": incorrect_into_incorrect,
                        "unchanged": unchanged,
                        "correct unchanged": correct_unchanged,
                    }
                )

    pd.DataFrame(result).to_csv(predictions_dir / "results.csv", index=False)

    # with open(predictions_dir / "dhmc.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=4)


if __name__ == "__main__":
    auto_cli(compute_results)
