from itertools import product
from pathlib import Path
from typing import Literal, get_args

from jsonargparse import auto_cli
import pandas as pd

from luad_stress_test.path_manager import PathManager
from luad_stress_test.predict.predict import predict_on_patches
from luad_stress_test.prediction_postprocessing.compute_results import compute_results
from luad_stress_test.prediction_postprocessing.assess_predominant_pattern import (
    assess_predominant_pattern,
)
from luad_stress_test.prediction_postprocessing.smooth_predictions import (
    smooth_predictions,
)
from luad_stress_test.preprocessing.anorak.filter import write_to_patch
from luad_stress_test.preprocessing.luad_c.patch_splitting import save_patches_from_dir
from luad_stress_test.preprocessing.wsi.tiler import WSITiler
from luad_stress_test.preprocessing.wsi.tissue_image import WSISlide
from luad_stress_test.preprocessing.wsi.saturation_tissue_mask import (
    SaturationTissueMask,
)
from luad_stress_test.utils import ArtifactType, DatasetName


def generate_anorak_base():
    image_path = PathManager.anorak_raw() / "image"
    label_path = PathManager.anorak_raw() / "maskPng"
    save_path = PathManager.anorak_processed() / "patches"

    train_data = sorted(list(image_path.glob("train*")))
    print("data number: ", len(train_data))
    print(train_data)

    write_to_patch(data_files=train_data, label_path=label_path, save_path=save_path)


def generate_luad_c_base():
    img_size = 384

    save_patches_from_dir(
        PathManager.luad_c_images(),
        PathManager.luad_c_annotations(),
        PathManager.luad_c_processed(),
        patch_size=(img_size, img_size),
        overlap_pct=0.25,
        target_mpp=0.9,
        patch_pattern_area_criterion=0.6,
    )


def generate_dhmc_base():
    dhmc_path = PathManager.dhmc_raw()
    tiles_path = PathManager.dhmc_processed()

    metadata = pd.read_csv(PathManager.dhmc_metadata())
    mask = SaturationTissueMask(7.2, saturation_threshold=20)
    tiler = WSITiler(mask=mask, target_mpp=0.9, patch_size=384, offset=333)

    for _, row in metadata.iterrows():
        file_name = row["File Name"]
        mpp = row["Microns Per Pixel"]

        slide = WSISlide(dhmc_path / "wsi" / file_name, slide_mpp=mpp)
        tiler.save_patch_set(slide, tiles_path / Path(file_name).stem, "{row}-{column}")


def generate_dataset(
    dataset_name: DatasetName | Literal["all"] = "all",
):
    match dataset_name:
        case "anorak" | "all":
            generate_anorak_base()
        case "luad_c" | "all":
            generate_luad_c_base()
        case "dhmc" | "all":
            generate_dhmc_base()


def predict_on_artifact(
    model_name: str | Literal["all"] = "all",
    dataset_name: DatasetName | Literal["all"] = "all",
    artifact_name: ArtifactType | Literal["all"] | None = "all",
    write_embedings: bool | None = None,
) -> None:
    models = (
        [model_name]
        if model_name != "all"
        else [p.stem for p in PathManager.model_dir().iterdir()]
    )
    datasets = [dataset_name] if dataset_name != "all" else get_args(DatasetName)
    artifacts = (
        [artifact_name] if artifact_name != "all" else (None,) + get_args(ArtifactType)
    )

    for model, dataset, artifact in product(models, datasets, artifacts):
        predict_on_patches(
            model_name=model,
            dataset_name=dataset,
            artifact_name=artifact,
            write_embedings=write_embedings,
        )


def postprocess_predictions(
    model_name: str | Literal["all"] = "all",
    artifact_name: ArtifactType | None | Literal["all"] = "all",
):
    models = (
        [model_name]
        if model_name != "all"
        else [p.stem for p in PathManager.model_dir().iterdir()]
    )
    artifacts = (
        [artifact_name] if artifact_name != "all" else (None,) + get_args(ArtifactType)
    )

    for model, artifact in product(models, artifacts):
        smooth_predictions(
            model_id=model,
            artifact=artifact,
            weighted=True,
        )
        assess_predominant_pattern(
            model_id=model,
            artifact=artifact,
        )
    compute_results()


def pipeline(
    step: Literal["prepare", "predict", "postprocess", "full"] = "full",
    model_name: str | Literal["all"] = "all",
    dataset_name: DatasetName | Literal["all"] = "all",
    artifact_name: ArtifactType | None | Literal["all"] = "all",
    write_embedings: bool | None = None,
):
    if step in ("prepare", "full"):
        generate_dataset(dataset_name=dataset_name)
    if step in ("predict", "full"):
        predict_on_artifact(
            model_name=model_name,
            dataset_name=dataset_name,
            artifact_name=artifact_name,
            write_embedings=write_embedings,
        )
    if step in ("postprocess", "full") and dataset_name in ("dhmc", "all"):
        postprocess_predictions(model_name=model_name, artifact_name=artifact_name)


def cli():
    auto_cli(pipeline)
