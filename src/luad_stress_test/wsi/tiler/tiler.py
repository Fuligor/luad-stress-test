from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import shutil
import sys
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd

from luad_stress_test.wsi.saturation_tissue_mask import SaturationTissueMask
from luad_stress_test.wsi.tissue_image import WSISlide, WSIPatch
from luad_stress_test.wsi.tiler._wsi_patch_sampler import _WSIPatchSampler
from luad_stress_test.wsi.tiler._tiler_slide_configuration import (
    _WSITilerSlideConfiguration,
)
from luad_stress_test.wsi.utils import Grid, GridLocation, TileManifest


class WSITiler:
    _last_generated_grid: Grid

    def __init__(
        self,
        patch_size: int = 224,
        target_mpp: Optional[float] = None,
        tissue_area: Optional[float] = None,
        tissue_threshold: float = 0.05,
        offset: int | None = None,
        mask: SaturationTissueMask | None = None,
    ):
        self.patch_size = patch_size

        if target_mpp and tissue_area:
            logger.warning(
                "Both target_mpp and tissue_area are provided! tissue_area will be used for patch generation! I could lead to unexpected results if target_mpp is not compatible with tissue_area!"
            )

        if tissue_area:
            target_mpp = tissue_area / self.patch_size

        if target_mpp is None:
            raise ValueError("You have to specify at least target_mpp or tissue_area")

        self.target_mpp: float = target_mpp
        self.mask = mask
        self.tissue_threshold = tissue_threshold * 255
        self.offset = offset if offset else patch_size

    def _get_downsample(self, slide_mpp: float) -> float:
        return self.target_mpp / slide_mpp if self.target_mpp else 1.0

    def save_patch_set(
        self,
        slide: WSISlide,
        output_path: Path,
        template_name: str | None,
        overwrite: bool = False,
    ) -> None:
        if output_path.exists():
            if not overwrite:
                logger.error(
                    "Output path ({}) already exists. Use overwrite=True to overwrite.",
                    output_path,
                )
                sys.exit(1)

            shutil.rmtree(output_path)

        logger.info("Saving patches to {}", output_path)
        output_path.mkdir(parents=True)
        tissue_ratio = []

        slide_metadata = {"slide_name": slide.path.stem}

        if template_name is None:
            template_name = "{row}_{column}"

        logger.info(
            "Generating patches for slide {} with target mpp {}",
            slide.path,
            self.target_mpp,
        )

        config = _WSITilerSlideConfiguration(
            slide, self.patch_size, self.target_mpp, self.offset
        )
        self._last_generated_grid = config.patches_grid

        logger.info("Expected grid shape: {}", self._last_generated_grid)

        generated_samples = 0
        rejected_samples = 0

        for x, y in product(config.y_range, config.x_range):
            patch_sampler = _WSIPatchSampler(slide, x, y, config.patch_size)

            patch = WSIPatch(
                patch_sampler.get().resize((self.patch_size, self.patch_size)),
                patch_sampler.tissue_sample,
                slide.path,
                GridLocation(
                    patch_sampler.pixel_sample.x // config.patch_size,
                    patch_sampler.pixel_sample.y // config.patch_size,
                ),
            )

            patch_name = patch.get_patch_name(template_name)
            if self.mask:
                mean_mask_value = self.mask(patch).astype(np.float32).mean()
                tissue_ratio.append(
                    dict(
                        patch_name=patch_name,
                        tissue_ratio=mean_mask_value / 255.0,
                    )
                )

                if mean_mask_value < self.tissue_threshold:
                    rejected_samples += 1

                    continue

            generated_samples += 1
            patch.pil_image.save((output_path / patch_name).with_suffix(".png"))

        total_samples = generated_samples + rejected_samples
        logger.info("Patches generation done!")
        logger.debug(
            "Generated samples: {} ({:.2f}%)",
            generated_samples,
            generated_samples / total_samples * 100,
        )
        logger.debug(
            "Rejected samples: {} ({:.2f}%)",
            rejected_samples,
            rejected_samples / total_samples * 100,
        )

        manifest: TileManifest = {
            "version": version("ideas-wsi"),
            "slide": {
                "path": str(slide.path),
                "name": slide_metadata["slide_name"],
                "grid": {
                    "rows": self._last_generated_grid.rows,
                    "columns": self._last_generated_grid.columns,
                },
            },
            "patch": {
                "name": template_name,
                "size": self.patch_size,
                "offset": self.offset,
                "mpp": self.target_mpp,
            },
        }

        with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        pd.DataFrame(tissue_ratio).to_csv(output_path / "tissue_ratio.csv", index=False)

        logger.info("Patches saved successfully!")
