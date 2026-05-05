from math import ceil
from pathlib import Path

import openslide
from openslide import open_slide
from PIL.Image import Image


class WSISlide:
    def __init__(self, path: Path, *, slide_mpp: float | None = None):
        self.path = path
        self.backbone_slide = open_slide(path)

        self.__slide_mpp = (
            slide_mpp
            if slide_mpp
            else float(self.backbone_slide.properties[openslide.PROPERTY_NAME_MPP_X])
        )

    @property
    def base_dimensions(self) -> tuple[int, int]:
        return self.backbone_slide.dimensions

    @property
    def image_size(self) -> tuple[int, int]:
        return self.base_dimensions

    @property
    def base_mpp(self) -> float:
        return self.__slide_mpp

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self.backbone_slide.get_best_level_for_downsample(downsample)

    def get_image(self, target_mpp: float | None = None) -> Image:
        if target_mpp:
            donwsample_ratio = target_mpp / self.base_mpp
            image_dimensions = (
                ceil(self.base_dimensions[0] / donwsample_ratio),
                ceil(self.base_dimensions[1] / donwsample_ratio),
            )
        else:
            image_dimensions = self.base_dimensions

        return self.backbone_slide.get_thumbnail(image_dimensions)
