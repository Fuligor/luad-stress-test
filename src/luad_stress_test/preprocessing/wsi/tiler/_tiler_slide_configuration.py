from math import ceil, floor
from typing import Optional

from luad_stress_test.preprocessing.wsi.tissue_image import WSISlide
from luad_stress_test.preprocessing.wsi.utils import Grid


class _WSITilerSlideConfiguration:
    def __init__(
        self,
        slide: WSISlide,
        patch_size: int,
        target_mpp: Optional[float],
        offset: int,
    ):
        self.downsample = self._get_downsample(target_mpp, slide.base_mpp)
        self.target_mpp = target_mpp
        self.patch_size = ceil(patch_size * self.downsample)
        self.offset = ceil(offset * self.downsample)
        self.columns, self.rows = slide.base_dimensions
        self.patches_grid = Grid(
            columns=floor((self.columns - self.patch_size) / self.offset + 1),
            rows=floor((self.rows - self.patch_size) / self.offset + 1),
        )

    @staticmethod
    def _get_downsample(target_mpp: Optional[float], slide_mpp: float) -> float:
        return target_mpp / slide_mpp if target_mpp else 1.0

    @property
    def x_range(self):
        return range(0, self.offset * self.patches_grid.columns, self.offset)

    @property
    def y_range(self):
        return range(0, self.offset * self.patches_grid.rows, self.offset)
