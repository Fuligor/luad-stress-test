from PIL.Image import Image

from luad_stress_test.wsi.tissue_image import WSISlide
from luad_stress_test.wsi.utils import PatchSample, TissueSample


class _WSIPatchSampler:
    def __init__(self, slide: WSISlide, x: int, y: int, patch_size: int):
        self.slide = slide

        self.pixel_sample = PatchSample(x, y, patch_size)
        self.tissue_sample = TissueSample.from_patch_sample(
            self.pixel_sample, self.slide.base_mpp
        )

        self._patch: Image | None = None

    def get(self) -> Image:
        if self._patch is None:
            self._patch = self.slide.backbone_slide.read_region(
                (self.pixel_sample.x, self.pixel_sample.y),
                0,
                (self.pixel_sample.tissue_size, self.pixel_sample.tissue_size),
            ).convert("RGB")

        return self._patch
