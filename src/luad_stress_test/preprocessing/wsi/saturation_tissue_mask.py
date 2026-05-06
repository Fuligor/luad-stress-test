import numpy as np

from PIL import ImageFilter

from luad_stress_test.preprocessing.wsi.tissue_image.wsi_patch import WSIPatch
from luad_stress_test.preprocessing.wsi.tissue_image.wsi_slide import WSISlide


class SaturationTissueMask:
    def __init__(
        self,
        mask_mpp,
        median_kernel_size: int = 7,
        closing_kernel_size: int = 7,
        saturation_threshold: int = 20,
    ):
        self.mask_mpp = mask_mpp

        self.median_filter = ImageFilter.MedianFilter(median_kernel_size)
        self.closing_kernel_size = closing_kernel_size
        self.saturation_theshold = saturation_threshold

    def __call__(self, slide: WSISlide | WSIPatch) -> np.ndarray:
        image_slide = slide.get_image(self.mask_mpp)

        thumbnail = image_slide.convert("HSV")
        saturation = thumbnail.getchannel("S")
        saturation_med = saturation.filter(self.median_filter)
        mask = saturation_med.point(
            lambda x: 255 if x > self.saturation_theshold else 0
        )
        mask_closed = mask.filter(
            ImageFilter.MaxFilter(self.closing_kernel_size)
        ).filter(ImageFilter.MinFilter(self.closing_kernel_size))

        return np.array(mask_closed, dtype=np.uint8)
