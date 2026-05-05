from dataclasses import dataclass

from luad_stress_test.wsi.utils.patch_sample import PatchSample


@dataclass
class TissueSample:
    x: float  # left edge of the sample in microns from begining of the image
    y: float  # top edge of the sample in microns from begining of the image
    tissue_size: float  # tissue size in microns

    @staticmethod
    def from_patch_sample(
        pixel_sample: PatchSample, patch_mpp: float
    ) -> "TissueSample":
        return TissueSample(
            pixel_sample.x * patch_mpp,
            pixel_sample.y * patch_mpp,
            pixel_sample.tissue_size * patch_mpp,
        )
