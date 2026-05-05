from copy import deepcopy
from math import ceil
from pathlib import Path

from PIL.Image import Image, Resampling, Resampling

from luad_stress_test.wsi.utils import GridLocation, TissueSample


class WSIPatchMetadata:
    def __init__(
        self,
        tissue_sample: TissueSample,
        slide_path: Path,
        grid_position: GridLocation,
    ):
        self._tissue_sample = tissue_sample
        self._slide_path = slide_path
        self._grid_position = grid_position

    @property
    def slide_path(self) -> Path:
        return self._slide_path

    @property
    def tissue_sample(self) -> TissueSample:
        return self._tissue_sample

    @property
    def grid_position(self) -> GridLocation:
        return self._grid_position

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "column": str(self.grid_position.column),
            "row": str(self.grid_position.row),
            "slide_name": self.slide_path.stem,
        }

    def get_patch_name(self, tamplate_name: str) -> str:
        output_patch_name = deepcopy(tamplate_name)

        for name, value in self.metadata.items():
            output_patch_name = output_patch_name.replace("{" + name + "}", value)

        return output_patch_name

    def create_patch(self, image: Image) -> "WSIPatch":
        return WSIPatch(
            image,
            self._tissue_sample,
            self._slide_path,
            self._grid_position,
        )


class WSIPatch(WSIPatchMetadata):
    def __init__(
        self,
        image: Image,
        tissue_sample: TissueSample,
        slide_path: Path,
        grid_position: GridLocation,
    ):
        WSIPatchMetadata.__init__(
            self,
            tissue_sample,
            slide_path,
            grid_position,
        )

        self._image = image
        self._image_mpp = self._tissue_sample.tissue_size / self.image_size[0]

    @property
    def base_mpp(self):
        return self._image_mpp

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image.size

    @property
    def pil_image(self) -> Image:
        return self._image

    def get_image(self, target_mpp: float | None = None) -> Image:
        if target_mpp:
            downsample_ratio = target_mpp / self.base_mpp
            target_size = [ceil(i / downsample_ratio) for i in self.image_size]

            return self._image.resize(target_size, Resampling.BICUBIC)

        return self._image
