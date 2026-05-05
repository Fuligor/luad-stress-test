from pathlib import Path

from luad_stress_test.wsi.saturation_tissue_mask import SaturationTissueMask
from luad_stress_test.wsi.tiler import WSITiler
from luad_stress_test.wsi.tissue_image import WSISlide

from torchvision.transforms import v2  # type: ignore

import pandas as pd

from luad_stress_test.utils import Label
from luad_stress_test.dataset.abstract_patch_dataset import AbstractPatchDataset
from luad_stress_test.path_manager import PathManager


class DHMCDataset(AbstractPatchDataset):
    __labels_mapping = {
        "solid": Label.SOL,
        "lepidic": Label.LEP,
        "acinar": Label.ACC,
        "micropapillary": Label.MIP,
        "papillary": Label.PAP,
        "cribriform": Label.CRB,
    }

    def __init__(self, path: Path, transform: v2.Transform | None = None) -> None:
        super().__init__(path, transform)
        self._patch_list = [file_name for file_name in self._root_path.glob("*.png")]

        metadata = pd.read_csv(PathManager.dhmc_metadata(), index_col=0)
        self.label = DHMCDataset.__labels_mapping[
            str(metadata.loc[self._root_path.stem + ".tif"]["Class"])
        ]

    def patch_label(self, _: int) -> Label:
        return self.label


def prepare_dhmc_patches():
    dhmc_path = PathManager.data_raw("dhmc")
    tiles_path = PathManager.data_processed("dhmc/patches")
    metadata = pd.read_csv(dhmc_path / "metadata.csv")
    mask = SaturationTissueMask(7.2, saturation_threshold=20)
    tiler = WSITiler(mask=mask, target_mpp=0.9, patch_size=384, offset=333)

    for _, row in metadata.iterrows():
        file_name = row["File Name"]
        mpp = row["Microns Per Pixel"]

        slide = WSISlide(dhmc_path / "wsi" / file_name, slide_mpp=mpp)
        tiler.save_patch_set(slide, tiles_path / Path(file_name).stem, "{row}-{column}")


if __name__ == "__main__":
    prepare_dhmc_patches()
