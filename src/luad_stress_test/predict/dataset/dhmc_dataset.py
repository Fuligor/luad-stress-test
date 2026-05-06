from pathlib import Path

from torchvision.transforms import v2  # type: ignore

import pandas as pd

from luad_stress_test.utils import Label
from luad_stress_test.predict.dataset.abstract_patch_dataset import AbstractPatchDataset
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
