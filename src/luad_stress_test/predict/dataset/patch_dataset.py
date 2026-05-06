from pathlib import Path
import re

from torchvision.transforms import v2  # type: ignore

from luad_stress_test.utils import Label
from luad_stress_test.predict.dataset.abstract_patch_dataset import AbstractPatchDataset


class PatchDataset(AbstractPatchDataset):
    __labels_mapping = {
        "N": Label.NTU,
        "A": Label.ACC,
        "C": Label.CRB,
        "L": Label.LEP,
        "M": Label.MIP,
        "P": Label.PAP,
        "S": Label.SOL,
    }

    def __init__(self, path: Path, transform: v2.Transform | None = None) -> None:
        super().__init__(path, transform)
        self._name_template = re.compile(
            r"^(?P<dataset_name>.+)_(?P<slide_id>\d+)_(?P<patch_id>\d+)_(?P<class_label>[NACLMPS])\.png$"
        )
        self._patch_list = [
            file_name
            for file_name in self._root_path.iterdir()
            if self._name_template.match(file_name.name)
        ]

    def patch_label(self, index: int) -> Label:
        path = self._patch_list[index]
        match = self._name_template.match(path.name)
        assert (
            match is not None
        ), f"File name {path.name} does not match the expected pattern"

        return PatchDataset.__labels_mapping[match.group("class_label")]
