from abc import ABCMeta, abstractmethod
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode  # type: ignore
from torchvision.transforms import v2  # type: ignore

from luad_stress_test.utils import Label


class AbstractPatchDataset(Dataset, metaclass=ABCMeta):
    _patch_list: list[Path]

    def __init__(self, path: Path, transform: v2.Transform | None = None) -> None:
        self._root_path = path
        self._transform = transform

    @property
    def root_path(self) -> Path:
        return self._root_path

    def patch_name(self, index: int) -> str:
        path = self._patch_list[index]

        return path.name

    @abstractmethod
    def patch_label(self, index: int) -> Label:
        pass

    def __len__(self) -> int:
        return len(self._patch_list)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path = self._patch_list[index]
        patch = read_image(path, mode=ImageReadMode.RGB)

        if self._transform:
            patch = self._transform(patch)

        return patch, self.patch_label(index).value
