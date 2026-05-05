import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore
from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize  # type: ignore

from luad_stress_test.models.encoders.base_model import BaseModel


class ResNet18Encoder(BaseModel):
    def __init__(self, freeze_: bool = False, pretrained: bool = False) -> None:
        super().__init__(freeze_)

        self.model = resnet18(weights="DEFAULT" if pretrained else None)
        self._num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.transform = Compose(
            [
                Resize(224),
                ToDtype(dtype=torch.float16, scale=True),
                Normalize(
                    mean=(0.4850, 0.4560, 0.4060),
                    std=(0.2290, 0.2240, 0.2250),
                ),
            ]
        )

    @property
    def num_features(self) -> int:
        return self._num_features

    def forward(self, image):
        transformed = self.transform(image)

        return self.model(transformed)
