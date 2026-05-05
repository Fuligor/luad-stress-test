import torch
from torch import nn, Tensor
from torchvision.models import vit_b_16  # type: ignore
from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize  # type: ignore


from luad_stress_test.models.encoders.base_model import BaseModel


class ViTB16Encoder(BaseModel):
    def __init__(self, freeze_: bool = False, pretrained: bool = True):
        super().__init__(freeze_)

        self.model = vit_b_16(weights="DEFAULT" if pretrained else None)
        self.model.heads = nn.Identity()

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
        return 768

    def forward(self, x: Tensor) -> Tensor:
        transformed = self.transform(x)

        return self.model(transformed)
