from timm import create_model
from timm.models import VisionTransformer
import torch
from torch import Tensor
from torchvision.transforms.v2 import (  # type: ignore
    Compose,
    CenterCrop,
    Normalize,
    Resize,
    InterpolationMode,
    ToDtype,
)

from luad_stress_test.predict.models.encoders.base_model import BaseModel


class GigaPathEncoder(BaseModel):
    def __init__(self, freeze_: bool = True) -> None:
        super().__init__(freeze_)

        self.model: VisionTransformer = create_model(  # type: ignore
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )

        self.transform = Compose(
            [
                Resize(256, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToDtype(torch.half, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @property
    def num_features(self) -> int:
        return self.model.embed_dim

    def forward(self, X: Tensor) -> Tensor:
        transformed = self.transform(X)

        return self.model(transformed)
