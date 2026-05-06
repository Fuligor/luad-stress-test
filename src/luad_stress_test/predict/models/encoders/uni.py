import timm
from timm.models import VisionTransformer
import torch
from torch import Tensor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize  # type: ignore

from luad_stress_test.predict.models.encoders.base_model import BaseModel


class UniEncoder(BaseModel):
    def __init__(self, freeze_: bool = True) -> None:
        super().__init__(freeze_)

        self.model: VisionTransformer = timm.create_model(  # type: ignore
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
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
        return self.model.embed_dim

    def forward(self, X: Tensor) -> Tensor:
        transformed = self.transform(X)
        return self.model(transformed)
