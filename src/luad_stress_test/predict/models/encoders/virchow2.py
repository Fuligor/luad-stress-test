from typing import Literal
import timm
from timm.models import VisionTransformer
from timm.layers import SwiGLUPacked
import torch
from torch import Tensor
from torchvision.transforms.v2 import (  # type: ignore
    Compose,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from luad_stress_test.predict.models.encoders.base_model import BaseModel


class Virchow2Encoder(BaseModel):
    def __init__(
        self, freeze_: bool = True, pooling_mode: Literal["cls", "mean"] = "cls"
    ) -> None:
        super().__init__(freeze_)

        self.model: VisionTransformer = timm.create_model(  # type: ignore
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.pooling_mode = pooling_mode

        self.transform = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                ToDtype(dtype=torch.float16, scale=True),
                Normalize(
                    mean=(0.4850, 0.4560, 0.4060),
                    std=(0.2290, 0.2240, 0.2250),
                ),
            ]
        )

    @property
    def num_features(self) -> int:
        return (
            self.model.embed_dim
            if self.pooling_mode == "cls"
            else 2 * self.model.embed_dim
        )

    def forward(self, X: Tensor) -> Tensor:
        transformed = self.transform(X)
        output = self.model(transformed)

        class_token: Tensor = output[:, 0]  # size: 1 x 1280
        patch_tokens: Tensor = output[
            :, 5:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        match self.pooling_mode:
            case "cls":
                return class_token
            case "mean":
                return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
