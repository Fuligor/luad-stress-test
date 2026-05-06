import torch
from transformers import EfficientNetImageProcessorFast, EfficientNetModel
from torch import Tensor

from luad_stress_test.predict.models.encoders.base_model import BaseModel


class EfficientNetEncoder(BaseModel):
    def __init__(self, freeze_: bool = False) -> None:
        super().__init__(freeze_)

        self.model: EfficientNetModel = EfficientNetModel.from_pretrained(
            "google/efficientnet-b0"
        ).to(memory_format=torch.channels_last)
        self.transform = EfficientNetImageProcessorFast.from_pretrained(
            "google/efficientnet-b0"
        )

    @property
    def num_features(self) -> int:
        return self.model.pooler.kernel_size  # type: ignore

    def forward(self, X: Tensor) -> Tensor:
        transformed = self.transform(X, return_tensors="pt")["pixel_values"].to(
            memory_format=torch.channels_last
        )

        return self.model.forward(transformed).pooler_output
