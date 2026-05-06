from torch import nn, Tensor

from luad_stress_test.predict.models.heads.abstract import AbstractHead
from luad_stress_test.utils import Label


class LinearHead(AbstractHead):
    def __init__(self, input_features: int, num_classes: int = len(Label)):
        super().__init__(input_features, num_classes)

        self.linear = nn.Sequential(
            # nn.Dropout(dropout),
            nn.InstanceNorm1d(input_features),
            nn.Linear(
                in_features=input_features,
                out_features=num_classes,
            ),
        )

    def forward(self, X: Tensor) -> Tensor:
        features = self.linear(X)

        return features
