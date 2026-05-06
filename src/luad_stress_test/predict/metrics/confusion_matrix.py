from typing import Any

from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay  # type: ignore


class ConfusionMatrix(MulticlassConfusionMatrix):
    def __init__(self, labels=None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.labels = labels

    def plot(self, *_: Any, **__: Any) -> Any:
        display_format = "d" if self.normalize == "none" else ".2f"

        display = ConfusionMatrixDisplay(
            self.compute(), display_labels=self.labels
        ).plot(
            include_values=True,
            cmap="Blues",
            values_format=display_format,
            colorbar=False,
        )
        display.ax_.tick_params(axis="x", labelrotation=45)
        display.ax_.tick_params(axis="y", labelrotation=45)

        return display.figure_, display.ax_
