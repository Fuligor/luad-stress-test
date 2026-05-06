# pylint: disable=arguments-differ
from contextlib import nullcontext
from typing import Any
import lightning as L
import torch
from torch import nn, Tensor
from lightning.pytorch.loggers.neptune import NeptuneLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchvision.transforms.v2 import MixUp  # type: ignore

from luad_stress_test.utils import Label
from luad_stress_test.predict.metrics.prepare_metrics import prepare_metrics
from luad_stress_test.predict.metrics.confusion_matrix import ConfusionMatrix
from luad_stress_test.predict.models.encoders import BaseModel as AbstractEncoder
from luad_stress_test.predict.models.heads import AbstractHead


class ClassificationModel(L.LightningModule):
    def __init__(
        self,
        encoder: AbstractEncoder,
        head: AbstractHead,
        finetune_encoder: bool = False,
        train_mixup: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.finetune_encoder = finetune_encoder
        self.encoder_context_manager = (
            nullcontext if finetune_encoder else torch.no_grad
        )

        self.head = head
        self.objective = nn.CrossEntropyLoss()

        self.confusion_matrix = ConfusionMatrix(
            num_classes=len(Label), normalize="true"
        )
        self.train_metrics: MetricCollection = prepare_metrics(len(Label))
        self.train_metrics.prefix = "train/"
        self.val_metrics: MetricCollection = prepare_metrics(len(Label))
        self.val_metrics.prefix = "val/"
        self.test_metrics: MetricCollection = prepare_metrics(len(Label))
        self.test_metrics.prefix = "test/"

        self.test_images: dict[str, torch.Tensor] = {}
        self.test_reconstructions: dict[str, torch.Tensor] = {}
        self.pred_color = torch.tensor([0, 255, 0])[:, None, None]
        self.train_mixup = train_mixup

        self.mixup = MixUp(num_classes=len(Label))

        self.strict_loading = finetune_encoder

    def state_dict(  # type: ignore
        self, *, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        return {
            k: v
            for k, v in super().state_dict(prefix=prefix, keep_vars=keep_vars).items()
            if (self.finetune_encoder or "encoder" not in k)
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["hyper_parameters"]["head"]["init_args"][
            "input_features"
        ] = self.encoder.output_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with self.encoder_context_manager():
            features: torch.Tensor = self.encoder(x)

        return {"features": features, "logits": self.head(features)}

    def _eval(
        self,
        predicted: torch.Tensor,
        expected: torch.Tensor,
        metrics: MetricCollection,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
    ):
        loss = self.objective(predicted, expected)
        metrics(predicted, expected if expected.dim() == 1 else expected.argmax(-1))

        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch)

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor]) -> STEP_OUTPUT:
        x, y = batch

        pred: Tensor

        if self.train_mixup:
            with self.encoder_context_manager():
                features: torch.Tensor = self.encoder(x)
                features, y = self.mixup(features[:, None, None], y)

            pred = self.head(features.squeeze(1, 2))
        else:
            pred = self(x)["logits"]

        loss = self._eval(pred, y, self.train_metrics)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch) -> None:
        x, y = batch
        pred: Tensor = self(x)["logits"]

        loss = self._eval(pred, y, self.val_metrics, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch) -> None:
        x, y = batch
        preds: Tensor = self(x)["logits"]
        self._eval(preds, y, self.test_metrics, on_step=False, on_epoch=True)
        self.confusion_matrix(preds, y)

    def predict_step(self, batch) -> dict[str, torch.Tensor]:
        x, _ = batch
        preds = self(x)

        return preds

    def on_test_start(self):
        self.test_images = {}
        self.test_reconstructions = {}

        return super().on_test_start()

    def on_test_epoch_end(self) -> None:
        namespace = "test"

        if isinstance(self.logger, NeptuneLogger):
            confusion_matrix: torch.Tensor = self.confusion_matrix.compute()
            id2label = {label.value: label.name for label in Label}
            labels = [i[1] for i in sorted(id2label.items(), key=lambda x: x[0])]

            fig, _ = self.confusion_matrix.cpu().plot(confusion_matrix, labels=labels)
            self.logger.experiment[f"{namespace}/confusion matrix"].upload(fig)

        self.confusion_matrix.reset()
        return super().on_test_epoch_end()
