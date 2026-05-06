from collections import defaultdict
from pathlib import Path

from lightning.pytorch.callbacks import BasePredictionWriter
import torch


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, write_embedings: bool = False):
        super().__init__(write_interval="epoch")

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embedings = write_embedings

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        raise NotImplementedError("PredictionWriter only supports writing on epoch end")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices) -> None:
        results = defaultdict(list)

        for prediction in predictions:
            for key, value in prediction.items():
                if not self.write_embedings and key == "features":
                    continue

                results[key].append(value.cpu())

        for batch_index in batch_indices[0]:
            results["batch_indices"].append(torch.tensor(batch_index))

        torch.save(
            {key: torch.concat(value) for key, value in results.items()},
            self.output_dir / "predictions.pt",
        )
