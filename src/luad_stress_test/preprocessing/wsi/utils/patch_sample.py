from dataclasses import dataclass


@dataclass
class PatchSample:
    x: int  # left edge of the sample in pixels from begining of the image
    y: int  # top edge of the sample in pixels from begining of the image
    tissue_size: int  # sample size in pixels
