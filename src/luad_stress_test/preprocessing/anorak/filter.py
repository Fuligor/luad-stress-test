"""Slightly modified version of the original code from ANORAK repository:
http://github.com/xi11/AIgrading
"""

from collections import defaultdict
from dataclasses import dataclass
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2  # type: ignore

from tqdm import tqdm  # type: ignore
import pandas as pd

labels_names = {0: "N", 1: "C", 2: "M", 3: "S", 4: "P", 5: "A", 6: "L"}


@dataclass
class SlideRepresentation:
    id_: int
    count: int = 0


def read_img(data_file, label_file):  # to be checked with if file or path
    data = cv2.imread(data_file)  # pylint: disable=no-member
    labels = cv2.imread(label_file)  # pylint: disable=no-member
    return data, labels


def get_label_file(data_file, label_path):  # get mask with corresponding name
    data_file_name = os.path.basename(data_file)
    label_file = os.path.join(label_path, data_file_name)
    return label_file, data_file_name[:-4]


def write_to_patch(data_files, label_path, save_path):
    name_matching: List[Tuple[str, str]] = []
    slide_representations: Dict[str, SlideRepresentation] = {}

    patch_statistics = defaultdict(int)

    for file_n in tqdm(range(len(data_files))):
        curr_data_file = str(data_files[file_n])
        curr_label_file, file_base_name = get_label_file(curr_data_file, label_path)
        data, labels = read_img(curr_data_file, curr_label_file)
        # num_patch = extract_patches_img_label(data, labels, save_path, file_base_name, img_patch_h=768, img_patch_w=768, stride_h=768, stride_w=768, label_patch_h=768, label_patch_w=768)
        label_index = filter_patches_img_label(labels, threshold=0.6)

        if label_index is not None:
            label_name = labels_names[label_index]

            match_obj = re.match(
                r"^train(?P<slide_name>.+)_(?P<patch_id>\d+)$", file_base_name
            )
            if match_obj is None:
                raise ValueError(f"Cound not parse patch name for {file_base_name}")

            slide_name: str = match_obj.group("slide_name")
            if slide_name not in slide_representations:
                slide_representations[slide_name] = SlideRepresentation(
                    len(slide_representations) + 1, 1
                )

            slide_representation: SlideRepresentation = slide_representations[
                slide_name
            ]

            save_name = f"A_{slide_representation.id_:03d}_{slide_representation.count:03d}_{label_name}"
            slide_representation.count += 1
            name_matching.append((file_base_name, save_name))
            patch_statistics[label_name] += 1

            cv2.imwrite(  # pylint: disable=no-member
                os.path.join(save_path, save_name + ".png"),
                cv2.resize(data, (384, 384)),  # pylint: disable=no-member
            )

    pd.DataFrame(name_matching, columns=["orginal_name", "mapped_name"]).to_csv(
        os.path.join(save_path, "mapping.csv")
    )
    print(patch_statistics)


def filter_patches_img_label(
    label: np.ndarray, threshold: float, background_label: int = 0
) -> Optional[int]:
    labels, counts = np.unique(label, return_counts=True)

    tumor_labels = [i for i in labels if i != background_label]

    if len(tumor_labels) > 1:
        return None

    label_share = counts / label.size

    if max(label_share) < threshold:
        return None

    label_index = np.argmax(counts)
    return labels[label_index]
