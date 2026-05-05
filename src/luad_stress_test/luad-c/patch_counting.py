from collections import Counter
from pathlib import Path
from tabulate import tabulate

from stress_test_preprocessing.constants import ADN_DIR, IMG_DIR, PROCESSED_DIR

def patch_count(
        patch_path: Path, 
        class_names: list[str],
        patch_prefix: str|None = None,
        patch_ext: str = ".png"
):
    file_list = [f for f in patch_path.iterdir() if f.suffix == patch_ext]

    if patch_prefix is not None:
        file_list = [f for f in file_list if f.name.startswith(patch_prefix)]


    found_classes = [class_name for f in file_list for class_name in class_names if class_name in f.name]

    patch_count = Counter(found_classes)

    for name in class_names:
        patch_count.setdefault(name, 0)

    patch_count["Total"] = sum(patch_count.values())

    return patch_count


if __name__ == "__main__":

    img_size = 384

    count = patch_count(
        PROCESSED_DIR / f"TCGA-LUAD_pattern_patches_{img_size}x{img_size}_v4",
        class_names=[
            'SOLID',
            'NC',
            'MICROPAP',
            'ACINAR',
            'CRIB',
        ],
        patch_prefix="TCGA"
    )

    print(count)
