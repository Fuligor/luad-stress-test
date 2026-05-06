from collections import Counter
from pathlib import Path


def patch_count(
    patch_path: Path,
    class_names: list[str],
    patch_prefix: str | None = None,
    patch_ext: str = ".png",
):
    file_list = [f for f in patch_path.iterdir() if f.suffix == patch_ext]

    if patch_prefix is not None:
        file_list = [f for f in file_list if f.name.startswith(patch_prefix)]

    found_classes = [
        class_name
        for f in file_list
        for class_name in class_names
        if class_name in f.name
    ]

    count = Counter(found_classes)

    for name in class_names:
        count.setdefault(name, 0)

    count["Total"] = sum(count.values())

    return count
