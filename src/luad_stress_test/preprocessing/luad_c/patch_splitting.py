import json
from pathlib import Path
from PIL import Image, ImageDraw
from math import ceil
import numpy as np
from openslide import OpenSlide

from luad_stress_test.preprocessing.luad_c.polygon_extraction import (
    extract_coords_from_xml,
    get_bbox,
)


def get_patch_start_pixels(roi_dim, patch_dim, pct_overlap=0.0, align_last=False):

    new_patch_dim = ceil((1 - pct_overlap) * patch_dim)

    # print(roi_dim, "/", patch_dim, "=", roi_dim//patch_dim)
    # print(roi_dim, "/", new_patch_dim, "=", roi_dim//new_patch_dim)

    indicies = [i for i in range(0, roi_dim - patch_dim, new_patch_dim)]

    if align_last:
        indicies[-1] = roi_dim - patch_dim + 1

    effective_pct_overlap = (patch_dim - new_patch_dim) / patch_dim

    return len(indicies), effective_pct_overlap, indicies


def save_patch(
    center_point,
    size,
    image,
    mask,
    mask_area_criterion,
    patch_dir,
    patch_name,
    img_ext="png",
    save_mask: bool = False,
):

    left, top = (center - size // 2 for center, size in zip(center_point, size))

    if left < 0:
        # print(patch_name)
        # print(f"Left = {left}")
        return 0, 0, 1

    if top < 0:
        # print(patch_name)
        # print(f"Top = {top}")
        return 0, 0, 1

    W, H = image.size
    right, bottom = left + size[0], top + size[1]

    if right > W:
        # print(patch_name)
        # print(f"Right = {right}, W = {W} ({W-right})")
        return 0, 0, 1

    if bottom > H:
        # print(patch_name)
        # print(f"Bottom = {bottom}, H = {H} ({H-bottom})")
        return 0, 0, 1

    cropped_patch = image.crop((left, top, right, bottom))
    cropped_mask = mask.crop((left, top, right, bottom))

    mask_pattern_pct_area = np.array(cropped_mask).mean()

    if mask_pattern_pct_area < mask_area_criterion:
        print(f"BAD patch pattern area: {mask_pattern_pct_area:.4f}")
        if save_mask:
            mask_path = (
                patch_dir / "bad_mask" / f"{patch_name}_x{left}_y{top}.{img_ext}"
            )
            cropped_mask.save(mask_path)
        return 0, 1, 0

    patch_path = patch_dir / f"{patch_name}_x{left}_y{top}.{img_ext}"
    cropped_patch.save(patch_path)

    return 1, 0, 0


def save_patches_from_file(
    slide_path,
    dest_dir,
    slide_level,
    verticies,
    patch_size,
    patch_pattern_area_criterion,
    target_mpp=None,
    overlap_pct=0.0,
    dim_attrs=None,
    coord_attrs=None,
    img_ext="png",
):

    dim_attrs = dim_attrs or ["W", "H"]
    coord_attrs = coord_attrs or ["X", "Y"]

    slide = OpenSlide(slide_path)

    factor = slide.level_downsamples[slide_level]

    tgt_mpp = target_mpp or (0.9, 0.9)

    if isinstance(tgt_mpp, float):
        tgt_mpp = (tgt_mpp, tgt_mpp)

    orig_mpp = tuple(float(slide.properties[f"openslide.mpp-{d}"]) for d in ["x", "y"])
    mpp_ratio = tuple(mpp1 / mpp2 for mpp1, mpp2 in zip(orig_mpp, tgt_mpp))

    desc_dict = {"slide": slide_path.name, "annotations": []}

    all_patch_count = 0
    all_oo_range = 0

    for cls_name, cls_polygons in verticies.items():

        print("\n", cls_name, len(cls_polygons))
        print("-" * 24)

        for i, polygon in enumerate(cls_polygons):

            bbox_min_point, bbox_size = get_bbox(polygon, factor)
            polygon_relative = [
                tuple(p - anchor for p, anchor in zip(point, bbox_min_point))
                for point in polygon
            ]
            # print("Polygon points:", *polygon_relative[0:4], "...", "\n")

            region = slide.read_region(bbox_min_point, slide_level, bbox_size)  # type: ignore
            region = region.convert("RGB")
            mask = Image.new("1", bbox_size, 0)  # type: ignore
            ImageDraw.Draw(mask).polygon(polygon_relative, outline=1, fill=1)

            tgt_size = tuple(
                int(dim * ratio) for dim, ratio in zip(bbox_size, mpp_ratio)
            )

            resized_region = region.resize(tgt_size, Image.Resampling.LANCZOS)  # type: ignore
            resized_mask = mask.resize(tgt_size, Image.Resampling.LANCZOS)  # type: ignore

            bbox_size_w, bbox_size_h = bbox_size
            patch_width, patch_height = patch_size
            center_x, center_y = bbox_size_w // 2, bbox_size_h // 2

            print(f"Center point: ({center_x}, {center_y})")
            patches_saved, bad_patches, oo_range = 0, 0, 0

            step_x = int(patch_width * (1 - overlap_pct))
            step_y = int(patch_height * (1 - overlap_pct))

            for offset_x in range(-center_x, center_x + 1, step_x):

                for offset_y in range(-center_y, center_y + 1, step_y):

                    anchor_x = center_x + offset_x
                    anchor_y = center_y + offset_y

                    print(f"Anchor point: ({anchor_x}, {anchor_y})")
                    good_patch, bad_patch, patch_oo_range = save_patch(
                        center_point=(anchor_x, anchor_y),
                        size=patch_size,
                        image=resized_region,
                        mask=resized_mask,
                        mask_area_criterion=patch_pattern_area_criterion,
                        patch_dir=dest_dir,
                        patch_name=f"{slide_path.stem}__{cls_name}_{i}",
                        img_ext=img_ext,
                        save_mask=False,
                    )
                    patches_saved += good_patch
                    bad_patches += bad_patch
                    oo_range += patch_oo_range

            desc_dict["annotations"].append(
                {
                    "class": cls_name,
                    "patches_saved": patches_saved,
                    "bad_patches": bad_patches,
                    "top_left": {
                        coord_name: val
                        for coord_name, val in zip(coord_attrs, bbox_min_point)
                    },
                    "original_size": {
                        dim_name: val for dim_name, val in zip(dim_attrs, bbox_size)
                    },
                    "target_size": {
                        dim_name: val for dim_name, val in zip(dim_attrs, tgt_size)
                    },
                }
            )
            all_patch_count += patches_saved
            all_oo_range += oo_range

    return desc_dict, all_patch_count, all_oo_range


def save_patches_from_dir(
    slide_dir: Path,
    xml_dir: Path,
    dest_dir: Path,
    patch_size: tuple,
    patch_pattern_area_criterion: float,
    overlap_pct: float,
    target_mpp: float,
    xml_element: str = "Region",
    annot_attr: str = "Text",
    coord_xml_element="Vertex",
    coord_attrs: list | None = None,
    slide_level: int = 0,
    img_ext: str = "png",
    slide_ext: str = "svs",
):
    coord_attrs = coord_attrs or ["X", "Y"]

    if not dest_dir.is_dir():
        dest_dir.mkdir(parents=True)

    description = []
    all_patch_count = 0

    for xml_file in xml_dir.iterdir():

        slide_name = xml_file.stem
        slide_file = slide_dir / f"{slide_name}.{slide_ext}"

        if not slide_file.is_file():
            raise RuntimeError(f"Slide image not found {slide_file}")

        if not xml_file.is_file():
            raise RuntimeError(f"XML annotation file not found {xml_file.name}")

        verticies = extract_coords_from_xml(
            xml_file,
            xml_element,
            annot_attr,
            coord_xml_element,
            coord_attrs,
        )

        region_desc, patch_count, oo_range_count = save_patches_from_file(
            slide_file,
            dest_dir,
            slide_level,
            verticies,
            patch_size,
            patch_pattern_area_criterion,
            target_mpp,
            overlap_pct=overlap_pct,
            img_ext=img_ext,
        )
        description.append(region_desc)
        print("PATCH COUNT:", patch_count, "OO RANGE COUNT:", oo_range_count)
        all_patch_count += patch_count

    with open(dest_dir / "description.json", "w", encoding="utf-8") as f:
        json.dump(description, f, sort_keys=False, indent=2)

    print("SAVED PATCH COUNT: ", all_patch_count)
