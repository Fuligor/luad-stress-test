
from pathlib import Path
from openslide import OpenSlide
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import cv2
import json

import numpy as np

from stress_test_preprocessing.constants import ADN_DIR, IMG_DIR, PROCESSED_DIR

def extract_coords_from_xml(
        xml_path: Path,
        xml_element: str = "Region",
        annot_attr: str = "Text",
        coord_xml_element = "Vertex",
        coord_attrs: list[str] = ["X", "Y"],
):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    annot_elements = list(root.findall(f".//{xml_element}"))

    annot_objects = {}
    for annotation in annot_elements:

        part_of_group = annotation.attrib.get(annot_attr, "Nieznane")

        if part_of_group not in annot_objects:
            annot_objects[part_of_group] = []

        coords = list(annotation.findall(f".//{coord_xml_element}"))

        new_polygon = []
        for coord in coords:
            v_coords = tuple(int(float(coord.attrib[c])) for c in coord_attrs)
            new_polygon.append(v_coords)

        annot_objects[part_of_group].append(new_polygon)

    return annot_objects


def get_bbox(polygon_vertices, scale_factor=None):

    if len(polygon_vertices) < 4:
        raise RuntimeError()
    
    min_point = tuple(map(min, zip(*polygon_vertices)))
    max_point = tuple(map(max, zip(*polygon_vertices)))

    if scale_factor and scale_factor!= 1:
        min_point = tuple(int(c*scale_factor) for c in min_point)
        max_point = tuple(int(c*scale_factor) for c in max_point)

    # print("MIN: ", min_point, "MAX: ", max_point)

    bbox_size = tuple(max_c - min_c for max_c, min_c in zip(max_point, min_point))

    return min_point, bbox_size


def save_polygons_from_file(
        slide_path, 
        xml_path,
        dest_dir, 
        slide_level,
        xml_element,
        annot_attr,
        coord_xml_element,
        coord_attrs,
        target_mpp=None,
):
    verticies = extract_coords_from_xml(
        xml_path,
        xml_element,
        annot_attr,
        coord_xml_element,
        coord_attrs,
    )
    dim_attrs = ["W", "H"]

    slide = OpenSlide(slide_path)

    factor = slide.level_downsamples[slide_level]

    tgt_mpp = target_mpp or (0.9, 0.9)
    print(tgt_mpp)

    if isinstance(tgt_mpp, float):
        tgt_mpp = (tgt_mpp, tgt_mpp)

    orig_mpp = tuple(float(slide.properties[f'openslide.mpp-{d}']) for d in ["x", "y"])
    mpp_ratio = tuple(mpp1/mpp2 for mpp1, mpp2 in zip(orig_mpp, tgt_mpp))

    desc_dict = {"slide": slide_path.name, "annotations": []}

    for cls_name, cls_polygons in verticies.items():

        print(cls_name, len(cls_polygons))
        print("-"*24)

        for i, polygon in enumerate(cls_polygons):

            bbox_min_point, bbox_size = get_bbox(polygon, factor)

            patch = slide.read_region(bbox_min_point, slide_level, bbox_size) # type: ignore
            patch = patch.convert('RGB')
            patch_np = np.array(patch)

            mask = Image.new('L', bbox_size, 0) # type: ignore

            polygon_relative = list(map(lambda point: tuple(p-anchor for p, anchor in zip(point, bbox_min_point)), polygon))

            print("Polygon points:", *polygon_relative[0:4], "...", "\n")

            ImageDraw.Draw(mask).polygon(polygon_relative, outline=1, fill=255)
            mask_np = np.array(mask)

            tgt_size = tuple(int(dim*ratio) for dim, ratio in zip(bbox_size, mpp_ratio))

            masked_image = cv2.bitwise_and(patch_np, patch_np, mask=mask_np)
            masked_image = cv2.resize(masked_image, tgt_size, interpolation=cv2.INTER_AREA)

            patch_path = dest_dir / f"{slide_path.stem}__{cls_name}_{i}.png"

            Image.fromarray(masked_image).save(patch_path)

            desc_dict["annotations"].append({
                "class": cls_name,
                "top_left": {coord_name: val for coord_name, val in zip(coord_attrs, bbox_min_point)},
                "original_size": {dim_name: val for dim_name, val in zip(dim_attrs, bbox_size)},
                "target_size": {dim_name: val for dim_name, val in zip(dim_attrs, tgt_size)},
            })

    return desc_dict



def save_polygons_from_dir(
        slide_dir: Path, 
        xml_dir: Path,
        dest_dir: Path, 
        slide_level: int,
        save_description_json: bool = False, 
        target_mpp: tuple|None = None,
        xml_element: str = "Region",
        annot_attr: str = "Text",
        coord_xml_element = "Vertex",
        coord_attrs: list|None = None,
        slide_ext: str = "svs",
):
    coord_attrs = coord_attrs or ["X", "Y"]

    if not dest_dir.is_dir():
        dest_dir.mkdir(parents=True)

    description = []

    for file in xml_dir.iterdir():

        slide_name = file.stem
        slide_file = slide_dir / f"{slide_name}.{slide_ext}"
        xml_file = xml_dir / f"{slide_name}.xml"

        if not slide_file.is_file():
            raise RuntimeError(f"Slide image not found {slide_file.name}")
        
        if not xml_file.is_file():
            raise RuntimeError(f"XML annotation file not found {xml_file.name}")
    
        extracted_items_desc = save_polygons_from_file(
                IMG_DIR / f"{slide_name}.svs",
                ADN_DIR / f"{slide_name}.xml",
                dest_dir=dest_dir,
                slide_level=slide_level,
                target_mpp=target_mpp,
                xml_element=xml_element,
                annot_attr=annot_attr,
                coord_xml_element=coord_xml_element,
                coord_attrs=coord_attrs,
        )

        description.append(extracted_items_desc)

    if save_description_json:
        with open(dest_dir / "descrption.json", "w") as out_file:
            json.dump(description, out_file, sort_keys=False, indent=2)

def is_slide_folder_complete(slide_folder):

    slide_file_names = [f.name for f in slide_folder.iterdir() if f.is_file()]
    if "Index.dat" not in slide_file_names:
        return False
    if "Slidedat.ini" not in slide_file_names:
        return False

    data_file_codes = [int(name[4:8]) for name in slide_file_names if "Data" in name]
    data_file_codes = sorted(data_file_codes)

    if data_file_codes[-1]+1 != len(data_file_codes):
        return False
    
    return True

def save_roi_from_dir(
        slide_dir: Path, 
        xml_dir: Path,
        dest_dir: Path, 
        slide_level: int,
        save_description_json: bool = False, 
        xml_element: str = "Region",
        annot_attr: str = "Text",
        coord_xml_element = "Vertex",
        coord_attrs: list|None = None,
        slide_ext: str = "svs",
        add_suffix: bool = True,
):
    coord_attrs = coord_attrs or ["X", "Y"]

    if not dest_dir.is_dir():
        dest_dir.mkdir(parents=True)

    description = []
    xml_files = [f for f in xml_dir.iterdir() if f.is_file() and f.suffix == ".xml"]

    for xml_file in xml_files:

        slide_name = xml_file.stem
        slide_file = slide_dir / f"{slide_name}.{slide_ext}"

        slide_folder = slide_dir / slide_name

        if not is_slide_folder_complete(slide_folder):
            print("Skipping incomplete slide", slide_name)
            continue

        print(slide_name)

        if not slide_file.is_file():
            raise RuntimeError(f"Slide image not found {slide_file}")
        
        if not xml_file.is_file():
            raise RuntimeError(f"XML annotation file not found {xml_file.name}")
    
        extracted_items_desc = save_rectangles_from_file(
                slide_file,
                xml_file,
                dest_dir=dest_dir,
                slide_level=slide_level,
                xml_element=xml_element,
                annot_attr=annot_attr,
                coord_xml_element=coord_xml_element,
                coord_attrs=coord_attrs,
                add_suffix=add_suffix,
        )

        description.append(extracted_items_desc)

    if save_description_json:
        with open(dest_dir / "descrption.json", "w") as out_file:
            json.dump(description, out_file, sort_keys=False, indent=2)


def save_rectangles_from_file(
        slide_path, 
        xml_path,
        dest_dir, 
        slide_level,
        xml_element,
        annot_attr,
        coord_xml_element,
        coord_attrs,
        add_suffix,
        custom_suffix=None, 
):
    verticies = extract_coords_from_xml(
        xml_path,
        xml_element,
        annot_attr,
        coord_xml_element,
        coord_attrs,
    )
    dim_attrs = ["W", "H"]

    slide = OpenSlide(slide_path)
    desc_dict = {"slide": slide_path.name, "annotations": []}

    n_classes = len(verticies)

    for cls_name, cls_polygons in verticies.items():

        print(cls_name, len(cls_polygons))
        print("-"*24)
        n_objects = len(cls_polygons)
        img_cls_name = custom_suffix if custom_suffix else cls_name

        for i, polygon in enumerate(cls_polygons):
            
            bbox_min_point, bbox_size = get_bbox(polygon)

            patch = slide.read_region(bbox_min_point, slide_level, bbox_size) # type: ignore
            patch = patch.convert('RGB')

            if not add_suffix or (n_classes==1 and n_objects==1):
                suffix = ""
            elif n_objects==1:
                suffix = f"__{img_cls_name}"
            else:
                suffix = f"__{img_cls_name}_{i}"

            roi_filename = f"{slide_path.stem}{suffix}.png"
            patch_path = dest_dir / roi_filename

            patch.save(patch_path)

            desc_dict["annotations"].append({
                "class": cls_name,
                "filename": roi_filename,
                "top_left": {coord_name: val for coord_name, val in zip(coord_attrs, bbox_min_point)},
                "size": {dim_name: val for dim_name, val in zip(dim_attrs, bbox_size)},
            })

    return desc_dict



if __name__ == "__main__":
    print(ADN_DIR)
    print(IMG_DIR)

    save_polygons_from_dir(
        IMG_DIR,
        ADN_DIR,
        PROCESSED_DIR / "TCGA-LUAD_extracted_patterns_mpp_0.9",
        target_mpp=(0.9, 0.9),
        slide_level=0,
        save_description_json=True
    )

