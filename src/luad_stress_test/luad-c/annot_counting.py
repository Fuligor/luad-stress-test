import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def count_annotations_in_file(file_path: Path, xml_element: str="Region", annot_attr: str="Text") -> dict:
    """Function returns counts of each annotation type in xml file given as Path.

    Args:
        file_path (Path): filepath for xml file fi the slide that need counting
        xml_element (str, optional): the name of the xml element that signifies one annotation. Defaults to "Region".
        annot_attr (str, optional): the name of xml attibute that carries the label value. Defaults to "Text".

    Returns:
        dict: number of each label type avaliable in given slide
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    counts = {"filename": file_path.name}
    attrs = list(root.findall(f".//{xml_element}"))
    counts[xml_element] = len(attrs)
    for annotation in attrs:
        part_of_group = annotation.attrib.get(annot_attr, "Nieznane")

        if part_of_group not in counts:
            counts[part_of_group] = 1
        else:
            counts[part_of_group] += 1

    return counts

def count_annotations_in_dir(dir_path, xml_element="Region", annot_attr="Text", return_sum=False):

    xml_files = [file for file in dir_path.iterdir() if file.suffix == ".xml"]

    counts = []
    annot_set = set()

    for file in xml_files:
        row = count_annotations_in_file(file, xml_element, annot_attr)
        annot_set.update(row.keys())
        counts.append(row)

    for c in counts:
        for annot_name in annot_set:
            if annot_name not in c:
                c[annot_name] = 0

    df = pd.DataFrame(counts)
    if return_sum:
        df.drop(columns=["filename"], inplace=True)
        return df.sum().reset_index()
    
    return df
        


if __name__ == "__main__":

    annot_path = Path("/home/malgorzata.sokol/Documents/Projekty/StressTest/Dane/Adnotacje/TCGA-LUAD-patterns_v2")

    df = count_annotations_in_dir(annot_path, return_sum=True, attr_name="Annotation", annot_attr="PartOfGroup")

    df.to_csv("annot_counts.csv", index=False)

    print(df)


