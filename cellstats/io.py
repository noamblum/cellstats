import os
import numpy as np
from cellpose import io as cpio
from lxml import etree

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def load_images(input_path: str):
    if os.path.isdir(input_path):
        image_names = [im for im in os.listdir(input_path) if any(im.endswith(e) for e in IMAGE_EXTENSIONS)]
        images = [cpio.imread(os.path.join(input_path, im)) for im in image_names]
        
    elif os.path.isfile(input_path):
        image_names = [input_path]
        images = [cpio.imread(input_path)]

    else:
        raise FileNotFoundError(f"Could not find {input_path}")

    return images, image_names


def load_metadata(input_path: str, metadata_suffix: str):
    if os.path.isdir(input_path):
        metadata = [etree.parse(f) for f in os.listdir(input_path) if f.endswith(metadata_suffix)]
        
    elif os.path.isfile(input_path) and input_path.endswith(".xml"):
        metadata = [etree.parse(input_path)]

    elif os.path.isfile(input_path) and not input_path.endswith(".xml"):
        metadata = [etree.parse(input_path[:input_path.rfind("_")] + metadata_suffix)]
    
    return metadata


def load_image_scales(input_path: str):
    metadata = load_metadata(input_path, ".tif_metadata.xml")
    return np.array([float(m.xpath("//Scaling/Items")[0][0].find("Value").text) for m in metadata])
