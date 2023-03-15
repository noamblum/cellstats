import os
import numpy as np
from cellpose import io as cpio
import xml.etree.ElementTree as ET
from aicsimageio import AICSImage
from PIL import Image
from cellpose.utils import masks_to_outlines
from matplotlib import colors
from skimage import color


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def load_images(input_path: str, channel=0, rgb=False):
    if os.path.isdir(input_path):
        image_names = []
        images = []
        for im in os.listdir(input_path):
            if any(im.endswith(e) for e in IMAGE_EXTENSIONS) or im.endswith(".czi"):
                images.append(__load_image(os.path.join(input_path, im), channel, rgb))
                image_names.append(im)

    elif os.path.isfile(input_path):
        image_names = [input_path]
        images = [__load_image(input_path, channel, rgb)]

    else:
        raise FileNotFoundError(f"Could not find {input_path}")

    return images, image_names


def load_metadata(input_path: str, metadata_suffix: str):
    metadata=[]
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.endswith(metadata_suffix):
                metadata.append(ET.parse(f))
            elif f.endswith(".czi"):
                metadata.append(AICSImage(f).metadata)
    
    elif os.path.isfile(input_path) and input_path.endswith(".czi"):
        metadata = [AICSImage(input_path).metadata]

    elif os.path.isfile(input_path) and input_path.endswith(".xml"):
        metadata = [ET.parse(input_path)]

    elif os.path.isfile(input_path) and not input_path.endswith(".xml"):
        metadata = [ET.parse(input_path[:input_path.rfind("_")] + metadata_suffix)]
    
    return metadata


def load_image_scales(input_path: str):
    metadata = load_metadata(input_path, ".tif_metadata.xml")
    return np.array([float(m.find(".//Scaling/Items/Distance/Value").text) for m in metadata])


def save_image_outlines(input_path: str, output_path: str, masks: np.ndarray, channel=0, color="r", verbose=False):
    if verbose:
        print(f"Saving masks to {output_path}. This might take a while...")
    input_images, input_image_names = load_images(input_path, rgb=True)
    metadata = load_metadata(input_path, ".tif_metadata.xml")
    for i in range(len(input_images)):
        mask = masks[i]
        image_name, image_ext = os.path.splitext(os.path.basename(input_image_names[i]))
        if image_ext == ".czi":
            channels_info = metadata[i].find(".//Information/Image/Dimensions/Channels")
            if isinstance(channel, str):
                c = channels_info.find(f'.//Channel[@Name="{channel}"]/Color').text
            elif isinstance(channel, int):
                c = channels_info.findall('.//Channel/Color')[channel].text
            else:
                c = channels_info.find('.//Channel/Color').text
            
            c = np.array(colors.to_rgb(f"#{c[3:]}")).reshape((3,1,1))
            image = input_images[i].copy()
            im_max = np.max(image)
            im_min = np.min(image)
            image = (image - im_min) / (im_max - im_min)
            image = (image * c).transpose(1,2,0)
            image = (image * 255).astype(np.uint8)
        elif image_ext in IMAGE_EXTENSIONS:
            image = input_images[i].copy()

        outline = masks_to_outlines(mask)
        image[outline] = (np.array(colors.to_rgb(color)) * 255).astype(np.uint8)

        Image.fromarray(image).save(f"{os.path.join(output_path, image_name)}_outlines.png")
        if verbose:
            print(f"Saved outline for {image_name}")


def __load_image(img_path: str, channel: int, rgb=False):
    if any(img_path.endswith(e) for e in IMAGE_EXTENSIONS):
        img = cpio.imread(img_path)
        if rgb:
            return img
        if channel == 0 or channel is None:
            return color.rgb2gray(img)
        return img[:,:,channel - 1]
    if img_path.endswith(".czi"):
        img = AICSImage(img_path)
        metadata = load_metadata(img_path, ".tif_metadata.xml")
        channels_info = metadata[0].find(".//Information/Image/Dimensions/Channels")
        if isinstance(channel, str):
            c = channels_info.find(f'.//Channel[@Name="{channel}"]')
            channels_info = list(channels_info)
            if c not in channels_info:
                raise ValueError(f"{channel} is not a channel in {img_path}")
            channel = channels_info.index(c)
        return img.get_image_data("YX", c=channel)
