from cellpose import io
from cellpose import models as cpmodels
import os
from typing import Optional
import shutil
import pathlib

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MODEL_DIR_ENV = os.environ.get("CELLSTATS_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath('.cellstats', 'models')
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)



def predict_masks(input_path, model_path, use_gpu, channels):
    if not os.path.isfile(model_path):
        model_path_copy = model_path
        model_path = os.fspath(MODEL_DIR.joinpath(model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model {model_path_copy} does not exist")

    if os.path.isdir(input_path):
        input_images = [io.imread(os.path.join(input_path, im)) for im in os.listdir(input_path)
                                                                if any(im.endswith(e) for e in IMAGE_EXTENSIONS)]
    elif os.path.isfile(input_path):
        input_images = [io.imread(input_path)]

    model = cpmodels.CellposeModel(gpu=use_gpu, pretrained_model=model_path)

    masks, _, _ = model.eval(input_images, channels=channels)

    return masks


def add_model(model_path: str, name: Optional[str], overwrite: bool):
    target_file_name = os.path.split(model_path)[-1] if name is None else name
    target_path = os.fspath(MODEL_DIR.joinpath(target_file_name))
    if os.path.isfile(target_path) and not overwrite:
        raise FileExistsError(f"{target_file_name} already exists in model list. Use --overwtire flag to overwtire")
    try:
        shutil.copyfile(model_path, target_path)
    except shutil.SameFileError:
        pass
    

def rename_model(old_name: str, new_name: str, overwrite:bool):
    if old_name == new_name: return
    source_path = os.fspath(MODEL_DIR.joinpath(old_name))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Model not found: {old_name}")
    target_path = os.fspath(MODEL_DIR.joinpath(new_name))
    if os.path.isfile(target_path) and not overwrite:
        raise FileExistsError(f"{new_name} already exists in model list. Use --overwtire flag to overwtire")
    os.rename(source_path, target_path)


def remove_model(model_name):
    model_path = os.fspath(MODEL_DIR.joinpath(model_name))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_name}")
    
    os.remove(model_path)

