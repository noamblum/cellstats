import os
from typing import Optional
import shutil
import pathlib

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MODEL_DIR_ENV = os.environ.get("CELLSTATS_ENVIRONMENT_MODEL_REPOSITORY")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath('.cellstats', 'models')


if not os.path.exists(_MODEL_DIR_DEFAULT):
    os.makedirs(_MODEL_DIR_DEFAULT)

if _MODEL_DIR_ENV and not os.path.exists(_MODEL_DIR_ENV):
    os.makedirs(_MODEL_DIR_ENV)


def __get_model_dir(environment: bool):
    if environment and not _MODEL_DIR_ENV:
        raise RuntimeError("Tried to access environment-wide model repository but it was never initialized.\n"\
                            "To initialize, set the CELLSTATS_ENVIRONMENT_MODEL_REPOSITORY environment variable.")
    return pathlib.Path(_MODEL_DIR_ENV) if environment else _MODEL_DIR_DEFAULT


def predict_masks(input_path, model_path, use_gpu, channels, verbose):
    from cellpose import io
    from cellpose import models as cpmodels
    if not os.path.isfile(model_path):
        model_path_local = os.fspath(__get_model_dir(False).joinpath(model_path))
        model_path_env = os.fspath(__get_model_dir(True).joinpath(model_path))

        if not os.path.isfile(model_path_local) and not os.path.isfile(model_path_env):
            raise FileNotFoundError(f"Model {model_path} does not exist")

        # At this point one is true for sure. Local takes precedence
        if os.path.isfile(model_path_local):
            model_path = model_path_local
        else:
            model_path = model_path_env

    if os.path.isdir(input_path):
        input_images = [io.imread(os.path.join(input_path, im)) for im in os.listdir(input_path)
                                                                if any(im.endswith(e) for e in IMAGE_EXTENSIONS)]
    elif os.path.isfile(input_path):
        input_images = [io.imread(input_path)]

    if verbose:
        from cellpose.io import logger_setup
        logger_setup()

    model = cpmodels.CellposeModel(gpu=use_gpu, pretrained_model=model_path)

    masks, _, _ = model.eval(input_images, channels=channels)

    return masks


def add_model(model_path: str, name: Optional[str], overwrite: bool, environment: bool):
    target_file_name = os.path.split(model_path)[-1] if name is None else name
    target_path = os.fspath(__get_model_dir(environment).joinpath(target_file_name))
    if os.path.isfile(target_path) and not overwrite:
        raise FileExistsError(f"{target_file_name} already exists in model list. Use --overwtire flag to overwtire")
    try:
        shutil.copyfile(model_path, target_path)
    except shutil.SameFileError:
        pass
    

def rename_model(old_name: str, new_name: str, overwrite:bool, environment: bool):
    if old_name == new_name: return
    source_path = os.fspath(__get_model_dir(environment).joinpath(old_name))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Model not found: {old_name}")
    target_path = os.fspath(__get_model_dir(environment).joinpath(new_name))
    if os.path.isfile(target_path) and not overwrite:
        raise FileExistsError(f"{new_name} already exists in model list. Use --overwtire flag to overwtire")
    os.rename(source_path, target_path)


def remove_model(model_name, environment: bool):
    model_path = os.fspath(__get_model_dir(environment).joinpath(model_name))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_name}")
    
    os.remove(model_path)


def list_models():
    ret_val = {}
    ret_val["environment"] = []
    if _MODEL_DIR_ENV:
        env_dir = __get_model_dir(True)
        ret_val["environment"] = os.listdir(env_dir)
    local_dir = __get_model_dir(False)
    ret_val["local"] = os.listdir(local_dir)
    return ret_val


def environment_repository_initialized() -> bool:
    return bool(_MODEL_DIR_ENV)
