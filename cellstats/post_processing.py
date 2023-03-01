from skimage.measure import regionprops
import numpy as np
import pandas as pd
from typing import List, Union, Optional
import warnings

class FeatureExtractor:

    ALL_FEATURES = ["length", "width", "area", "perimeter", "centroid", "aspect_ratio"]


    def __init__(self, masks: Union[List[np.ndarray], np.ndarray], files: List[str] = None,
                                scales: Optional[np.ndarray] = None, unit=10e-6) -> None:
        
        if scales is None:
            warnings.warn("scales not set, extracted features will be in pixels,"\
                " are you sure this is what you want?")
            self.__scales = 1
        else:
            self.__scales = scales / unit
        
        if isinstance(files, list) or files is None:
            self.__files: List[str] = files
        else:
            self.__files: List[str] = [files]

        # Multiple images
        if isinstance(masks, list) or (isinstance(masks, np.ndarray) and len(masks.shape) == 3):
            self.__regions: List[regionprops] = [regionprops(mask) for mask in masks]
        # Single image
        elif isinstance(masks, np.ndarray) and len(masks.shape) == 2:
            self.__regions: List[regionprops] = [regionprops(masks)]
        else:
            if isinstance(masks, np.ndarray):
                error_message = "masks should be a 2-d array (single image) or a 3-d array (multiple images)."\
                    f"Got: {masks.shape}"
                raise ValueError(error_message)
            raise TypeError(f"Expected list or numpy array in masks. Got: {type(masks)}")

        
    def get_lengths(self) -> np.ndarray:
        lengths = np.array([np.array([cell.axis_major_length for cell in cells])
                                        for cells in self.__regions], dtype=object)
        return np.concatenate(lengths * self.__scales).ravel()


    def get_widths(self) -> np.ndarray:
        widths = np.array([np.array([cell.axis_minor_length for cell in cells])
                                        for cells in self.__regions], dtype=object)
        return np.concatenate(widths * self.__scales).ravel()

    
    def get_areas(self) -> np.ndarray:
        areas = np.array([np.array([cell.area for cell in cells])
                                        for cells in self.__regions], dtype=object)
        return np.concatenate(areas * (self.__scales ** 2)).ravel()


    def get_perimeters(self) -> np.ndarray:
        perimeters = np.array([np.array([cell.perimeter for cell in cells])
                                        for cells in self.__regions], dtype=object)
        return np.concatenate(perimeters * self.__scales).ravel()

    
    def get_centroids(self) -> np.ndarray:
        return np.array([np.array(cell.centroid) for cells in self.__regions for cell in cells])

    
    def get_aspect_ratios(self) -> np.ndarray:
        return self.get_lengths() / self.get_widths()

    
    def get_features(self, features: Optional[List[str]]) -> pd.DataFrame:
        if features is None: features = FeatureExtractor.ALL_FEATURES
        res = {}

        if self.__files is not None:
            res["source"] = [self.__files[i] for i, cells in enumerate(self.__regions) for _ in cells]
        
        if "length" in features:
            res["length"] = self.get_lengths()
        if "width" in features:
            res["width"] = self.get_widths()
        if "area" in features:
            res["area"] = self.get_areas()
        if "centroid" in features:
            centroids = self.get_centroids().T
            res["centroidX"] = centroids[0]
            res["centroidY"] = centroids[1]
        if "perimeter" in features:
            res["perimeter"] = self.get_perimeters()
        if "aspect_ratio" in features:
            res["aspect_ratio"] = self.get_aspect_ratios()
        
        return pd.DataFrame(res)
