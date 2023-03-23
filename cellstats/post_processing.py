from skimage.measure import regionprops
import numpy as np
import pandas as pd
from typing import List, Union, Optional
import warnings
from cellstats import io
from skimage import color

class FeatureExtractor:

    ALL_GEOMETRICAL_FEATURES = ["length", "width", "area", "perimeter", "centroid", "aspect_ratio"]


    def __init__(self, masks: Union[List[np.ndarray], np.ndarray], files: List[str] = None,
                                scales: Optional[np.ndarray] = None, unit=1e-6) -> None:
        
        if scales is None or len(scales) == 0:
            warnings.warn("scales not set, extracted features will be in pixels,"\
                " are you sure this is what you want?")
            self.__scales = 1
        else:
            self.__scales = scales / unit
        
        if isinstance(files, list) or files is None:
            self.__files: List[str] = files
        else:
            self.__files: List[str] = [files]
        
        self.__regions: List[regionprops] = []
        # Multiple images
        if isinstance(masks, list) or (isinstance(masks, np.ndarray) and len(masks.shape) == 3):
            for i in range(len(masks)):
                img = io.load_image(files[i], 0, rgb=True, czi_all_channels=True) if files is not None else None
                self.__regions.append(regionprops(masks[i], img))
        # Single image
        elif isinstance(masks, np.ndarray) and len(masks.shape) == 2:
            img = io.load_image(files[0], 0, rgb=True, czi_all_channels=True) if files is not None else None
            self.__regions.append(regionprops(masks, img))
        
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
    

    def __parse_channel(self, channel):
        if isinstance(channel, str):
            if self.__files is None or any(not f.endswith(".czi") for f in self.__files):
                raise ValueError(f"String channels are only supported for .czi files. Got {self.__files}")
            else:
                channel_per_file = [io.get_czi_channel_index(f, channel) for f in self.__files]
                channel = [channel_per_file[i] for i, cells in enumerate(self.__regions) for _ in cells]
        return channel


    def get_centers_of_mass(self, channel) -> np.ndarray:
        data = np.array([np.array(cell.weighted_centroid) for cells in self.__regions for cell in cells])
        return data[np.arange(len(data)),:,self.__parse_channel(channel)]
    

    def get_delta_com_centroids(self, channel) -> np.ndarray:
        centroids = self.get_centroids()
        coms = self.get_centers_of_mass(channel)

        distances = np.sqrt(((centroids[:,0] - coms[:,0]) ** 2) + ((centroids[:,1] - coms[:,1]) ** 2))
        scales_vec = np.array([self.__scales[i] for i, cells in enumerate(self.__regions) for _ in cells])\
            if isinstance(self.__scales, np.ndarray) else self.__scales
        
        return distances * scales_vec


    
    def get_geometrical_features(self, features: Optional[List[str]]) -> pd.DataFrame:
        if features is None: features = FeatureExtractor.ALL_GEOMETRICAL_FEATURES
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
    

    def get_intensity_features(self, channel, features: Optional[List[str]]) -> pd.DataFrame:
        res = {}
        res["source"] = [self.__files[i] for i, cells in enumerate(self.__regions) for _ in cells]
        coms = self.get_centers_of_mass(channel).T
        res["center_of_mass_X"] = coms[0]
        res["center_of_mass_Y"] = coms[1]
        res["d_com_centroid"] = self.get_delta_com_centroids(channel)
        return pd.DataFrame(res)

