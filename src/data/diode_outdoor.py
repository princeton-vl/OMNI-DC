import os
import tarfile
from io import BytesIO

import numpy as np

from . import BaseDataset
from .diode_base import DIODE_Base

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

split_txt = '../data_json/marigold_data_split/diode/diode_val_outdoor_filename_list.txt'
from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

class DIODE_Outdoor(DIODE_Base):
    def __init__(self, args, mode):
        super(DIODE_Outdoor, self).__init__(args, mode, split_txt,
                                            edge_noise_filter_radius=1, edge_noise_filter_thres=0.15)