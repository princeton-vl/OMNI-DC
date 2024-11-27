import os
import tarfile
from io import BytesIO

import numpy as np

from . import BaseDataset
from .eth3d_base import ETH3D_Base, indoor_val_split, indoor_test_split

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ETH3D_Outdoor(ETH3D_Base):
    def __init__(self, args, mode):
        super(ETH3D_Outdoor, self).__init__(args, mode, scene='outdoor')
