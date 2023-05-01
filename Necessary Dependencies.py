pip install torch==1.9.0
pip install segmentation-models-pytorch

LIBRARAIES:
# importing required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import glob
import gc
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


%matplotlib inline
from IPython.display import Image, display
from skimage import io

from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import normalize

from PIL import Image

import torchvision
from torchvision import transforms