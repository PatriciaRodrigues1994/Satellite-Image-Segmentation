from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import dataset.transform_and_augment as trans_aug
import os
from multiprocessing import cpu_count

from dataset.dataset import TrainImageDataset, TestImageDataset
import multiprocessing

# change the working dirctory
project_dir = "/home/webwerks/my-projects/github/Segmentation"
os.chdir(project_dir)

data_directory = "data/"
annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json"
TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json"

VAL_IMAGES_DIRECTORY = "data/val/images"
VAL_ANNOTATIONS_PATH = "data/val/annotation.json"
VAL_ANNOTATIONS_SMALL_PATH = "data/val/annotation-small.json"


train_coco = COCO(os.path.join(TRAIN_ANNOTATIONS_PATH))
X_train = train_coco.getImgIds(catIds=train_coco.getCatIds())


input_img_resize = (300, 300)  # The resize size of the input images of the neural net
output_img_resize = (300, 300)  # The resize size of the output images of the neural net

batch_size = 3
epochs = 50
threshold = 0.5
validation_size = 0.2
sample_size = None  # Put 'None' to work on full dataset or a value between 0 and 1

# -- Optional parameters
threads = cpu_count()
use_cuda = torch.cuda.is_available()


train_ds = TrainImageDataset(X_data = X_train, cocodataset = train_coco, y_data = None, input_img_resize = input_img_resize, 
                             output_img_resize = output_img_resize, X_transform=trans_aug.augment_img)


import pdb; pdb.set_trace()