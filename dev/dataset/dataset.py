import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import dataset.transform_and_augment as transformer
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import skimage.io as io
import matplotlib.pyplot as plt
import glob


# Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L66
class TrainImageDataset(data.Dataset):
    def __init__(self, img_dir = None, cocodataset = None, y_data=None, input_img_resize=(128, 128), output_img_resize=(128, 128), X_transform=None, y_transform=None, threshold=0.5):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                threshold (float): The threshold used to consider the mask present or not
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                input_img_resize (tuple): Tuple containing the new size of the input images
                output_img_resize (tuple): Tuple containing the new size of the output images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                y_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
        """
        self.threshold = threshold
        self.y_train_masks = y_data
        self.input_img_resize = input_img_resize
        self.output_img_resize = output_img_resize
        self.y_transform = y_transform
        self.X_transform = X_transform
        self.coco = cocodataset
        self.X_train = self.coco.getImgIds(catIds=self.coco.getCatIds())
        self.IMAGES_DIRECTORY = img_dir
        
        
    def get_mask(self, img_index):
        annotation_ids = self.coco.getAnnIds(imgIds=[img_index])
        annotations = self.coco.loadAnns(annotation_ids)
        mask = self.coco.annToMask(annotations[0])
        for i in range(len(annotations)):
            mask = mask | self.coco.annToMask(annotations[i])
        return mask
    
    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is class_index of the target class.
        """
        index  = self.X_train[index]
        img = self.coco.loadImgs(index)[0]
        image_path = os.path.join(self.IMAGES_DIRECTORY, img["file_name"])
        img = Image.open(image_path)
        img = img.resize(self.input_img_resize, Image.ANTIALIAS)
        img = np.asarray(img.convert("RGB"), dtype=np.float32)
        # Pillow reads gifs
        mask = self.get_mask(index)

        if self.X_transform:
            img, mask = self.X_transform(img, mask)

        if self.y_transform:
            img, mask = self.y_transform(img, mask)
        
        img = transformer.image_to_tensor(img)
        mask = transformer.mask_to_tensor(mask, self.threshold)
        return img, mask

    def __len__(self):
        return len(self.X_train)


class TestImageDataset(data.Dataset):
    def __init__(self, X_data, img_dir = None, img_resize=(128, 128)):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                X_data (list): List of paths to the training images
                img_resize (tuple): Tuple containing the new size of the images
        """
        self.img_resize = img_resize
        self.X_train = X_data
        self.X_train = glob.glob(img_dir + "/*.jpg")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path = self.X_train[index]
        img = Image.open(img_path)
        img = img.resize(self.img_resize, Image.ANTIALIAS)
        img = np.asarray(img.convert("RGB"), dtype=np.float32)

        img = transformer.image_to_tensor(img)
        return img, img_path.split("/")[-1]

    def __len__(self):
        return len(self.X_train)