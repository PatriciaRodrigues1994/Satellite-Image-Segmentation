import cv2
import torch
import random
import numpy as np
from skimage import io, transform

def image_to_tensor(image, mean=0, std=1.):
    """
    Transforms an image to a tensor
    Args:
        image (np.ndarray): A RGB array image
        mean: The mean of the image values
        std: The standard deviation of the image values
    Returns:
        tensor: A Pytorch tensor
        Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225] # needs C, H, W
    # imgae = F.normalize(image, mean ,std , False)
    image = image.astype(np.float32)
    # image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


def mask_to_tensor(mask, threshold):
    """
    Transforms a mask to a tensor
    Args:
        mask (np.ndarray): A greyscale mask array
        threshold: The threshold used to consider the mask present or not
    Returns:
        tensor: A Pytorch tensor
    """
    mask = mask
    mask = (mask > threshold).astype(np.float32)
    # mask = transform.resize(mask, input_size)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor


# def random_shift_scale_rotate(image, angle, scale, aspect, shift_dx, shift_dy,
#                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
#     if np.random.random() < u:
#         if len(image.shape) == 3:  # Img or mask
#             height, width, channels = image.shape
#         else:
#             height, width = image.shape

#         sx = scale * aspect / (aspect ** 0.5)
#         sy = scale / (aspect ** 0.5)
#         dx = round(shift_dx * width)
#         dy = round(shift_dy * height)

#         cc = np.math.cos(angle / 180 * np.math.pi) * sx
#         ss = np.math.sin(angle / 180 * np.math.pi) * sy
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])

#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0, box1)

#         image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
#                                     borderMode=borderMode, borderValue=(0, 0, 0, 0))
#     return image


# def random_horizontal_flip(image, mask, u=0.5):
#     if np.random.random() < u:
#         image = cv2.flip(image, 1)
#         mask = cv2.flip(mask, 1)

#     return image, mask


def augmentation(image, label):

    h, w, _ = image.shape
    if random.random() > 0.5:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

    # # Padding to return the correct crop size
    # if random.random() > 0.5:
    #     pad_h = max(self.crop_size - h, 0)
    #     pad_w = max(self.crop_size - w, 0)
    #     pad_kwargs = {
    #         "top": 0,
    #         "bottom": pad_h,
    #         "left": 0,
    #         "right": pad_w,
    #         "borderType": cv2.BORDER_CONSTANT,}
    #     if pad_h > 0 or pad_w > 0:
    #         image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
    #         label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
        
    #     # Cropping 
    #     h, w, _ = image.shape
    #     start_h = random.randint(0, h - self.crop_size)
    #     start_w = random.randint(0, w - self.crop_size)
    #     end_h = start_h + self.crop_size
    #     end_w = start_w + self.crop_size
    #     image = image[start_h:end_h, start_w:end_w]
    #     label = label[start_h:end_h, start_w:end_w]

        # Random H flip
    if random.random() > 0.5:
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()

    return image, label

def augment_img(img, mask):
    # rotate_limit = (-45, 45)
    # aspect_limit = (0, 0)
    # scale_limit = (-0.1, 0.1)
    # shift_limit = (-0.0625, 0.0625)
    # shift_dx = np.random.uniform(shift_limit[0], shift_limit[1])
    # shift_dy = np.random.uniform(shift_limit[0], shift_limit[1])
    # angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    # scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    # aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
    
    # img = random_shift_scale_rotate(img, angle, scale, aspect, shift_dx, shift_dy)
    # mask = random_shift_scale_rotate(mask, angle, scale, aspect, shift_dx, shift_dy)

    img, mask = augmentation(img, mask)

    return img, mask