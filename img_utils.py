#!usr/bin/env python2
#!coding=utf-8
import cv2
import numpy as np



def load_img(path, color_mode='rgb', target_size=None, crop_size=None):
    """
    Load an image.
    
    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.
        crop_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.
        
    # Returns
        Image as numpy array.
    """

    img = cv2.imread(path)
    if color_mode == 'grayscale':
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color_mode == 'rgbe':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        canny = cv2.Canny(gray_img, 50, 150)
        img = cv2.merge([img, canny])

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)

    if crop_size:
        img = central_image_crop(img, crop_size[0], crop_size[1])

    if color_mode == 'grayscale':
        # 转化成带通道数的
        img = img.reshape((img.shape[0], img.shape[1], 1))

    return np.asarray(img, dtype=np.float32)



def central_image_crop(img, crop_width=150, crop_heigth=150):
    """
    Crop the input image centered in width and starting from the bottom
    in height.
    
    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.
        
    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_heigth: img.shape[0],
              half_the_width - int(crop_width / 2):
              half_the_width + int(crop_width / 2)]
    return img
