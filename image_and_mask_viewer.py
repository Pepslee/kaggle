import cv2
import glob
import os
import numpy as np
from shutil import copyfile

trainSet = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/'
imgs_folder = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/imgs/'
masks_folder = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/masks/'

imgs_pathes = os.listdir(imgs_folder)

def border_zero(img):
    mask = np.copy(img)
    mask[1:-1, 1:-1] = 0
    if np.sum(mask) >= 1:
        return False
    else:
        return True

count = 0
for img_path in imgs_pathes:
    img = cv2.imread(imgs_folder + img_path)
    mask = cv2.imread(masks_folder + img_path, cv2.IMREAD_GRAYSCALE)
    if border_zero(mask):
        count += 1
        # if mask.shape == (256, 256):
        #     new_mask_path = trainSet + '/256*256/masks/' + img_path
        #     copyfile(masks_folder + img_path, new_mask_path)
        #     new_img_path = trainSet + '/256*256/imgs/' + img_path
        #     copyfile(imgs_folder + img_path, new_img_path)
        # if mask.shape == (1024, 1024):
        #     new_mask_path = trainSet + '/1024*1024/masks/' + img_path
        #     copyfile(masks_folder + img_path, new_mask_path)
        #     new_img_path = trainSet + '/1024*1024/imgs/' + img_path
        #     copyfile(imgs_folder + img_path, new_img_path)
        # if mask.shape == (603, 1272):
        #     new_mask_path = trainSet + '/603*1272/masks/' + img_path
        #     copyfile(masks_folder + img_path, new_mask_path)
        #     new_img_path = trainSet + '/603*1272/imgs/' + img_path
        #     copyfile(imgs_folder + img_path, new_img_path)
        # cv2.namedWindow("img", 0)
        # cv2.namedWindow("mask", 0)
        # cv2.imshow('img', img)
        # cv2.imshow('mask', mask*255)
        print img.shape
        # cv2.waitKey(0)

print count