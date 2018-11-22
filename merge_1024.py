import cv2
import numpy as np
import os
import glob


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/1024*1024/'

mask_folder = path + 'masks/'
imgs_folder = path + 'imgs/'

names = os.listdir(path + 'imgs')


res_image_1 = np.concatenate((cv2.imread(imgs_folder + names[0]), cv2.imread(imgs_folder + names[1])), axis=1)
res_image_2 = np.concatenate((cv2.imread(imgs_folder + names[1]), cv2.imread(imgs_folder + names[0])), axis=1)

res_image = np.concatenate((res_image_1, res_image_2), axis=0)

res_mask_1 = np.concatenate((cv2.imread(mask_folder + names[0], cv2.IMREAD_GRAYSCALE), cv2.imread(mask_folder + names[1], cv2.IMREAD_GRAYSCALE)), axis=1)
res_mask_2 = np.concatenate((cv2.imread(mask_folder + names[1], cv2.IMREAD_GRAYSCALE), cv2.imread(mask_folder + names[0], cv2.IMREAD_GRAYSCALE)), axis=1)

res_mask = np.concatenate((res_mask_1, res_mask_2), axis=0)

cv2.imwrite("/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/img_1024_1024.tif", res_image)
cv2.imwrite("/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/mask_1024_1024.tif", res_mask)
