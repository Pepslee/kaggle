import cv2
import numpy as np
import glob

from shutil import copyfile
import os




path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/stage1_train/*/'

save_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet'

pathes = glob.glob(path)


for folder_path in pathes:
    images_path = glob.glob(folder_path + '/images/*')[0]
    new_image_path = save_path + '/imgs/' + os.path.basename(images_path)
    new_mask_path = save_path + '/masks/' + os.path.basename(images_path)
    copyfile(images_path, new_image_path)
    masks_pathes = glob.glob(folder_path + '/masks/*')
    end_mask = np.zeros(cv2.imread(images_path).shape[:-1], np.bool)
    print os.path.basename(images_path)
    for mask_path in masks_pathes:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.bool)
        end_mask = np.bitwise_or(end_mask, mask)
    cv2.imwrite(new_mask_path, end_mask.astype(np.uint8))

