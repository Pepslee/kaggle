import cv2
import numpy as np
import glob
import os

path_folder_imgs = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/neural_network/test/kaggle/test_dataset_red_inv'

# path_folder_masks = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/neural_network/TrainSet/togen/masks'

pathes = os.listdir(path_folder_imgs)

for i, name in enumerate(pathes):
    # path_mask = path_folder_masks + '/' + name
    # path_inv_mask = path_folder_masks + '/' + str(1000 + i) + '_' + name
    # mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(path_inv_mask, mask)

    path_img = path_folder_imgs + '/' + name
    # path_inv_image = path_folder_imgs + '/' + str(1000 + i) + '_' + name
    image = cv2.imread(path_img, cv2.IMREAD_COLOR)[:, :, -1]

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    if np.argmax(hist) > 128:
        image_inv = cv2.bitwise_not(image)
        cv2.imwrite(path_img, image_inv)
    print i
