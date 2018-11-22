import os
import glob
import cv2
import numpy as np


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/two_submittion'
folders = os.listdir(path)

listdir = os.listdir(path + '/' + folders[0])

for_zero = cv2.imread(path + '/' + folders[0] + '/' + listdir[0])

for name in listdir:
    img_zero = np.zeros_like(for_zero)
    save_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/merged' + '/' + name
    for folder in folders:
        img_path = path + '/' + folder + '/' + name
        img = cv2.imread(img_path)
        img_zero = np.bitwise_or(img_zero, img)
    cv2.imwrite(save_path, img_zero)
