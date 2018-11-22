import cv2
import numpy as np
import os
import glob
from random import shuffle


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/256*256/'

mask_folder = path + 'masks/'
imgs_folder = path + 'imgs/'

names = os.listdir(path + 'imgs')

pathes_img = glob.glob(imgs_folder + '*')
pathes_mask = glob.glob(mask_folder + '*')

img_list = list()
mask_list = list()


for path_img in pathes_img:
    img_list.append(cv2.imread(path_img))

for path_mask in pathes_mask:
    mask_list.append(cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE))


vertical_img_list = list()
vertical_mask_list = list()
for i in range(len(img_list)):
    vertical_img_list.append(np.concatenate(img_list, axis=1))
    vertical_mask_list.append(np.concatenate(mask_list, axis=1))
    c = list(zip(img_list, mask_list))
    shuffle(c)
    img_list, mask_list = zip(*c)

res_img = np.concatenate(vertical_img_list, axis=0)
res_mask = np.concatenate(vertical_mask_list, axis=0)




cv2.imwrite("/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/img_256_256.tif", res_img)
cv2.imwrite("/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/TrainSet/mask_256_256.tif", res_mask)
