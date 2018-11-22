import glob
import os
from dsel import my_io
import numpy as np
import cv2

path = '/home/panchenko/server_20/4Tb2/DeepGlobe/Roads/valid_prob'

thresh = 1.0

save_folder = '/home/panchenko/server_20/4Tb2/DeepGlobe/Roads/valid_72p_thresh_' + str(thresh)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

pathes = glob.glob(path + '/*.*')
for i, mask_path in enumerate(pathes):
    prob, _ = my_io.load_image(mask_path)
    prob_1 = np.nan_to_num(prob[1])
    img = (prob_1 >= thresh).astype(np.uint8)*255
    save_name = save_folder + '/' + os.path.splitext(os.path.basename(mask_path))[0] + '.png'
    print i, ' / ', len(pathes), ' ', mask_path, ' ', save_name
    cv2.imwrite(save_name, np.stack([img, img, img], axis=-1))
