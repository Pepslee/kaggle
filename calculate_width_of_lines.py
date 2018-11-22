import cv2
import numpy as np
import os
import glob
import scipy.ndimage.morphology as morf
path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train/masks'


imgs = glob.glob(path + '/*')


road_pixels_summ = list()
for i, name in enumerate(imgs):
    # os.rename(name, path + '/' + str(i) + '_' + os.path.basename(name))
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)





