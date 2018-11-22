import cv2
import numpy as np
from scipy import ndimage
import glob
import os
from dsel import my_io


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/tf_semantic_segmentation/static_test/22_8.tif'
img, geo = my_io.load_image(path, 'BIP')

img = img * [6, 5, 10, 2]
img = cv2.resize(img.astype(np.float32), dsize=(512, 512))

save_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/tf_semantic_segmentation/static_test/22_512.tif'
my_io.save_np_using_gdal(save_path, img.astype(np.uint16), geo_info=geo)

