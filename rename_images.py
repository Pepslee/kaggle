import cv2
import numpy as np
import os
import glob
from dsel import my_io


def trifle_removal(binary_mask, min_area=30, min_area_black=30):
    """
    :param binary_mask: numpy array 2d, 0 - background, not 0 - contours
    :param min_area_contours: minimum area for the contour
    :param min_area_polygons: minimum area for the polygons
    :return: numpy array without trifle with the same maximum value
    """
    if len(binary_mask.shape) != 2:
        raise RuntimeError('Only single-channel images!')

    max_elem = np.max(binary_mask)
    binary_mask = cv2.normalize(binary_mask, None, 0, 255, cv2.NORM_MINMAX)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    new_mask = binary_mask.copy()
    for label in xrange(1, num_labels):
        print num_labels, '/', label
        if stats[label][cv2.CC_STAT_AREA] <= min_area:
            slice_l = (slice(stats[label][cv2.CC_STAT_TOP], stats[label][cv2.CC_STAT_TOP] + stats[label][cv2.CC_STAT_HEIGHT]), slice(stats[label][cv2.CC_STAT_LEFT], stats[label][cv2.CC_STAT_LEFT] + stats[label][cv2.CC_STAT_WIDTH]))
            new_mask[slice_l][labels[slice_l] == label] = 0
    new_mask_not = np.bitwise_not(new_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_mask_not, connectivity=8)
    for label in xrange(1, num_labels):
        print num_labels, '/', label
        if stats[label][cv2.CC_STAT_AREA] <= min_area_black:
            slice_l = (slice(stats[label][cv2.CC_STAT_TOP], stats[label][cv2.CC_STAT_TOP] + stats[label][cv2.CC_STAT_HEIGHT]), slice(stats[label][cv2.CC_STAT_LEFT], stats[label][cv2.CC_STAT_LEFT] + stats[label][cv2.CC_STAT_WIDTH]))
            new_mask[slice_l][labels[slice_l] == label] = 255
    return new_mask


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/valid_bin_932400_morph'

imgs = glob.glob(path + '/*')


for i, name in enumerate(imgs):
    # os.rename(name, path + '/' + str(i) + '_' + os.path.basename(name))
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    save_name = name[:name.find('_sat')] + '_mask.tif'
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=3)

    img = trifle_removal(img, min_area=1000, min_area_black=30)

    print i
    cv2.imwrite(name, np.stack([img, img, img], axis=-1))
    # my_io.save_np_using_gdal(name, img)

