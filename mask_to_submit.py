import cv2
import numpy as np
import glob
import os
import mask_to_csv


def trifle_removal(binary_mask, min_area=10, min_area_black=20):
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
        if stats[label][cv2.CC_STAT_AREA] <= min_area:
            slice_l = (slice(stats[label][cv2.CC_STAT_TOP], stats[label][cv2.CC_STAT_TOP] + stats[label][cv2.CC_STAT_HEIGHT]), slice(stats[label][cv2.CC_STAT_LEFT], stats[label][cv2.CC_STAT_LEFT] + stats[label][cv2.CC_STAT_WIDTH]))
            new_mask[slice_l][labels[slice_l] == label] = 0
    new_mask_not = np.bitwise_not(new_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_mask_not, connectivity=8)
    for label in xrange(1, num_labels):
        if stats[label][cv2.CC_STAT_AREA] <= min_area_black:
            slice_l = (slice(stats[label][cv2.CC_STAT_TOP], stats[label][cv2.CC_STAT_TOP] + stats[label][cv2.CC_STAT_HEIGHT]), slice(stats[label][cv2.CC_STAT_LEFT], stats[label][cv2.CC_STAT_LEFT] + stats[label][cv2.CC_STAT_WIDTH]))
            new_mask[slice_l][labels[slice_l] == label] = 255
    return new_mask


folder_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/res'
image_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/test_dataset_red'

pathes = glob.glob(folder_path + '/*')




sub = open('/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/submission_14.csv', 'w')
sub.write('ImageId,EncodedPixels\n')
for path in pathes:
    img_path = image_path + '/' + os.path.basename(path)[:os.path.basename(path).find('_mask')] + '.png'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    real_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    hist = cv2.calcHist([real_image], [0], None, [256], [0, 256])
    area = 0
    if np.argmax(hist) > 128:
        area = 100
    else:
        area = 10
    mask_fuul = trifle_removal(image, area, 50)
    print np.argmax(hist)
    cv2.namedWindow("image", 0)
    cv2.namedWindow("mask_fuul", 0)
    cv2.imshow("image", image*255)
    cv2.namedWindow("real_image", 0)
    cv2.imshow("real_image", real_image)
    cv2.imshow("mask_fuul", mask_fuul)
    cv2.waitKey()

    # num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fuul)
    # for l in xrange(1, num):
    #     if stats[l][cv2.CC_STAT_AREA] > area:
    #         mask = (labels == l).astype(np.uint8)
    #         name = os.path.basename(path)
    #         sub.write(name[:name.find('_mask_')] + ',' + mask_to_csv.mask_to_str(mask) + '\n')



