import numpy as np
import cv2


mask_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/9893_mask.png'
img_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/9893_sat.jpg'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# mask = cv2.bitwise_not(mask)



mask_1 = np.zeros((mask.shape[0]+2, mask.shape[0]+2), np.uint8)

# mask_1[1:-1, 1:-1] = mask

new = cv2.floodFill(img, mask_1, (622,311), (255, 255, 255), (11, 12, 13), (10, 10, 10), cv2.FLOODFILL_MASK_ONLY)
cv2.imwrite('/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/9893_filled.jpg', cv2.bitwise_not(mask_1))

cv2.namedWindow("img", 0)
cv2.namedWindow("mask", 0)
cv2.namedWindow("new_mask", 0)

cv2.imshow('img', img)
cv2.imshow('mask', mask*255)
cv2.imshow('mask', mask_1*255)

cv2.waitKey()



