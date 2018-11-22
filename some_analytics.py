import cv2
import numpy as np
import os
import glob

path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train/masks'

statistic_file_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/statistic_file'

f = open(statistic_file_path, 'w')

imgs = glob.glob(path + '/*')


road_pixels_summ = list()
for i, name in enumerate(imgs):
    # os.rename(name, path + '/' + str(i) + '_' + os.path.basename(name))
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    summ = np.sum(img)
    road_pixels_summ.append(summ)
    print i, ' / ', len(imgs)

np_sum_list = np.stack(road_pixels_summ)

print np.min(np_sum_list), np.max(np_sum_list), np.mean(np_sum_list), np.median(np_sum_list)


size = 1024*1024
f.writelines([' min = ', str(size), '\n'])
f.writelines([' min = ', str(np.min(np_sum_list)/size), '\n'])
f.writelines([' max = ', str(np.max(np_sum_list)/size), '\n'])
f.writelines([' mean = ', str(np.mean(np_sum_list)/size), '\n'])
f.writelines([' median = ', str(np.median(np_sum_list)/size), '\n'])
print np.argmin(np_sum_list)
print np.argmax(np_sum_list)



