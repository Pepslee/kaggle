import glob
import os
import shutil

train_folder_imgs = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train_test/train/imgs'
train_folder_masks = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train_test/train/masks'

test_folder_imgs = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train_test/test/imgs'
test_folder_masks = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train_test/test/masks'

image_names = os.listdir('/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/Kostya_train_val_split/imgs')
# masks_names = os.listdir(test_folder_masks)


for img_name in image_names:
    mask_name = os.path.splitext(img_name)[0] + '_mask.tif'
    img_name = os.path.splitext(img_name)[0] + '_sat.tif'

    shutil.move(os.path.join(train_folder_imgs, img_name), os.path.join(test_folder_imgs, img_name))
    shutil.move(os.path.join(train_folder_masks, mask_name), os.path.join(test_folder_masks, mask_name))

