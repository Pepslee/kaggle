import os
import glob
import shutil


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/stage1_test'
save_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/test_dataset'

folders = os.listdir(path)

for name in folders:
    shutil.copyfile(path + '/' + name + '/images/' + name + '.png', save_path + '/' + name + '.png')




