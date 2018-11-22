import numpy as np
import glob
import os

save_folder = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/submittions/new_images_ansamble'

file_indexes_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/submittions/file_indexes.txt'
file_indexes = open(file_indexes_path, 'r').read()
indexes = file_indexes.split('\n')[:-1]
indexes = [index.split(':')[1] for index in indexes]

folder_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/submittions/ansamble'
pathes = glob.glob(folder_path + '/*/')


submites = list()
for path in pathes:
    score_path = path + '/scores.txt'
    f = open(score_path, 'r')
    table = f.read()

    table_split_line = table.split('\n')
    scores = list()
    for line in (table_split_line[:-3]):
        scores.append(float(line.split(':')[1]))
    submites.append(scores)

table = np.array(submites)

argmax_ind = np.argmax(table, axis=0)
print np.mean(np.max(table, axis=0))

for i, folder_ind in enumerate(argmax_ind):
    img_path = pathes[folder_ind] + 'imgs/' + indexes[i]
    dest_path = save_folder + '/' + indexes[i]
    os.system('cp ' + img_path + ' ' + dest_path)
    print i

