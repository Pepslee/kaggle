import numpy as np
import zipfile

path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/submittions/images/Untitled Folder/Pepslee-15-output.zip'


with zipfile.ZipFile(path, 'r') as myzip:
    if len(myzip.namelist()) != 3:
        print 'Wrong zip archive'
    table = myzip.read('scores.txt')
    table_split_line = table.split('\n')
    scores = list()
    for line in(table_split_line[:-2]):
        scores.append(float(line.split(':')[1]))

    print 'mean = ', np.mean(np.array(scores))