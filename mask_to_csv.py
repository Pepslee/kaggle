import cv2
import numpy as np


def mask_to_str(mask):
    flatten = mask.flatten(order='F')
    flatten_rle = rle(flatten)
    values = flatten_rle[2]
    mask_pixels = values != 0
    starts = flatten_rle[1][mask_pixels]
    lenghts = flatten_rle[0][mask_pixels]
    mask_in_text = []
    for i in range(len(starts)):
        mask_in_text.append(str(starts[i] + 1))
        # Case for last values
        if starts[i] + 1 + lenghts[i] > flatten.size:
            mask_in_text.append(str(lenghts[i] - 1))
        else:
            mask_in_text.append(str(lenghts[i]))
    result = ' '.join(mask_in_text)
    return result


def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray)  #force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/stage1_train/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/masks/0e548d0af63ab451616f082eb56bde13eb71f73dfda92a03fbe88ad42ebb4881.png'
mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# mask = np.transpose(mask)
# mask = mask.flatten()
# sub_nonzero = np.nonzero(mask)[0]
# sub_revers = 65536 - np.nonzero(mask[::-1])[0][::-1]

sub = mask_to_str(mask)
pass
