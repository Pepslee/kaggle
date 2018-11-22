import cv2
import numpy as np
import os

DEBUG = True


def debug_print(index, comment, percent, link):
    if DEBUG:
        print('%-4d     %-13s     %-8f     %s' % (index, comment, percent, link))


def get_percent(content, line):
    for cur_line in content:
        if line in cur_line:
            data = cur_line.split(':')
            return float(data[1].rstrip('\n'))


def get_image_list(path):
    fs = sorted(os.listdir(path))
    images_path = []
    for fn in fs:
        if '_mask' in fn:
            temp_path = os.path.join(path, fn)
            images_path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + temp_path)
    return images_path


def get_statistic_data_from_result(path, images_path):
    with open(path + '.txt') as f:
        content = f.readlines()
    i = 0
    statistic_data = []
    for fn in images_path:
        i += 1
        line_from_score = 'set' + str(i) + '_score'
        statistic_data.append(get_percent(content, line_from_score))
        debug_print(i, line_from_score, get_percent(content, line_from_score), fn)
    return statistic_data


def IoU_metric(pred, label):
    pred = pred.astype(np.bool)
    label = label.astype(np.bool)
    return np.divide(np.sum(np.bitwise_and(pred, label)), np.sum(np.bitwise_or(pred, label)), dtype=np.float32)


def print_statisctic(data, is_train_data):
    images_path = get_image_list(data)
    print("Num images:   " + str(len(images_path)))

    if (is_train_data): #TODO
        statistic_data = []
        for fn in images_path:
            statistic_data.append(0)
    else:
        statistic_data = get_statistic_data_from_result(data, images_path)
    min_score = min(statistic_data)
    min_index = statistic_data.index(min_score)
    max_score = max(statistic_data)
    max_index = statistic_data.index(max_score)
    debug_print(min_index + 1, 'min score', min_score, images_path[min_index])
    debug_print(max_index + 1, 'max score', max_score, images_path[max_index])

    road_pixels_summ = list()
    for name in images_path:
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
        summ = np.sum(img)
        road_pixels_summ.append(summ)

    np_sum_list = np.stack(road_pixels_summ)

    size = 1024 * 1024

    debug_print(np.argmin(np_sum_list) + 1, 'min roads img', np.min(np_sum_list) / size, images_path[np.argmin(np_sum_list)])
    debug_print(np.argmax(np_sum_list) + 1, 'max roads img', np.max(np_sum_list) / size, images_path[np.argmax(np_sum_list)])

    print 'mean = ', str(np.mean(np_sum_list) / size)
    print 'median = ', str(np.median(np_sum_list) / size)


def main():
    print_statisctic('valid_thresh_9999', False)
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    # print_statisctic('train', True)


if __name__ == "__main__":
    main()
