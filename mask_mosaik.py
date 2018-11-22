import glob
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm







def next_batch(img,  pathes, ind, sess, tensors):
    top = load_slice(img, 'top')
    right = load_slice(img, 'right')
    bot = load_slice(img, 'bot')
    left = load_slice(img, 'left')

    top_list = list()
    right_list = list()
    bot_list = list()
    left_list = list()

    for i in ind:
        path = pathes[i]
        img_ = cv2.imread(path)
        top_ = load_slice(img_, 'top')
        right_ = load_slice(img_, 'right')
        bot_ = load_slice(img_, 'bot')
        left_ = load_slice(img_, 'left')

        top_list.append(np.concatenate([bot_, top], axis=0))
        right_list.append(np.concatenate([right, left_], axis=0))
        bot_list.append(np.concatenate([bot, top_], axis=0))
        left_list.append(np.concatenate([right_, left], axis=0))

    top_batch = np.stack(top_list, axis=0)
    right_batch = np.stack(right_list, axis=0)
    bot_batch = np.stack(bot_list, axis=0)
    left_batch = np.stack(left_list, axis=0)

    res_top = np.argmax(sess.run(tensors['prediction'], feed_dict={tensors['input']: top_batch, tensors['pahse']: 0, tensors['is_rotated']: 0}), axis=-1)
    res_right = np.argmax(sess.run(tensors['prediction'], feed_dict={tensors['input']: right_batch, tensors['pahse']: 0, tensors['is_rotated']: 0}), axis=-1)
    res_bot = np.argmax(sess.run(tensors['prediction'], feed_dict={tensors['input']: bot_batch, tensors['pahse']: 0, tensors['is_rotated']: 0}), axis=-1)
    res_left = np.argmax(sess.run(tensors['prediction'], feed_dict={tensors['input']: left_batch, tensors['pahse']: 0, tensors['is_rotated']: 0}), axis=-1)
    t, r, b, l = [None, None, None, None]
    if np.sum(res_top) > 0:
        t = ind[np.argmax(res_top)]
    if np.sum(res_right) > 0:
        r = ind[np.argmax(res_right)]
    if np.sum(res_bot) > 0:
        b = ind[np.argmax(res_bot)]
    if np.sum(res_left) > 0:
        l = ind[np.argmax(res_left)]
    return t, r, b, l





def load_slice(img, side=None):
    sl = None
    if side is not None:
        if side == 'top':
            sl = img[:16, :]
        if side == 'right':
            sl = np.transpose(img[:, -16:], axes=[1, 0, 2])
        if side == 'bot':
            sl = img[-16:, :]
        if side == 'left':
            sl = np.transpose(img[:, :16], axes=[1, 0, 2])
        return sl
    else:
        print('Wrong image side')
        exit()






def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def restore_tensors():
    graph = tf.get_default_graph()
    tensors = dict()
    tensors['pahse'] = graph.get_tensor_by_name("phase:0")
    tensors['input'] = graph.get_tensor_by_name("input:0")
    tensors['output'] = graph.get_tensor_by_name("output:0")
    tensors['prediction'] = graph.get_tensor_by_name("prediction:0")
    tensors['is_rotated'] = graph.get_tensor_by_name("is_rotated:0")
    return tensors


def restore_session(path_to_checkpoints):
    path = tf.train.latest_checkpoint(path_to_checkpoints)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(path + ".meta")
    saver.restore(sess, path)
    print ("Model restored:", path)
    return sess


def local_metric(img, img_, sess, tensors):
    merged = np.concatenate((img, img_), axis=0)
    merged = np.expand_dims(merged, 0)
    res = softmax(sess.run(tensors['prediction'], feed_dict={tensors['input']: merged, tensors['pahse']: 0, tensors['is_rotated']:0})[0])
    return res


def similarity_metric(img, img_, sess, tensors):
    summm = list()
    # b_t
    summm.append(local_metric(img[-16:, :], img_[:16, :], sess, tensors)[1])
    # t_b
    summm.append(local_metric(img_[-16:, :], img[:16, :],  sess, tensors)[1])
    # l_r
    summm.append(local_metric(np.transpose(img_[:, -16:], axes=[1, 0, 2]), np.transpose(img[:, :16], axes=[1, 0, 2]), sess, tensors)[1])
    # r_l
    summm.append(local_metric(np.transpose(img[:, -16:], axes=[1, 0, 2]), np.transpose(img_[:, :16], axes=[1, 0, 2]), sess, tensors)[1])

    ind = np.nanargmax(np.array(summm))
    if max(summm) > 0.5:
        return ind, max(summm)
    else:
        return None, None


def main():
    sides = ['t', 'r', 'b', 'l']
    np.seterr(divide='ignore', invalid='ignore')

    path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/road_test_sat'
    ckpts_path = '/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/mosaik/ch/ckpts'

    pathes = glob.glob(path + '/*')
    pathes_ = glob.glob('/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/valid/*') + glob.glob('/media/panchenko/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/DeepGlobe/Roads/train/imgs/*')
    pathes_ = pathes + pathes_
    sess = restore_session(ckpts_path)
    tensors = restore_tensors()
    batch_size = 100
    indexes = range(len(pathes))
    for j, path in enumerate(pathes):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print j, '  ', path
        for i in xrange(0, len(pathes), batch_size):
            ind = indexes[i:i+batch_size]
            res_indexes = next_batch(img, pathes, ind, sess, tensors)
            for k, res_index in enumerate(res_indexes):
                if res_index is not None:
                    print '         ', sides[k], ' ', pathes[res_index]



if __name__ == "__main__":
    main()

