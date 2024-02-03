import os
import imageio
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('test_name_hard', 'hard_nature',
                    'what set of shreads to load')


def load_imgs(name):
    file_names = os.listdir(os.path.join('shredded-images', name))
    file_names.sort()
    Is = []
    for f in file_names:
        I = imageio.v2.imread(os.path.join('shredded-images', name, f))
        Is.append(I)
    return Is

def min_distance(img1, img2):
    min_dist = float('inf')
    min_offst = 0
    offst = img1.shape[0] // 5
    for current_offst in range(-offst, offst + 1):
        start_idx = max(0, -current_offst)
        end_idx = min(img1.shape[0], img2.shape[0] - current_offst)
        if end_idx > start_idx:  # 确保有有效范围
            pixel_number = end_idx - start_idx + 1
            diff = img1[start_idx:end_idx, 0] - img2[start_idx + current_offst:end_idx + current_offst, -1]
            sum_double = np.sum(diff ** 2) / pixel_number
            if sum_double < min_dist:
                min_dist = sum_double
                min_offst = current_offst

    return min_dist, min_offst


def pairwise_distance(Is):
    # dist[i, j]   i: right first column  j: left last column
    dist = np.ones((len(Is), len(Is)))
    offset = np.ones((len(Is), len(Is)))
    for i in range(len(Is)):
        for j in range(len(Is)):
            if i != j:
                dist[i, j], offset[i, j] = min_distance(Is[i], Is[j])
    return dist, offset


def solve(Is):
    '''
    :param Is: list of N images
    :return order: order list of N images
    :return offsets: offset list of N ordered images
    '''

    # order = [10, 3, 15, 16, 13, 0, 11, 1, 2, 7, 8, 9, 5, 17, 4, 14, 6, 12]
    # offsets = [43, 0, 7, 24, 51, 49, 52, 35, 48, 45, 17, 21, 27, 2, 38, 32, 31, 34]
    # We are returning the order and offsets that will work for 
    # hard_campus, you need to write code that works in general for any given
    # Is. Use the solution for hard_campus to understand the format for
    # what you need to return
    dist, offsets = pairwise_distance(Is)

    inds = np.arange(len(Is))
    # run greedy matching
    order = [0]
    for i in range(len(Is) - 1):
        d1 = np.min(dist[0, 1:])
        d2 = np.min(dist[1:, 0])
        if d1 < d2:
            ind = np.argmin(dist[0, 1:]) + 1
            order.insert(0, inds[ind])
            dist[0, :] = dist[ind, :]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            inds = np.delete(inds, ind, 0)
        else:
            ind = np.argmin(dist[1:, 0]) + 1
            order.append(inds[ind])
            dist[:, 0] = dist[:, ind]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            inds = np.delete(inds, ind, 0)
    # recalculate offsets
    relative_offsets = []
    for i in range(len(order)-1):
        relative_offsets.append(-offsets[order[i+1], order[i]])
    absolute_offsets = np.cumsum(relative_offsets)
    absolute_offsets = np.insert(absolute_offsets, 0, 0)
    max_offsets = np.max(absolute_offsets)
    norm_offsets = max_offsets - absolute_offsets
    return order, norm_offsets


def composite(Is, order, offsets):
    Is = [Is[o] for o in order]
    strip_width = 1
    W = np.sum([I.shape[1] for I in Is]) + len(Is) * strip_width
    H = np.max([I.shape[0] + o for I, o in zip(Is, offsets)])
    H = int(H)
    W = int(W)
    I_out = np.ones((H, W, 3), dtype=np.uint8) * 255
    w = 0
    for I, o in zip(Is, offsets):
        I_out[o:o + I.shape[0], w:w + I.shape[1], :] = I
        w = w + I.shape[1] + strip_width
    return I_out

def main(_):
    Is = load_imgs(FLAGS.test_name_hard)
    order, offsets = solve(Is)
    order = [int(x) for x in order]
    offsets = [int(round(x)) for x in offsets]
    I = composite(Is, order, offsets)
    import matplotlib.pyplot as plt
    plt.imshow(I)
    plt.show()

if __name__ == '__main__':
    app.run(main)
