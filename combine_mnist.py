#!/usr/bin/env python

import numpy as np

# ave width:
# [22, 12, 19, 15, 19, 28, 21, 28, 26, 28]
# ave height:
# [28, 28, 28, 28, 28, 15, 28, 12, 23, 20]

def draw(canvas, image, offset):
    x_off, y_off = offset
    h, w, d = image.shape

    #image = image[3:-3, 3:-3]

    # print(image.shape)
    for x in range(w):
        for y in range(h):
            j = image[y][x]
            if j[-1]:
                #print(x, x_off, y, y_off)
                canvas[y + y_off][x + x_off] = j
                # canvas.itemset((y + y_off, x + x_off), j)

import glob, os.path
from keras.utils import to_categorical


FILES = glob.glob(os.path.join('/home/darwin/Projects/datasets/shapenet/render/screenshots/modelsByCategory', '**/*.png'), recursive=True)
TEST_FILES = glob.glob(os.path.join('/home/darwin/Projects/datasets/shapenet/render/screenshots/test', '**/*.png'), recursive=True)

#a = np.load('/home/deeplearning2/Downloads/car.train.npz')
#CAR_IMG = a['images']
#CAR_LABELS = a['labels']

#a = np.load('/home/deeplearning2/Downloads/motorcycle.train.npz')
#MOTOR_IMG = a['images']
#MOTOR_LABELS = a['labels']

a = np.load('/home/deeplearning2/Downloads/car.test.npz')
CAR_IMG = a['images']
CAR_LABELS = a['labels']
#
a = np.load('/home/deeplearning2/Downloads/motorcycle.test.npz')
MOTOR_IMG = a['images']
MOTOR_LABELS = a['labels']

N = len(CAR_LABELS)


def get_label(f):
    if '/car/' in f:
        label = 0
    elif '/motorcycle/' in f:
        label = 1
    # else:
    #     label = 2
    return to_categorical(label, 2)

import os.path

def get_pose(f):
    f = os.path.basename(f)
    n, e, a, d = f.split('_')
    d = d.replace('.png', '')
    pose = np.array(list(map(float, [e, a, d])))
    return pose


def sample_and_combine(x_pool, y_pool, overlap_factor, FILES=TEST_FILES):
    # n = len(FILES)
    # first = second = np.random.randint(n)
    # while np.array_equal(get_label(FILES[second]), get_label(FILES[first])):
    #     second = np.random.randint(n)
    # x1 = imread(FILES[first])
    # y1 = get_label(FILES[first])
    # pose1 = get_pose(FILES[first])
    # x2 = imread(FILES[second])
    # y2 = get_label(FILES[second])
    # pose2 = get_pose(FILES[second])
    #
    # a = '/home/darwin/Projects/datasets/shapenet/render/screenshots/test/motorcycle/b767982d38b5171e429f1c522640e6f0-7_-0.3715715_1.1131217_1.3595791.png'
    # b = '/home/darwin/Projects/datasets/shapenet/render/screenshots/test/car/501ac8493044eff04d44f5db04bf14b8-44_-0.14040226_1.6006266_1.3633044.png'
    #
    #
    # print(FILES[first], FILES[second])

    n = np.random.randint(N)
    x1 = CAR_IMG[n]
    pose1 = CAR_LABELS[n]
    y1 = np.array([1, 0])

    n = np.random.randint(N)
    x2 = MOTOR_IMG[n]
    pose2 = MOTOR_LABELS[n]
    y2 = np.array([0, 1])

    # Swap 50% of the time
    if np.random.choice(2):
        xt = x1
        yt = y1
        poset = pose1
        x1 = x2
        y1 = y2
        pose1 = pose2
        x2 = xt
        y2 = yt
        pose2 = poset


    h, w, d = x1.shape

    # Config: bounding box dimensions
    #142.03138294
    #105.22550292
    #
    # bb_w = 142//2
    # bb_h = 106//2
    # # Config: overlap
    # # overlap_factor = 0.0
    #
    # area_overlap = overlap_factor * bb_w * bb_h
    # min_x = round(area_overlap / bb_h)
    # min_y = round(area_overlap / bb_w)
    # x_range = bb_w - min_x
    # left_x = np.random.randint(-x_range, x_range + 1)
    # x_overlap = bb_w - abs(left_x)
    # y_overlap = round(area_overlap / x_overlap) if x_overlap else 0
    #
    # total_width = bb_w * 2 - x_overlap + (w - bb_w)
    # total_height = bb_h * 2 - y_overlap + (h - bb_h)
    #
    # # print(total_width, total_height, x_overlap, y_overlap)
    #
    # max_width = bb_w * 2 - min_x + (w - bb_w)
    # max_height = bb_h * 2 - min_y + (h - bb_h)
    # max_dim = max(max_width, max_height)
    #
    #combined = np.zeros((128, 128, 4), dtype=x1.dtype)
    #
    # x_offset1 = np.random.randint(0, max_width - total_width + 1)
    # y_offset1 = np.random.randint(0, max_height - total_height + 1)
    #
    # x_offset2 = x_offset1 + bb_w - x_overlap# + (w - bb_w)/2
    # y_offset2 = y_offset1 + bb_h - y_overlap# + (h - bb_h)/2

    x_offset1 = 0
    y_offset1 = 0
    x_offset2 = 0
    y_offset2 = 128-96

    off1 = (round(x_offset1), round(y_offset1))
    off2 = (round(x_offset2), round(y_offset2))

    offsets = [off1, off2]

    idx = np.random.choice([0, 1])
    final_off1 = offsets[idx]
    final_off2 = off2 if final_off1 is off1 else off1

    #draw(combined, x1, final_off1)
    #draw(combined, x2, final_off2)

    y_off1 = final_off1[1]
    if y_off1 == 0:
        y_pad1 = (0, 128-96)
        y_pad2 = (128-96, 0)
    else:
        y_pad2 = (0, 128 - 96)
        y_pad1 = (128 - 96, 0)

    x1 = np.pad(x1, (y_pad1,(0,0),(0,0)), mode='constant', constant_values=0)
    x2 = np.pad(x2, (y_pad2,(0,0),(0,0)), mode='constant', constant_values=0)
    y = y1.copy()
    y[np.argmax(y2)] = 1

    combined = x1.copy()
    visible = x2 > 0
    combined[visible] = x2[visible]

    # preprocess
    x1 = x1 / 127.5 - 1.
    x2 = x2 / 127.5 - 1.
    combined = combined / 127.5 - 1.

    return x1, x2, combined, y1, y2, y, pose1, pose2


# def main():
#     bb_w = [0 for i in range(10)]
#     bb_h = [0 for i in range(10)]
#     (x, y), (_, _) = mnist.load_data()
#     for i in range(len(x)):
#         x1 = x[i]
#         y1 = y[i]
#         left_x = ((x1 > 0).argmax(axis=0) > 0).argmax()
#         top_y = ((x1 > 0).argmax(axis=1) > 0).argmax()
#         x1 = np.fliplr(x1)
#         x1 = np.flipud(x1)
#         right_x = 28 - ((x1 > 0).argmax(axis=0) > 0).argmax()
#         bottom_y = 28 - ((x1 > 0).argmax(axis=1) > 0).argmax()
#         w = right_x - left_x
#         h = bottom_y - top_y
#         bb_w[y1] += w
#         bb_h[y1] += h
#
#     for i in range(10):
#         bb_w[i] /= 6000.
#         bb_h[i] /= 6000.
#
#     return bb_w, bb_h

import glob
from skimage.io import imread


def main2():
    bb_w = 0
    bb_h = 0
    count = 0

    for img in glob.iglob('/home/darwin/Projects/datasets/shapenet/transparent/**/*.png', recursive=True):
        x1 = imread(img)
        try:
            height, width, d = x1.shape
        except ValueError:
            print(img)
        x1 = np.split(x1, d, axis=-1)[-1].squeeze()
        left_x = ((x1 > 0).argmax(axis=0) > 0).argmax()
        top_y = ((x1 > 0).argmax(axis=1) > 0).argmax()
        x1 = np.fliplr(x1)
        x1 = np.flipud(x1)
        right_x = width - ((x1 > 0).argmax(axis=0) > 0).argmax()
        bottom_y = height - ((x1 > 0).argmax(axis=1) > 0).argmax()
        w = right_x - left_x
        h = bottom_y - top_y
        bb_w += w
        bb_h += h
        count += 1

    bb_h /= count
    bb_w /= count

    print(bb_w, bb_h)



if __name__ == '__main__':
    main2()
