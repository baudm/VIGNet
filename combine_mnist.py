#!/usr/bin/env python

from keras.datasets import fashion_mnist as mnist
import numpy as np

# ave width:
# [22, 12, 19, 15, 19, 28, 21, 28, 26, 28]
# ave height:
# [28, 28, 28, 28, 28, 15, 28, 12, 23, 20]

def draw(canvas, image, offset):
    x_off, y_off = offset
    w, h, d = image.shape

    #image = image[3:-3, 3:-3]

    # print(image.shape)
    for x in range(w):
        for y in range(h):
            j = image[y][x]
            if j > 1/255.:
                #print(x, x_off, y, y_off)
                canvas.itemset((y + y_off, x + x_off), j)


def sample_and_combine(x_pool, y_pool):
    n = x_pool.shape[0]
    first = second = np.random.randint(n)
    while np.array_equal(y_pool[second], y_pool[first]):
        second = np.random.randint(n)
    x1 = x_pool[first]
    y1 = y_pool[first]
    x2 = x_pool[second]
    y2 = y_pool[second]

    w, h, d = x1.shape

    # Config: bounding box dimensions
    bb_w = 22
    bb_h = 24
    # Config: overlap
    overlap_factor = 0.25

    area_overlap = overlap_factor * bb_w * bb_w
    min_x = round(area_overlap / bb_h)
    min_y = round(area_overlap / bb_w)
    x_range = bb_w - min_x
    left_x = np.random.randint(-x_range, x_range + 1)
    x_overlap = bb_w - abs(left_x)
    y_overlap = round(area_overlap / x_overlap) if x_overlap else 0

    total_width = bb_w * 2 - x_overlap + (w - bb_w)
    total_height = bb_h * 2 - y_overlap + (h - bb_h)

    max_width = bb_w * 2 - min_x + (w - bb_w)
    max_height = bb_h * 2 - min_y + (h - bb_h)
    max_dim = max(max_width, max_height)

    combined = np.zeros((max_dim, max_dim), dtype=x1.dtype)


    x_offset1 = np.random.randint(0, max_dim - total_width + 1)
    y_offset1 = np.random.randint(0, max_dim - total_height + 1)

    x_offset2 = x_offset1 + bb_w - x_overlap# + (w - bb_w)/2
    y_offset2 = y_offset1 + bb_h - y_overlap# + (h - bb_h)/2

    off1 = (round(x_offset1), round(y_offset1))
    off2 = (round(x_offset2), round(y_offset2))

    offsets = [off1, off2]

    idx = np.random.choice([0, 1])
    final_off1 = offsets[idx]
    final_off2 = off2 if final_off1 is off1 else off1

    draw(combined, x1, final_off1)
    draw(combined, x2, final_off2)
    y = y1.copy()
    y[np.argmax(y2)] = 1
    return x1, x2, combined.reshape((max_dim, max_dim, 1)), y1, y2, y


def main():
    bb_w = [0 for i in range(10)]
    bb_h = [0 for i in range(10)]
    (x, y), (_, _) = mnist.load_data()
    for i in range(len(x)):
        x1 = x[i]
        y1 = y[i]
        left_x = ((x1 > 0).argmax(axis=0) > 0).argmax()
        top_y = ((x1 > 0).argmax(axis=1) > 0).argmax()
        x1 = np.fliplr(x1)
        x1 = np.flipud(x1)
        right_x = 28 - ((x1 > 0).argmax(axis=0) > 0).argmax()
        bottom_y = 28 - ((x1 > 0).argmax(axis=1) > 0).argmax()
        w = right_x - left_x
        h = bottom_y - top_y
        bb_w[y1] += w
        bb_h[y1] += h

    for i in range(10):
        bb_w[i] /= 6000.
        bb_h[i] /= 6000.

    return bb_w, bb_h


if __name__ == '__main__':
    main()