#!/usr/bin/env python

import numpy as np

def draw(canvas, image, k):
    #x_off = np.random.randint(0, 9)
    #y_off = np.random.randint(0, 9)
    x_off = 0 if k == 1 else 48-28
    y_off = 0 if k == 1 else 48-28
    #image = image[3:-3, 3:-3]
    w, h, d = image.shape
    # print(image.shape)
    for x in range(w):
        for y in range(h):
            j = image[x][y]
            if j > 1/255.:
                #print(x, x_off, y, y_off)
                canvas.itemset((x + x_off, y + y_off), j)


def sample_and_combine(x_pool, y_pool):
    n = x_pool.shape[0]
    first = second = np.random.randint(n)
    while np.array_equal(y_pool[second], y_pool[first]):
        second = np.random.randint(n)
    x1 = x_pool[first]
    y1 = y_pool[first]
    x2 = x_pool[second]
    y2 = y_pool[second]
    combined = np.zeros((48, 48), dtype=x1.dtype)
    draw(combined, x1, 1)
    draw(combined, x2, 2)
    y = y1.copy()
    y[np.argmax(y2)] = 1
    return x1, x2, combined.reshape(48, 48, 1), y1, y2, y


def main():
    num_samples = 60000000
    num_test = 10000000
    (x_train, y_train), (x_test, y_test) = (1,2),(1,2)

    c_x_train = []
    c_y_train = []
    while num_samples:
        x, y = sample_and_combine(x_train, y_train)
        c_x_train.append(x)
        c_y_train.append(y)
        num_samples -= 1

    c_x_train = np.stack(c_x_train)
    c_y_train = np.stack(c_y_train)

    c_x_test = []
    c_y_test = []
    while num_test:
        x, y = sample_and_combine(x_test, y_test)
        c_x_test.append(x)
        c_y_test.append(y)
        num_test -= 1

    c_x_test = np.stack(c_x_test)
    c_y_test = np.stack(c_y_test)

    with open('data.npz') as f:
        np.savez(f, x_train=c_x_train, y_train=c_y_train, x_test=c_x_test, y_test=c_y_test)


if __name__ == '__main__':
    main()