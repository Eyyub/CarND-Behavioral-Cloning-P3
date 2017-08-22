import os
import sys
import random
import csv
import numpy as np
import h5py

from PIL import Image, ImageOps

def _image_preprocessing(filename, croptop, cropbot, xsize, ysize):
    im = Image.open(filename)

    width, height = im.size[0], im.size[1]
    im2 = im.crop((0, croptop, width, height - cropbot))
    downsampled_im = ImageOps.fit(im2, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.transpose(np.array(downsampled_im, dtype=np.float32), (1,0,2)) / 255.

    im2.close()
    im.close()

    return norm_im

if __name__ == '__main__':
    path = sys.argv[1]
    xy = []
    f_ds = h5py.File('dataset.hdf5', 'w')

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            xy.append((row[0], float(row[3])))
            xy.append((row[1], float(row[3]) + 0.25))
            xy.append((row[2], float(row[3]) - 0.25))

    random.shuffle(xy)
    imgs, steers = zip(*xy)
    dataset_xs = f_ds.create_dataset('xs', (len(imgs), 160, 40, 3), dtype='f')
    for i in range(len(imgs)):
        print(i)
        if i == 1:
            print(dataset_xs[0])
        dataset_xs[i] = _image_preprocessing(imgs[i], 60, 25, 160, 40)

    dataset_ys = f_ds.create_dataset('ys', (len(steers), 1), dtype='f')
    dataset_ys[...] = np.asarray(steers).reshape((len(steers), 1))#2))
    f_ds.close()
