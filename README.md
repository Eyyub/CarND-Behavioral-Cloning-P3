# Behaviorial Cloning Project

Original README [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/README.md)

[@eyyub_s](https://twitter.com/eyyub_s)
[Video](https://twitter.com/eyyub_s/status/899647418379124736)
## How to build the project
Run `python build_dataset.py path/to/driving_log.csv`
Then run `python model.py` and `python drive.py model.hdf5`


## Preprocessing
The image preprocessing is done in `build_dataset.py`.

## Model summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 80, 20, 24)        1824
_________________________________________________________________
batch_normalization_1 (Batch (None, 80, 20, 24)        96
_________________________________________________________________
activation_1 (Activation)    (None, 80, 20, 24)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 40, 10, 36)        21636
_________________________________________________________________
batch_normalization_2 (Batch (None, 40, 10, 36)        144
_________________________________________________________________
activation_2 (Activation)    (None, 40, 10, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 40, 10, 48)        15600
_________________________________________________________________
batch_normalization_3 (Batch (None, 40, 10, 48)        192
_________________________________________________________________
activation_3 (Activation)    (None, 40, 10, 48)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 40, 10, 48)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 19200)             0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               9830912
_________________________________________________________________
batch_normalization_4 (Batch (None, 512)               2048
_________________________________________________________________
activation_4 (Activation)    (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 100)               51300
_________________________________________________________________
batch_normalization_5 (Batch (None, 100)               400
_________________________________________________________________
activation_5 (Activation)    (None, 100)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 101
=================================================================
Total params: 9,924,253
Trainable params: 9,922,813
Non-trainable params: 1,440
_________________________________________________________________
