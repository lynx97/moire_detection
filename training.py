import tensorflow as tf
import cv2
import numpy as np
import random
import pywt
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)  #chay server thi comment
sess = tf.Session(config=config)
set_session(sess)

def build_model(h, w, c=1):
    LL = tf.keras.layers.Input(shape=(h,w, c))
    # print("LL", LL)
    LH = tf.keras.layers.Input(shape=(h,w, c))
    HL = tf.keras.layers.Input(shape=(h,w, c))
    #conv block 1
    ll1 = tf.keras.layers.Conv2D(32, (7, 7), strides=(1,1), padding='same')(LL)
    ll1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(ll1)
    lh1 = tf.keras.layers.Conv2D(32, (7, 7), strides=(1,1), padding='same')(LH)
    lh1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(lh1)
    hl1 = tf.keras.layers.Conv2D(32, (7, 7), strides=(1,1), padding='same')(HL)
    hl1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(hl1)

    #get maximum
    max_lh_hl = tf.keras.layers.maximum([lh1, hl1])
    # print("MAX", max_lh_hl)
    #multiply
    mul_ll_max = tf.keras.layers.multiply([ll1, max_lh_hl])
    # print("MUL", mul_ll_max)
    mul_ll_max = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(mul_ll_max)
    mul_ll_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(4,4), padding='same')(mul_ll_max)
    # print(mul_ll_max)
    mul_ll_max = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(mul_ll_max)
    mul_ll_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(mul_ll_max)
    # print(mul_ll_max)
    mul_ll_max = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(mul_ll_max)
    mul_ll_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(mul_ll_max)
    # print(mul_ll_max)

    flatten = tf.keras.layers.Flatten()(mul_ll_max)
    flatten = tf.keras.layers.Dropout(0.25)(flatten)
    flatten_32 = tf.keras.layers.Dense(32, activation='relu')(flatten)
    output = tf.keras.layers.Dense(2, activation='softmax')(flatten_32)

    model = tf.keras.Model(inputs=[LL, LH, HL], outputs=output)
    model.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['acc'])
    return model

import matplotlib.pyplot as plt

def scale(x, mode=0, axis=None):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    if mode != 0:
        x = (x - 0.5) * 2
    return x

def DWT(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2 
    norm_LL = scale(LL, 0, 2)
    norm_LH = scale(LH, -1, 2)
    norm_HL = scale(HL, -1, 2)
    return norm_LL, norm_LH, norm_HL

# DWT(cv2.imread("test.jpg"))

def preproces_data(img_path, size):
    img = cv2.imread(img_path, 0)
    if img is None:
        return img
    w, h = img.shape[1], img.shape[0]
    #random crop
    rand = np.random.randint(10, size=4)
    crop_left = int(rand[0]/100 * w)
    crop_right = int(rand[1]/100 * w)
    crop_top = int(rand[2]/100 * h)
    crop_bottom = int(rand[3]/100 * h)
    img = img[crop_top:(h-crop_bottom), crop_left:(w-crop_right)]

    #resize image
    img = cv2.resize(img, size)
    return img

def generator(folder_path_natural, folder_path_recapture, batch_size=2, is_training=True):
    num_0 = os.listdir(folder_path_natural)
    num_1 = os.listdir(folder_path_recapture)
    random.shuffle(num_0)
    random.shuffle(num_1)
    num_step = (len(num_0) + len(num_1))//batch_size
    while(True):
        for i in range(num_step):
            idx = i
            if idx < min(len(num_0), len(num_1)):
                path_0 = os.path.join(folder_path_natural, num_0[idx])
                try:
                    ll0, lh0, hl0 = DWT(preproces_data(path_0, (1280, 960)))
                    ll0 = np.expand_dims(ll0, -1)
                    lh0 = np.expand_dims(lh0, -1)
                    hl0 = np.expand_dims(hl0, -1)
                    l0 = [1.0, 0]

                    path_1 = os.path.join(folder_path_recapture, num_1[idx])
                    ll1, lh1, hl1 = DWT(preproces_data(path_1, (1280, 960)))
                    ll1 = np.expand_dims(ll1, -1)
                    lh1 = np.expand_dims(lh1, -1)
                    hl1 = np.expand_dims(hl1, -1)
                    l1 = [0.0, 1.0]
                    ll_arr = np.array([ll0, ll1])
                    lh_arr = np.array([lh0, lh1])
                    hl_arr = np.array([hl0, hl1])
                    outp = np.array([l0, l1])
                    yield ([ll_arr, lh_arr, hl_arr], outp)
                except Exception as e:
                    continue

natural_train_path = "/home/liem/hai/vinID_project/recapture_detection/moire_pattern/old_front/training/natural"
recapture_train_path = "/home/liem/hai/vinID_project/recapture_detection/moire_pattern/old_front/training/recapture"
natural_val_path = "/home/liem/hai/vinID_project/recapture_detection/moire_pattern/old_front/validation/natural"
recapture_val_path = "/home/liem/hai/vinID_project/recapture_detection/moire_pattern/old_front/validation/recapture"
train_generator = generator(natural_train_path, recapture_train_path)
val_generator = generator(natural_val_path, recapture_val_path)
CHECKPOINT = "checkpoint_origin"
if not os.path.exists(CHECKPOINT):
    os.mkdir(CHECKPOINT)
filepath=os.path.join(CHECKPOINT, "weights-{epoch:02d}-{val_acc:.4f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model_train = build_model(482, 642, 1)
model_train.fit_generator(generator=train_generator,
                    steps_per_epoch=6392,
                    validation_data=val_generator,
                    validation_steps=559,
                    epochs=1000,
                    callbacks=[checkpoint]
)