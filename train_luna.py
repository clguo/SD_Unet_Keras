import os

import numpy as np
import cv2
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
np.random.seed(42)
import scipy.misc as mc
data_location = ''
training_images_loc = data_location + 'Luna/train/image/'
training_label_loc = data_location + 'Luna/train/label/'
testing_images_loc = data_location + 'Luna/test/image/'
testing_label_loc = data_location + 'Luna/test/label/'

train_files = os.listdir(training_images_loc)
train_data = []
train_label = []


for i in train_files:
    train_data.append(cv2.resize((mc.imread(training_images_loc + i)), (512, 512)))

    temp = cv2.resize(mc.imread(training_label_loc + i.split('.')[0] + '_mask.tif',mode="L"),
                      (512, 512))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)
train_data = np.array(train_data)

train_label = np.array(train_label)

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []

for i in test_files:
    test_data.append(cv2.resize((mc.imread(testing_images_loc + i)), (512, 512)))
    # Change '_manual1.tiff' to the label name
    temp = cv2.resize(mc.imread(testing_label_loc + i.split('.')[0] + '_mask.tif'),
                      (512, 512))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)
test_data = np.array(test_data)
test_label = np.array(test_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 512, 512, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), 512, 512, 1))  # adapt this if using `channels_first` im

x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 512, 512, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), 512, 512, 1))  # adapt this if using `channels_first` im

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

from  SD_Unet import *
model=SD_UNet(input_size=(512,512,3),start_neurons=16,keep_prob=0.9,block_size=7)
weight="Model/Luna/SD_UNet.h5"
restore=False
if restore and os.path.isfile(weight):
    model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)

model.fit(x_train, y_train,
                epochs=300,
                batch_size=4,
                validation_split=0.12,
                # validation_data=(x_test, y_test),
                shuffle=True,
                callbacks= [TensorBoard(log_dir='./autoencoder'), model_checkpoint]
          )

