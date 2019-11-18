import os
import cv2
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix
from keras.callbacks import ModelCheckpoint
import scipy.misc as mc
import math
import numpy as np
from util import *
data_location = ''
testing_images_loc = data_location + 'RC_SLO/test/image/'
testing_label_loc = data_location + 'RC_SLO/test/label/'

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []
desired_size=368
for i in test_files:
    im = mc.imread(testing_images_loc + i)
    label=mc.imread(testing_label_loc + i.split(".")[0] + "_GT.tif")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2=[0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))
    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color2)
    temp = cv2.resize(new_label,(desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

test_data = np.array(test_data)
test_label = np.array(test_label)

x_test = test_data.astype('float32') / 255.

y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first` im
y_test=crop_to_shape(y_test,(len(y_test), 320, 360, 1))

from  SD_Unet import *
model=SD_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,keep_prob=1,block_size=1)
weight="Model/RC_SLO/SD_UNet.h5"


if os.path.isfile(weight): model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)

# plot_model(model, to_file='unet_resnet.png', show_shapes=False, show_layer_names=False)

y_pred = model.predict(x_test)
y_pred= crop_to_shape(y_pred,(40,320,360,1))

y_pred_threshold = []
i=0
for y in y_pred:
    _, temp = cv2.threshold(y, 0.4, 1, cv2.THRESH_BINARY)
    ground = temp * 255
    y_pred_threshold.append(temp)
    y = y * 255
    cv2.imwrite('./RC_SLO/test/result/%d.png' % i, y)
    i+=1
y_test = list(np.ravel(y_test))
y_pred_threshold = list(np.ravel(y_pred_threshold))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
print('Sensitivity:', recall_score(y_test, y_pred_threshold))
print('Specificity', tn / (tn + fp))
print('NPV', tn / (tn + fn))
print('PPV', tp / (tp + fp))
print('AUC:', roc_auc_score(y_test, list(np.ravel(y_pred))))
print("F1:",2*tp/(2*tp+fn+fp))
N=tn+tp+fn+fp
S=(tp+fn)/N
P=(tp+fp)/N
print("MCC:",(tp/N-S*P)/math.sqrt(P*S*(1-S)*(1-P)))