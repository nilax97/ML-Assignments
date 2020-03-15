import os
import scipy.io
import json
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, rmsprop, adam
from keras.metrics import binary_accuracy
from keras import backend as K

import time
import sys
import warnings

if not sys.warnoptions:
	warnings.simplefilter("ignore")

data_fldr = sys.argv[1];
test_fldr = sys.argv[2];

model = Sequential()
model.add(Conv2D(filters=32, 
                 kernel_size=(3, 3),
                 strides=(2,2),
                 activation='relu',
                 input_shape=(810,144,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3),
                 strides=(2,2),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=binary_crossentropy,
              optimizer=adam(),
             metrics=['accuracy']);

folders = sorted(os.listdir(data_fldr))
X = list();
Y = list();
count = 0;
for sub_fldr in folders:
    images = list();
    if(sub_fldr=='.DS_Store'):
        continue;
    img_fld = os.path.join(data_fldr, sub_fldr);
    for file_name in sorted(os.listdir(img_fld)):
        file = os.path.join(img_fld, file_name);
        if(file=='.DS_Store'):
            continue;
        if(file[-3:]=="csv"):
            scores = np.genfromtxt(file,delimiter=',');
            scores = np.insert(scores, 0, 0);
            continue;
        img = cv2.imread(file);
        img = img[32:194,8:152].astype("uint8");
        images.append(img);
    images = np.asarray(images);
    X.append(images);
    Y.append(scores);
    count+= 1;

X_cnn = list();
Y_cnn = list();
for x in range(len(X)):
    A = X[x];
    B = Y[x];
    assert len(A) == len(B);
    for i in range(7, A.shape[0]-3):
        weight = 1;
        if(B[i]==1):
            weight = 15;
        for j in range(weight):
            ind = np.arange(6);
            np.random.shuffle(ind);
            ind = ind + i - 7;
            ind = np.insert(ind,0,i-1);
            ind[0:5].sort();
            temp_x = A[ind[0:5]];
            tempp_x = np.r_[temp_x[0],temp_x[1], temp_x[2], temp_x[3], temp_x[4]];
            X_cnn.append(tempp_x);
            Y_cnn.append(B[i]);
X_cnn = np.asarray(X_cnn)
Y_cnn = np.asarray(Y_cnn)

X_train, X_val, Y_train, Y_val = train_test_split(X_cnn/255.0, Y_cnn,test_size=0.1);

model.fit(X_train, Y_train,
          batch_size=32,
          epochs=2,
          verbose=0,
          validation_data=(X_val, Y_val))

X_test = list();
count = 0;
for sub_fldr in sorted(os.listdir(test_fldr)):
    if(sub_fldr=='.DS_Store'):
        continue;
    if(sub_fldr[-3:]=="csv"):
        scores = np.genfromtxt(os.path.join(test_fldr, sub_fldr),delimiter=',');
        continue;
    img_fld = os.path.join(test_fldr, sub_fldr);
    images = list();
    for file_name in sorted(os.listdir(img_fld)):
        file = os.path.join(img_fld, file_name);
        if(file=='.DS_Store'):
            continue;
        img = cv2.imread(file);
        img = img[32:194,8:152].astype("uint8")
        images.append(img);
    images = np.asarray(images);
    temp_x = np.r_[images[0],images[1], images[2], images[3], images[4]];
    pred = model.predict(temp_x.reshape(1, 
                                        temp_x.shape[0],
                                        temp_x.shape[1],
                                        temp_x.shape[2])/255.0)
    
    X_test.append(pred);
    count+=1;
X_test = np.asarray(X_test)

X_test = X_test.ravel()

preds = X_test.ravel().copy();
mean = np.average(preds)
for i in range(preds.shape[0]):
    if(preds[i]>=0.7):
        preds[i] = 1;
    else:
        preds[i] = 0;
preds = preds.astype(int)

file = open("predictions.csv", "w");
file.write("id,prediction\n");
for i in range(preds.shape[0]):
    file.write(str(i) + "," + str(preds[i]) + '\n');
file.close();