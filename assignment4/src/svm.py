import os
import scipy.io
import json
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PIL import Image

import time
import sys
import warnings

if not sys.warnoptions:
	warnings.simplefilter("ignore")

data_fldr = sys.argv[1];
test_fldr = sys.argv[2];

X = list();
Y = list();
images = list();
count = 0;
for sub_fldr in sorted(os.listdir(data_fldr)):
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
        img = Image.open(file).convert("L");
        img = np.asarray(img);
        img = img[33:194,8:152];
        img = img[::2,::2]
        images.append(img);
    images = np.asarray(images);
    X.append(images);
    Y.append(scores);
    images = list();
    count+= 1;


pca_X = X[0];
for i in range(1,50):
    pca_X = np.r_[pca_X,X[i]];

pca = PCA(n_components=50)
pca.fit(pca_X.reshape(pca_X.shape[0],-1));

X_pca = list();
for i in range(len(X)):
    X_pca.append(pca.transform(X[i].reshape(X[i].shape[0],-1)));

X_svm = list();
Y_svm = list();
for x in range(len(X_pca)):
    A = X_pca[x];
    B = Y[x];
    assert len(A) == len(B);
    for i in range(7, A.shape[0]-3):
        if(B[i]==1):
            weight = 15;
        else:
            weight = 1;
        for j in range(weight):
            select = np.arange(10);
            np.random.shuffle(select);
            if(select[0]!=5):
                continue;
            ind = np.arange(6);
            np.random.shuffle(ind);
            ind = ind + i - 7;
            ind = np.insert(ind,0,i-1);
            ind[0:5].sort();
            temp_x = A[ind[0:5]].ravel()
            assert temp_x.shape[0] == 250;
            X_svm.append(temp_x);
            Y_svm.append(B[i+2]);
X_svm = np.asarray(X_svm)
Y_svm = np.asarray(Y_svm) 

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
        img = Image.open(file).convert("L");
        img = np.asarray(img);
        img = img[33:194,8:152];
        img = img[::2,::2]
        images.append(img);
    images = np.asarray(images);
    images = pca.transform(images.reshape(images.shape[0],-1));
    X_test.append(images.ravel());
    count+=1;

X_test = np.asarray(X_test)
Y_test = scores;

svc_lin = LinearSVC();
svc_lin.fit(X_svm, Y_svm);
print("LINEAR KERNEL");
preds = svc_lin.predict(X_svm);
print("Training Accuracy: ", accuracy_score(Y_svm, preds));
print(classification_report(Y_svm, preds));
print(confusion_matrix(Y_svm, preds));

preds = svc_lin.predict(X_test);
print("Test Accuracy: ", accuracy_score(Y_test, preds));
print(classification_report(Y_test, preds));
print(confusion_matrix(Y_test, preds));
print("GAUSSIAN KERNEL");
svc_rbf = SVC();
svc_rbf.fit(X_svm, Y_svm);

preds = svc_rbf.predict(X_svm);
print("Training Accuracy: ", accuracy_score(Y_svm, preds));
print(classification_report(Y_svm, preds));
print(confusion_matrix(Y_svm, preds));

preds = svc_rbf.predict(X_test);
print("Test Accuracy: ", accuracy_score(Y_test, preds));
print(classification_report(Y_test, preds));
print(confusion_matrix(Y_test, preds));

