import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

import time
import sys
import warnings

if not sys.warnoptions:
	warnings.simplefilter("ignore")

sub_part = int(sys.argv[1]);
path_train = sys.argv[2];
path_test = sys.argv[3];
path_val = sys.argv[4];

train_arr = np.genfromtxt(path_train,delimiter=',');
train_arr = train_arr[2:,1:].copy()
X_train = train_arr[:,:-1].astype(int)
Y_train = train_arr[:,-1].astype(int)

val_arr = np.genfromtxt(path_val,delimiter=',');
val_arr = val_arr[2:,1:].copy()
X_val = val_arr[:,:-1].astype(int)
Y_val = val_arr[:,-1].astype(int)

test_arr = np.genfromtxt(path_test,delimiter=',');
test_arr = test_arr[2:,1:].copy()
X_test = test_arr[:,:-1].astype(int)
Y_test = test_arr[:,-1].astype(int)

def one_hot(X,c):
    X = X.reshape(-1).astype(int);
    X = X + np.min(X);
    one_hot = np.eye(c)[X];
    return one_hot;    

def cat_edit(X):
    final_arr = X[:,0].copy();
    edits = [3,4,6,7,8,9,10,11];
    for i in range(2,X.shape[1]+1):
        data = X[:,i-1];
        if i in edits:
            if(i==3):
                oh = one_hot(data, 7);
            elif(i==4):
                oh = one_hot(data, 4);
            else:
                oh = one_hot(data, 12);
            final_arr = np.c_[final_arr, oh];
        else:
            final_arr = np.c_[final_arr, data];
    return final_arr;

if(sub_part == 4):
	dc = DecisionTreeClassifier(max_depth=3, min_samples_split=1150, min_samples_leaf=75);
	dc.fit(X_train, Y_train);
	train_pred = dc.predict(X_train);
	print("Train Accuracy :",accuracy_score(Y_train, train_pred));
	val_pred = dc.predict(X_val);
	print("Val Accuracy :",accuracy_score(Y_val, val_pred));
	test_pred = dc.predict(X_test);
	print("Test Accuracy :",accuracy_score(Y_test, test_pred));

elif(sub_part == 5):
	train1_X = cat_edit(X_train)
	test1_X = cat_edit(X_test)
	val1_X = cat_edit(X_val)
	dc = DecisionTreeClassifier(max_depth=3, min_samples_split=1150, min_samples_leaf=75);
	dc.fit(train1_X, Y_train);
	train_pred = dc.predict(train1_X);
	print("Train Accuracy :",accuracy_score(Y_train, train_pred));
	val_pred = dc.predict(val1_X);
	print("Val Accuracy :",accuracy_score(Y_val, val_pred));
	test_pred = dc.predict(test1_X);
	print("Test Accuracy :",accuracy_score(Y_test, test_pred));

elif(sub_part== 6):
	train1_X = cat_edit(X_train)
	test1_X = cat_edit(X_test)
	val1_X = cat_edit(X_val)
	rmfr = RandomForestClassifier(n_estimators=100, max_features=10, bootstrap=True)
	rmfr.fit(train1_X, Y_train);
	train_pred = rmfr.predict(train1_X);
	print("Train Accuracy :",accuracy_score(Y_train, train_pred));
	val_pred = rmfr.predict(val1_X);
	print("Val Accuracy :",accuracy_score(Y_val, val_pred));
	test_pred = rmfr.predict(test1_X);
	print("Test Accuracy :",accuracy_score(Y_test, test_pred));
else:
	print("Part not attempted");