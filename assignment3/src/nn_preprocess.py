import numpy as np

import time
import sys
import warnings

if not sys.warnoptions:
	warnings.simplefilter("ignore")

path_train = sys.argv[1];
path_test = sys.argv[2];
one_train = sys.argv[3];
one_test = sys.argv[4];

def one_hot(array):
	n = array.shape[0];
	X = np.zeros((n,85));
	Y = np.zeros((n,10));
	for i in range(n):
		offset = 0;
		for j in range(10):
			temp = int(array[i,j] + offset -1);
			X[i, temp] = 1;
			if(j%2==0):
				offset+=4;
			else:
				offset+=13;
		temp = int(array[i,10]);
		Y[i, temp] = 1;
	return X,Y

train_arr = np.genfromtxt(path_train,delimiter=',');
test_arr = np.genfromtxt(path_test,delimiter=',');

X_train, Y_train = one_hot(train_arr);
X_test, Y_test = one_hot(test_arr);

train_one = np.c_[X_train, Y_train]
test_one = np.c_[X_test, Y_test]

np.savetxt(one_train, train_one, delimiter=",");
np.savetxt(one_test, test_one, delimiter=",");