import numpy as np
import pickle
import cvxopt
from cvxopt import matrix, solvers

import sklearn
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from python.svmutil import *

import time
import sys
import warnings

if not sys.warnoptions:
	warnings.simplefilter("ignore")

path_train = sys.argv[1];
path_test = sys.argv[2];
binary = int(sys.argv[3]);
part = sys.argv[4];

train_array = np.genfromtxt(path_train,delimiter=',');
train_array_x = train_array[:,0:784]/255;
train_array_y = train_array[:,784].reshape(train_array.shape[0],1);

test_array = np.genfromtxt(path_test,delimiter=',');
test_array_x = test_array[:,0:784]/255;
test_array_y = test_array[:,784].reshape(test_array.shape[0],1);

def linear_kernel(X,y):
	M = y * X;
	return np.dot(M, M.T);

def gaussian_kernel(X, sigma=0.05):
	m = X.shape[0];
	X2 = np.sum(np.multiply(X, X),axis=1, keepdims=True);
	K0 = X2 + X2.T - 2 * np.matmul(X, X.T)
	return np.power(np.exp(-sigma),K0);

def gaussian_kernel_elem(X1,X2, sigma = 0.05):
	return np.exp(-sigma * np.linalg.norm(X1-X2)**2)

class SVM(object):
	def __init__(self, kernel=linear_kernel, C=1.0):
		self.kernel = kernel;
		self.C = C;
	
	def fit(self, X, y):
		m,n = X.shape;
		if(self.kernel == linear_kernel):
			P = matrix(self.kernel(X,y));
		else:
			K = gaussian_kernel(X);
			P = matrix(y * y.T * K);
		q = matrix(np.ones(m) * -1.0);
		A = matrix(y,(1,m));
		b = matrix(0.0);
		
		G1 = np.eye(m) * -1.0;
		G2 = np.eye(m);
		G = matrix(np.vstack((G1, G2)));
		h1 = np.zeros(m);
		h2 = np.ones(m) * self.C;
		h = matrix(np.hstack((h1, h2)));
		solvers.options['show_progress'] = False
		solution = solvers.qp(P, q, G, h, A, b);
		
		alpha = np.ravel((solution['x'])).reshape(m,1);	  
		self.supp_vec_flag = (alpha > 1e-5).ravel();
		self.indices = np.arange(len(alpha))[self.supp_vec_flag];
		self.alpha = alpha[self.supp_vec_flag];
		self.supp_vec = X[self.supp_vec_flag];
		self.supp_vec_y = y[self.supp_vec_flag];
		
		if(self.kernel==linear_kernel):
			self.w = np.sum(self.alpha * self.supp_vec * self.supp_vec_y, axis = 0, keepdims=True).T;
			b1 = np.min(np.matmul(X[(y == 1).ravel()], self.w));
			b2 = np.max(np.matmul(X[(y == -1).ravel()], self.w));
			self.b = -(b1+b2)/2;
		else:
			self.w = None;
			self.b = 0.0;

	def predict(self, X):
		if (self.kernel == linear_kernel):
			return np.sign(np.dot(X, self.w) + self.b);
		else:
			m = X.shape[0];
			pred = np.zeros(m);
			for i in range(m):
				temp = 0;
				s=0;
				#if(i%100==0):
					#print(i)
				for alpha, supp_vec, supp_vec_y in zip(self.alpha, self.supp_vec, self.supp_vec_y):
					s += alpha * supp_vec_y * gaussian_kernel_elem(X[i],supp_vec);
				pred[i] = s + self.b;
			return np.sign(pred);

if(binary == 0):
	X_train_bin = train_array_x[(train_array_y==3).ravel() | (train_array_y==4).ravel()];
	Y_train_bin = train_array_y[(train_array_y==3).ravel() | (train_array_y==4).ravel()];
	Y_train_bin = -1.0 * (Y_train_bin==3) + 1.0 * (Y_train_bin==4);

	X_test_bin = test_array_x[(test_array_y==3).ravel() | (test_array_y==4).ravel()];
	Y_test_bin = test_array_y[(test_array_y==3).ravel() | (test_array_y==4).ravel()];
	Y_test_bin = -1.0 * (Y_test_bin==3) + 1.0 * (Y_test_bin==4);

	if(part == 'a'):
		clf = SVM();
		clf.fit(X_train_bin, Y_train_bin);
		preds = clf.predict(X_test_bin);
		print("Linear Score: ", round(accuracy_score(preds,Y_test_bin),5))

	if(part == 'b'):
		clf = SVM(kernel=gaussian_kernel);
		clf.fit(X_train_bin, Y_train_bin);
		preds = clf.predict(X_test_bin);
		print("RBF Score: ", round(accuracy_score(preds,Y_test_bin),5))

	if(part == 'c'):
		model = svm_train(Y_train_bin.ravel(),X_train_bin, '-s 0 -t 0 -g 0.05 -q');
		label_predict, accuracy, decision_values=svm_predict(Y_test_bin.ravel(),X_test_bin,model, '-q');
		print("Linear Score: ", round(accuracy_score(label_predict,Y_test_bin),5))

		model = svm_train(Y_train_bin.ravel(),X_train_bin, '-s 0 -t 2 -g 0.05 -q');
		label_predict_g, accuracy_g, decision_values_g=svm_predict(Y_test_bin.ravel(),X_test_bin,model,'-q');
		print("RBF Score: ", round(accuracy_score(label_predict_g,Y_test_bin),5))

else:
	if(part == 'a'):
		for i in range(10):
			for j in range(i+1,10):
				suffix = str(i) + "-_-" + str(j);
				globals()["Xtrain_" + suffix] = train_array_x[(train_array_y==i).ravel() | (train_array_y==j).ravel()];
				temp = train_array_y[(train_array_y==i).ravel() | (train_array_y==j).ravel()];
				globals()["Ytrain_" + suffix] = -1.0 * (temp==i) + 1.0 * (temp==j);
				globals()["clf_" + suffix] = SVM(kernel=gaussian_kernel);
				globals()["clf_" + suffix].fit(globals()["Xtrain_" + suffix], globals()["Ytrain_" + suffix]);
		
		preds_train = np.zeros((train_array_y.shape[0],10,10))
		for i in range(10):
			for j in range(i+1,10):
				suffix = str(i) + "-_-" + str(j);
				temp = globals()["clf_" + suffix].predict(train_array_x);
				preds_train[:,i,j] = (temp-1)/(-2);
				preds_train[:,j,i] = (temp+1)/2;

		preds_train = np.abs(preds_train)
		preds_train = np.argmin(np.sum(preds_train, axis=1), axis = 1)
		print("Training Accuracy: ", accuracy_score(preds_train,train_array_y))

		preds_test = np.zeros((test_array_y.shape[0],10,10))
		for i in range(10):
			for j in range(i+1,10):
				suffix = str(i) + "-_-" + str(j);
				temp = globals()["clf_" + suffix].predict(test_array_x);
				preds_test[:,i,j] = (temp-1)/(-2);
				preds_test[:,j,i] = (temp+1)/2;

		preds_test = np.abs(preds_test)
		preds_test = np.argmin(np.sum(preds_test, axis=1), axis = 1)
		print("Test Accuracy: ", accuracy_score(preds_test,test_array_y))

	if(part=='b'):
		model = svm_train(train_array_y.ravel(),train_array_x, '-s 0 -t 2 -g 0.05 -q');
		label_predict, accuracy, decision_values=svm_predict(train_array_y.ravel(),train_array_x,model,'-q');
		print("Train Accuracy :", accuracy_score(label_predict, train_array_y));
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');
		print("Test Accuracy :", accuracy_score(label_predict, test_array_y));

	if(part =='c'):
		model = svm_train(train_array_y.ravel(),train_array_x, '-s 0 -t 2 -g 0.05 -q');
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');

		conf_mat = np.zeros((10,10));
		for i in range(test_array_y.shape[0]):
			conf_mat[int(test_array_y[i])][int(label_predict[i])] = conf_mat[int(test_array_y[i])][int(label_predict[i])] + 1;
		print(conf_mat.astype(int));

	if(part == 'd'):
		x_train, x_dev,y_train, y_dev = train_test_split(train_array_x, train_array_y, test_size = 0.1);
		x_train = train_array_x;
		y_train = train_array_y;
		print("C = 1e-5 (Accuracies - Dev & Test");
		model = svm_train(y_train.ravel(),x_train, '-s 0 -t 2 -g 0.05 -c 1e-5 -q');
		label_predict, accuracy, decision_values=svm_predict(y_dev.ravel(),x_dev,model, '-q');
		print("Dev Set :", accuracy_score(label_predict, y_dev));
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');
		print("Test Set :", accuracy_score(label_predict, test_array_y));

		print("C = 1e-3 (Accuracies - Dev & Test");

		model = svm_train(y_train.ravel(),x_train, '-s 0 -t 2 -g 0.05 -c 1e-3 -q');
		label_predict, accuracy, decision_values=svm_predict(y_dev.ravel(),x_dev,model, '-q');
		print("Dev Set :", accuracy_score(label_predict, y_dev));
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');
		print("Test Set :", accuracy_score(label_predict, test_array_y));

		print("C = 1 (Accuracies - Dev & Test");

		model = svm_train(y_train.ravel(),x_train, '-s 0 -t 2 -g 0.05 -c 1 -q');
		label_predict, accuracy, decision_values=svm_predict(y_dev.ravel(),x_dev,model, '-q');
		print("Dev Set :", accuracy_score(label_predict, y_dev));
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');
		print("Test Set :", accuracy_score(label_predict, test_array_y));

		print("C = 5 (Accuracies - Dev & Test");

		model = svm_train(y_train.ravel(),x_train, '-s 0 -t 2 -g 0.05 -c 5 -q');
		label_predict, accuracy, decision_values=svm_predict(y_dev.ravel(),x_dev,model, '-q');
		print("Dev Set :", accuracy_score(label_predict, y_dev));
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');
		print("Test Set :", accuracy_score(label_predict, test_array_y));

		print("C = 10 (Accuracies - Dev & Test");

		model = svm_train(y_train.ravel(),x_train, '-s 0 -t 2 -g 0.05 -c 10 -q');
		label_predict, accuracy, decision_values=svm_predict(y_dev.ravel(),x_dev,model, '-q');
		print("Dev Set :", accuracy_score(label_predict, y_dev));
		label_predict, accuracy, decision_values=svm_predict(test_array_y.ravel(),test_array_x,model, '-q');
		print("Test Set :", accuracy_score(label_predict, test_array_y));