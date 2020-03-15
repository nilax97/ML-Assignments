import pandas as pd
import numpy as np
import utils
import pickle
import random
import time
import sys
import warnings

import sklearn
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, mean_squared_error

if not sys.warnoptions:
	warnings.simplefilter("ignore")

path_train = sys.argv[1];
path_test = sys.argv[2];
part = sys.argv[3];

def f1_score(A,B):
	conf_mat = np.zeros((5,5));
	for i in range(B.shape[0]):
		conf_mat[int(B[i]-1)][int(A[i]-1)] = conf_mat[int(B[i]-1)][int(A[i]-1)] + 1;
	
	p1 = np.zeros(5);
	r1 = np.zeros(5);
	f1 = np.zeros(5);

	for i in range(5):
		p1[i] = conf_mat[i,i]/(np.sum(conf_mat, axis = 0)[i])
		r1[i] = conf_mat[i,i]/(np.sum(conf_mat, axis = 1)[i])
		f1[i] = (2*p1[i]*r1[i])/(p1[i]+r1[i]);

	return round(np.average(f1),5);

#Read and process data
yelp_train = pd.read_json(path_train, lines=True);
yelp_test = pd.read_json(path_test, lines=True);

X_train1 = yelp_train['text'][(yelp_train['stars']==1)].copy();
X_train2 = yelp_train['text'][(yelp_train['stars']==2)].copy();
X_train3 = yelp_train['text'][(yelp_train['stars']==3)].copy();
X_train4 = yelp_train['text'][(yelp_train['stars']==4)].copy();
X_train5 = yelp_train['text'][(yelp_train['stars']==5)].copy();
   
X_train = yelp_train['text'].copy();
Y_train = yelp_train['stars'].copy();
X_test = yelp_test['text'].copy();
Y_test = yelp_test['stars'].copy();

if(part == 'd'):
	for i in X_train1.keys():
		X_train1[i] = utils.getStemmedDocuments(X_train1[i], False);
	for i in X_train2.keys():
		X_train2[i] = utils.getStemmedDocuments(X_train2[i], False);
	for i in X_train3.keys():
		X_train3[i] = utils.getStemmedDocuments(X_train3[i], False);
	for i in X_train4.keys():
		X_train4[i] = utils.getStemmedDocuments(X_train4[i], False);
	for i in X_train5.keys():
		X_train5[i] = utils.getStemmedDocuments(X_train5[i], False);

	for i in X_train.keys():
		X_train[i] = utils.getStemmedDocuments(X_train[i], False);
	for i in X_test.keys():
		X_test[i] = utils.getStemmedDocuments(X_test[i], False);
	
X_merged1 = [X_train1.str.cat(sep=' ')];
X_merged2 = [X_train2.str.cat(sep=' ')];
X_merged3 = [X_train3.str.cat(sep=' ')];
X_merged4 = [X_train4.str.cat(sep=' ')];
X_merged5 = [X_train5.str.cat(sep=' ')];
X_merged = [X_train.str.cat(sep=' ')];

#Train
i = 0;
num = 0;
nb_dict = dict();
for x in X_merged[0].split():
	num = num + 1;
	if(x not in nb_dict):
		nb_dict[x] = 1;
		i=i+1;

num1 = 0;
nb_dict1 = nb_dict.copy();
for x in X_merged1[0].split():
	num1 = num1 + 1;
	nb_dict1[x] = nb_dict1[x] + 1; 
	
num2 = 0;
nb_dict2 = nb_dict.copy();
for x in X_merged2[0].split():
	num2 = num2 + 1;
	nb_dict2[x] = nb_dict2[x] + 1; 

num3 = 0;
nb_dict3 = nb_dict.copy();
for x in X_merged3[0].split():
	num3 = num3 + 1;
	nb_dict3[x] = nb_dict3[x] + 1; 

num4 = 0;
nb_dict4 = nb_dict.copy();
for x in X_merged4[0].split():
	num4 = num4 + 1;
	nb_dict4[x] = nb_dict4[x] + 1; 

num5 = 0;
nb_dict5 = nb_dict.copy();
for x in X_merged5[0].split():
	num5 = num5 + 1;
	nb_dict5[x] = nb_dict5[x] + 1; 
	
for x in nb_dict1:
	nb_dict1[x] = np.log(nb_dict1[x]/(num1 + i))

for x in nb_dict2:
	nb_dict2[x] = np.log(nb_dict2[x]/(num2 + i))

for x in nb_dict3:
	nb_dict3[x] = np.log(nb_dict3[x]/(num3 + i))

for x in nb_dict4:
	nb_dict4[x] = np.log(nb_dict4[x]/(num4 + i))

for x in nb_dict5:
	nb_dict5[x] = np.log(nb_dict5[x]/(num5 + i))

phi1 = np.log(len(X_train1)/len(X_train));
phi2 = np.log(len(X_train2)/len(X_train));
phi3 = np.log(len(X_train3)/len(X_train));
phi4 = np.log(len(X_train4)/len(X_train));
phi5 = np.log(len(X_train5)/len(X_train));

if (part == 'a'):
	#Predict on train set
	correct = 0
	pred_train = np.zeros(Y_train.shape);
	for i in range(len(X_train)):
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		p5 = 0.0
		for x in X_train[i].split():
			p1 = p1 + nb_dict1[x];
			p2 = p2 + nb_dict2[x];
			p3 = p3 + nb_dict3[x];
			p4 = p4 + nb_dict4[x];
			p5 = p5 + nb_dict5[x];
		p1 = p1 + phi1;
		p2 = p2 + phi2;
		p3 = p3 + phi3;
		p4 = p4 + phi4;
		p5 = p5 + phi5;
		if(max(p1,p2,p3,p4,p5) == p1):
			pred_train[i] = int(1);
		elif(max(p1,p2,p3,p4,p5) == p2):
			pred_train[i] = int(2);
		elif(max(p1,p2,p3,p4,p5) == p3):
			pred_train[i] = int(3);
		elif(max(p1,p2,p3,p4,p5) == p4):
			pred_train[i] = int(4);
		elif(max(p1,p2,p3,p4,p5) == p5):
			pred_train[i] = int(5);
		if(pred_train[i] == Y_train[i]):
			correct = correct + 1;

	print("Train Set Accuracy =", round(correct/len(X_train),5));
	print("Macro F1 score =", f1_score(pred_train, Y_train));

	#Predict on test set
	correct = 0
	pred_test = np.zeros(Y_test.shape);
	for i in range(len(X_test)):
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		p5 = 0.0
		for x in X_test[i].split():
			if(x in nb_dict):
				p1 = p1 + nb_dict1[x];
				p2 = p2 + nb_dict2[x];
				p3 = p3 + nb_dict3[x];
				p4 = p4 + nb_dict4[x];
				p5 = p5 + nb_dict5[x];
		p1 = p1 + phi1;
		p2 = p2 + phi2;
		p3 = p3 + phi3;
		p4 = p4 + phi4;
		p5 = p5 + phi5;
		if(max(p1,p2,p3,p4,p5) == p1):
			pred_test[i] = int(1);
		elif(max(p1,p2,p3,p4,p5) == p2):
			pred_test[i] = int(2);
		elif(max(p1,p2,p3,p4,p5) == p3):
			pred_test[i] = int(3);
		elif(max(p1,p2,p3,p4,p5) == p4):
			pred_test[i] = int(4);
		elif(max(p1,p2,p3,p4,p5) == p5):
			pred_test[i] = int(5);
		if(pred_test[i] == Y_test[i]):
			correct = correct + 1;

	print("Test Set Accuracy = ", round(correct/len(X_test),5));
	print("Macro F1 score =", f1_score(pred_test, Y_test));


elif(part == 'b'):
		#Predict on test set
	correct = 0
	pred_test = np.zeros(Y_test.shape);
	for i in range(len(X_test)):
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		p5 = 0.0
		for x in X_test[i].split():
			if(x in nb_dict):
				p1 = p1 + nb_dict1[x];
				p2 = p2 + nb_dict2[x];
				p3 = p3 + nb_dict3[x];
				p4 = p4 + nb_dict4[x];
				p5 = p5 + nb_dict5[x];
		p1 = p1 + phi1;
		p2 = p2 + phi2;
		p3 = p3 + phi3;
		p4 = p4 + phi4;
		p5 = p5 + phi5;
		if(max(p1,p2,p3,p4,p5) == p1):
			pred_test[i] = int(1);
		elif(max(p1,p2,p3,p4,p5) == p2):
			pred_test[i] = int(2);
		elif(max(p1,p2,p3,p4,p5) == p3):
			pred_test[i] = int(3);
		elif(max(p1,p2,p3,p4,p5) == p4):
			pred_test[i] = int(4);
		elif(max(p1,p2,p3,p4,p5) == p5):
			pred_test[i] = int(5);
		if(pred_test[i] == Y_test[i]):
			correct = correct + 1;

	print("Test Set Accuracy = ", round(correct/len(X_test),5));
	print("Macro F1 score =", f1_score(pred_test, Y_test));

	phimax = max(phi1,phi2,phi3,phi4,phi5)
	major_pred = 0;
	if phimax == phi1:
		major_pred = 1;
	elif phimax == phi2:
		major_pred = 2;
	elif phimax == phi3:
		major_pred = 3;
	elif phimax == phi4:
		major_pred = 4;
	elif phimax == phi5:
		major_pred = 5;
	pred_major = np.zeros(Y_test.shape) + major_pred;

	correct = 0;
	for i in range(len(Y_test)):
		if(Y_test[i]==major_pred):
			correct += 1;
	
	print("Majority Prediction Accuracy = ", round(correct/len(X_test),5));
	print("Macro F1 score =", f1_score(pred_major, Y_test));

	pred_random = np.zeros(Y_test.shape);
	correct = 0;
	for i in range(pred_random.shape[0]):
		pred_random[i] = random.randint(1,5);
		if(Y_test[i]==pred_random[i]):
			correct += 1;	
	
	print("Random Prediction Accuracy = ", round(correct/len(X_test),5));
	print("Macro F1 score =", f1_score(pred_random, Y_test));

elif(part == 'c'):
		#Predict on test set
	correct = 0
	pred_test = np.zeros(Y_test.shape);
	for i in range(len(X_test)):
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		p5 = 0.0
		for x in X_test[i].split():
			if(x in nb_dict):
				p1 = p1 + nb_dict1[x];
				p2 = p2 + nb_dict2[x];
				p3 = p3 + nb_dict3[x];
				p4 = p4 + nb_dict4[x];
				p5 = p5 + nb_dict5[x];
		p1 = p1 + phi1;
		p2 = p2 + phi2;
		p3 = p3 + phi3;
		p4 = p4 + phi4;
		p5 = p5 + phi5;
		if(max(p1,p2,p3,p4,p5) == p1):
			pred_test[i] = int(1);
		elif(max(p1,p2,p3,p4,p5) == p2):
			pred_test[i] = int(2);
		elif(max(p1,p2,p3,p4,p5) == p3):
			pred_test[i] = int(3);
		elif(max(p1,p2,p3,p4,p5) == p4):
			pred_test[i] = int(4);
		elif(max(p1,p2,p3,p4,p5) == p5):
			pred_test[i] = int(5);
		if(pred_test[i] == Y_test[i]):
			correct = correct + 1;

	conf_mat = np.zeros((5,5));
	for i in range(Y_test.shape[0]):
		conf_mat[int(Y_test[i]-1)][int(pred_test[i]-1)] = conf_mat[int(Y_test[i]-1)][int(pred_test[i]-1)] + 1;
	print(conf_mat.astype(int))

elif(part == 'd'):
	#Predict on test set
	correct = 0
	pred_test = np.zeros(Y_test.shape);
	for i in range(len(X_test)):
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		p5 = 0.0
		for x in X_test[i].split():
			if(x in nb_dict):
				p1 = p1 + nb_dict1[x];
				p2 = p2 + nb_dict2[x];
				p3 = p3 + nb_dict3[x];
				p4 = p4 + nb_dict4[x];
				p5 = p5 + nb_dict5[x];
		p1 = p1 + phi1;
		p2 = p2 + phi2;
		p3 = p3 + phi3;
		p4 = p4 + phi4;
		p5 = p5 + phi5;
		if(max(p1,p2,p3,p4,p5) == p1):
			pred_test[i] = int(1);
		elif(max(p1,p2,p3,p4,p5) == p2):
			pred_test[i] = int(2);
		elif(max(p1,p2,p3,p4,p5) == p3):
			pred_test[i] = int(3);
		elif(max(p1,p2,p3,p4,p5) == p4):
			pred_test[i] = int(4);
		elif(max(p1,p2,p3,p4,p5) == p5):
			pred_test[i] = int(5);
		if(pred_test[i] == Y_test[i]):
			correct = correct + 1;

	print("Test Set Accuracy (for stemmed data) = ", round(correct/len(X_test),5));
	print("Macro F1 score =", f1_score(pred_test, Y_test));

elif (part == 'e'):
	import nltk
	import sklearn

	from nltk import word_tokenize
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.linear_model import LogisticRegression
	from sklearn.naive_bayes import MultinomialNB

	vectorizer = TfidfVectorizer(preprocessor=None, 
				tokenizer=word_tokenize, 
				analyzer='word', 
				stop_words=None, 
				strip_accents=None, 
				lowercase=True,
				ngram_range=(1,3), 
				min_df=0.0001, 
				max_df=0.9,
				binary=False,
				norm='l2',
				use_idf=1,
			   smooth_idf=1, 
				sublinear_tf=1);

	X_train = vectorizer.fit_transform(X_train)
	X_test = vectorizer.transform(X_test)


	mnb = MultinomialNB()
	mnb.fit(X_train,Y_train)
	predmnb = mnb.predict(X_test)
	print("Feature Engineering Score:",round(accuracy_score(Y_test,predmnb),5));
	print("Macro F1 score =", f1_score(predmnb, Y_test));

elif(part == 'f'):
			#Predict on test set
	correct = 0
	pred_test = np.zeros(Y_test.shape);
	for i in range(len(X_test)):
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		p5 = 0.0
		for x in X_test[i].split():
			if(x in nb_dict):
				p1 = p1 + nb_dict1[x];
				p2 = p2 + nb_dict2[x];
				p3 = p3 + nb_dict3[x];
				p4 = p4 + nb_dict4[x];
				p5 = p5 + nb_dict5[x];
		p1 = p1 + phi1;
		p2 = p2 + phi2;
		p3 = p3 + phi3;
		p4 = p4 + phi4;
		p5 = p5 + phi5;
		if(max(p1,p2,p3,p4,p5) == p1):
			pred_test[i] = int(1);
		elif(max(p1,p2,p3,p4,p5) == p2):
			pred_test[i] = int(2);
		elif(max(p1,p2,p3,p4,p5) == p3):
			pred_test[i] = int(3);
		elif(max(p1,p2,p3,p4,p5) == p4):
			pred_test[i] = int(4);
		elif(max(p1,p2,p3,p4,p5) == p5):
			pred_test[i] = int(5);
		if(pred_test[i] == Y_test[i]):
			correct = correct + 1;

	print("Test Set Accuracy = ", round(correct/len(X_test),5));
	print("Macro F1 score =", f1_score(pred_test, Y_test));

elif(part == 'g'):
	yelp_train = pd.read_json(path_train, lines=True);
	yelp_test = pd.read_json(path_test, lines=True);
   
	X_train = yelp_train['text'].copy();
	Y_train = yelp_train['stars'].copy();
	X_test = yelp_test['text'].copy();
	Y_test = yelp_test['stars'].copy();

	import nltk
	import sklearn

	from nltk import word_tokenize
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.linear_model import LogisticRegression
	from sklearn.naive_bayes import MultinomialNB

	vectorizer = TfidfVectorizer(preprocessor=None, 
				tokenizer=word_tokenize, 
				analyzer='word', 
				stop_words=None, 
				strip_accents=None, 
				lowercase=True,
				ngram_range=(1,3), 
				min_df=0.0001, 
				max_df=0.9,
				binary=False,
				norm='l2',
				use_idf=1,
			   smooth_idf=1, 
				sublinear_tf=1);

	X_train = vectorizer.fit_transform(X_train)
	X_test = vectorizer.transform(X_test)


	mnb = MultinomialNB()
	mnb.fit(X_train,Y_train)
	predmnb = mnb.predict(X_test)
	print("Feature Engineering Score:",round(accuracy_score(Y_test,predmnb),5));
	print("Macro F1 score =", f1_score(predmnb, Y_test));

