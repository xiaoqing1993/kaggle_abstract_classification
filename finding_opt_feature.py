# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 21:03:42 2016

@author: Xiaoqing
"""

#import csv
#import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from feature_extraction import FeatureGenerator
import matplotlib.pyplot as plt

feature_num = [100, 1000, 3000, 5000, 10000, 30000, 50000, 60000, 70000, 80000]
num = len(feature_num)
acc_lr = np.zeros(num)
acc_nbB = np.zeros(num)
acc_nbM = np.zeros(num)
acc_rf = np.zeros(num)

for index, max_f in enumerate(feature_num):
    F_gnr = FeatureGenerator(binary=True, max_features=max_f)
    X = F_gnr.F_extraction()
    y = F_gnr.read_y()
    print 'max feature num = ', max_f
    print 'training ...'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf_lr = LogisticRegression()
    clf_lr.fit(X_train,y_train)
    acc_lr[index] = accuracy_score(y_test,clf_lr.predict(X_test))

    clf_nbB = BernoulliNB()
    clf_nbB.fit(X_train,y_train)
    acc_nbB[index] = accuracy_score(y_test,clf_nbB.predict(X_test))
    
    clf_nbM = MultinomialNB()
    clf_nbM.fit(X_train,y_train)
    acc_nbM[index] = accuracy_score(y_test,clf_nbM.predict(X_test))

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    acc_rf[index] = accuracy_score(y_test,clf_rf.predict(X_test))

plt.figure(1)
plot_lr = plt.plot(feature_num, acc_lr)
plot_lr = plt.plot(feature_num, acc_nbB) 
plot_lr = plt.plot(feature_num, acc_nbM) 
plot_lr = plt.plot(feature_num, acc_rf)      
#plt.title('Cost VS number of iteration ')
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.show()

