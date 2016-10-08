# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 16:23:24 2016

@author: Xiaoqing
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 13:24:53 2016

@author: Xiaoqing
"""
#import csv
#import string
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from Multinomial import MultiNaiveBayes
#from Multinomial import BernouNaiveBayes
from sklearn.metrics import accuracy_score
#from sklearn.feature_extraction.text import CountVectorizer
from feature_extraction import FeatureGenerator


F_gnr = FeatureGenerator(max_features=1000)
X = F_gnr.F_extraction()
y = F_gnr.read_y()

print 'training ...'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_lr = LogisticRegression()
clf_lr.fit(X_train,y_train)
print 'Logistic:', accuracy_score(y_test,clf_lr.predict(X_test))

clf_nbB = BernoulliNB()
clf_nbB.fit(X_train,y_train)
print 'Bernoulli:', accuracy_score(y_test,clf_nbB.predict(X_test))

clf_nbM = MultinomialNB()
clf_nbM.fit(X_train,y_train)
print 'Multinomial:', accuracy_score(y_test,clf_nbM.predict(X_test))

clf_hm1 = MultiNaiveBayes()
clf_hm1.fit(X_train, y_train)
y_pre = clf_hm1.predict(X_test)
print 'multinomial by mxq:', accuracy_score(y_test, y_pre)

'''
clf_hm2 = BernouNaiveBayes()
clf_hm2.fit(X_train, y_train)
y_pre = clf_hm2.predict(X_test)
print 'Bernoulli by mxq:', accuracy_score(y_test, y_pre)

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train,y_train)
print 'Decision Tree:', clf_dt.score(X_test, y_test)

clf_nn = MLPClassifier()
clf_nn.fit(X_train,y_train)
print 'Neural:', clf_nn.score(X_test, y_test)

clf_svm = SVC()
clf_svm.fit(X_train, y_train)
print 'SVM:', clf_svm.score(X_test, y_test)
'''