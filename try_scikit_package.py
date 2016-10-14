# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 16:23:24 2016

@author: Xiaoqing
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from NaiveBayes_sp import NaiveBayes
from sklearn.metrics import accuracy_score
from feature_extraction import FeatureGenerator


F_gnr = FeatureGenerator(binary=True, max_features=60000)
X = F_gnr.F_extraction()
y = F_gnr.read_y()

print 'training ...'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
clf_lr = LogisticRegression()
clf_lr.fit(X_train,y_train)
print 'Logistic:', accuracy_score(y_test,clf_lr.predict(X_test))
'''
clf_nbB = BernoulliNB()
clf_nbB.fit(X_train,y_train)
y_b = clf_nbB.predict(X_test)
print 'Bernoulli:', accuracy_score(y_test, y_b)

clf_nbM = MultinomialNB()
clf_nbM.fit(X_train,y_train)
y_m = clf_nbM.predict(X_test)
print 'Multinomial:', accuracy_score(y_test, y_m)

clf_hm1 = NaiveBayes()
clf_hm1.fit(X_train, y_train)
y_pre = clf_hm1.predict(X_test)
print 'naive bayes by mxq:', accuracy_score(y_test, y_pre)