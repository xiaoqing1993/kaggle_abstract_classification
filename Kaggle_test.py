# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 22:41:01 2016

@author: Xiaoqing
"""


#import string
#import numpy as np
import pandas as pd
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from Multinomial import MultiNaiveBayes
#from Multinomial import BernouNaiveBayes
from feature_extraction import FeatureGenerator


F_gnr = FeatureGenerator(max_features=30000)
X_train, X_test = F_gnr.F_extraction_Kaggle()
y_train = F_gnr.read_y()

print 'training ...'
clf_lr = LogisticRegression()
clf_lr.fit(X_train,y_train)

print 'predicting...'
y_pre = clf_lr.predict(X_test)

print 'saving result...'
output_name = 'test_y_1000.csv'
test = pd.DataFrame(columns=['id','category'])
test.id = range(len(y_pre))
test.category = y_pre
test.to_csv(output_name)

print 'FINISHED!'