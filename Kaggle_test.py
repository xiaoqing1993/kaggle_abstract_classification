# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 22:41:01 2016

@author: Xiaoqing
"""


#import string
#import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


#from Multinomial import MultiNaiveBayes
#from Multinomial import BernouNaiveBayes
from feature_stemmer import FeatureGenerator


F_gnr = FeatureGenerator(binary=False, max_features=70000)
X_train, X_test = F_gnr.F_extraction_Kaggle()
y_train = F_gnr.read_y()

print 'training ...'

clf_nbM = MultinomialNB()
clf_nbM.fit(X_train, y_train)

print 'predicting...'
y_pre = clf_nbM.predict(X_test)

print 'saving result...'
output_name = 'test_y_70k_stemmer.csv'
test = pd.DataFrame(columns=['id','category'])
test.id = range(len(y_pre))
test.category = y_pre
test.to_csv(output_name)

print 'FINISHED!'