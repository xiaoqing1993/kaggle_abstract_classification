# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 10:39:54 2016

@author: Xiaoqing
"""

'''
class for reaading data and extracting feaatures  
'''

import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class FeatureGenerator():  
    def __init__(self, binary=True, max_features=5000):
        self.binary = binary
        self.max_features = max_features
        self.split_point = 88636
        self.train_in = 'train_in.csv'
        self.train_out = 'train_out.csv'
        self.test_in = 'test_in.csv'
        
    def read_corpus(self, filename):
        print 'reading corpus ...'
        with open(filename,'rb') as originaldata:
            abstract_info = csv.reader(originaldata)
            corpus_X= list(abstract_info)
        corpus_X.pop(0)
        for index, text in enumerate(corpus_X):
            text.pop(0)
            corpus_X[index] = text[0]#.translate(None, string.punctuation)
        return corpus_X
        
    def attach_test_to_train(self, corpus_train, corpus_test):
        for row in corpus_test:
            corpus_train.append(row)
        return corpus_train
        
    def read_y(self):
        print 'reading train_out ...'
        filename = self.train_out
        with open(filename,'rb') as originaldata:
            categ_info = csv.reader(originaldata)
            categ = list(categ_info)
        categ.pop(0)
        y = []
        for index, item in enumerate(categ):
            y.append(item[-1])
        y = np.array(y)
        return y
    
    def F_extraction(self):
        vect = CountVectorizer(binary=self.binary, stop_words='english', max_features=self.max_features) 
        # vect = TfidfVectorizer(stop_words='english')  
        filename_train = self.train_in
        corpus = self.read_corpus(filename_train)
        print 'feature extracting ...'
        X = vect.fit_transform(corpus)  # sparse matrix
        return X
    
    def F_extraction_Kaggle(self):
        vect = CountVectorizer(binary=self.binary, stop_words='english', max_features=self.max_features) 
        # vect = TfidfVectorizer(stop_words='english') 
        file_train = self.train_in
        file_test = self.test_in
        corpus_train = self.read_corpus(file_train)
        self.split_point = len(corpus_train)  # keep num of features less than samples!!!
        corpus_test = self.read_corpus(file_test)
        corpus = self.attach_test_to_train(corpus_train, corpus_test)
        print 'feature extracting ...'
        X = vect.fit_transform(corpus)
        split_p = self.split_point
        X_train = X[0:split_p]
        X_test = X[split_p:]
        return X_train, X_test
