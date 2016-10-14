# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 01:01:59 2016

@author: Xiaoqing
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 16:00:32 2016

@author: Xiaoqing
"""
import numpy as np
import operator

class NaiveBayes():
    def __init__(self, X = [0], y = 0, n_categ = 4, alpha = 0.01):
        self.n_categ = n_categ
        self.alpha = alpha
        
    def prior_count(self, y):
        set_y = set(y)
        dict_y = {}
        for categ in set_y:
            dict_y[categ] = np.sum(y==categ)
        return dict_y
    
    def fit(self, X, y):
        self.X = X
        self.y = np.array(y)
        self.dict_y = self.prior_count(self.y)
        self.n_categ = len(set(y))
        N, M = self.X.get_shape() 
        self.dict_prior = {}   # prior probabilities for each category
        self.dict_condi1 = {}   # conditional probabilities for features (use categories as keys)
        self.dict_condi0 = {}
        for categ in self.dict_y:
            self.dict_prior[categ] = self.dict_y[categ]/float(N)
            X_categ = self.X[self.y == categ]
            X_feature = X_categ.transpose()
            n, m = X_categ.get_shape()
            condi_prob1 = [(feature.sum()+self.alpha)/(float(n)+self.n_categ*self.alpha) for feature in X_feature]  # P(x=1|y=categ)
            condi_prob0 = [(n-feature.sum()+self.alpha)/(float(n)+self.n_categ*self.alpha) for feature in X_feature]  # P(x=0|y=categ)
            self.dict_condi1[categ] = condi_prob1
            self.dict_condi0[categ] = condi_prob0

    def predict(self, X):
        Xtest = X
        N, M = Xtest.get_shape()
        predicted = []
        for index in range(N):
            # print index
            prob_pre = {}
            abstract = Xtest.getrow(index)
            abstract = abstract.toarray()
            for categ in self.dict_y:
                condi_prob1 = self.dict_condi1[categ]
                condi_prob0 = self.dict_condi0[categ]
                condi_prob1 = np.array(condi_prob1)
                condi_prob0 = np.array(condi_prob0)
                prior = self.dict_prior[categ]
                prob_pre[categ] = prior * np.prod(condi_prob1[abstract[0]==1]) * np.prod(condi_prob0[abstract[0]==0])
            p = sorted(prob_pre.iteritems(),key=operator.itemgetter(1),reverse=True)
            predicted.append(p[0][0])
        predicted = np.array(predicted)
        return predicted