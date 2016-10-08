# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 16:00:32 2016

@author: Xiaoqing
"""
import numpy as np
import operator
class MultiNaiveBayes():
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
        self.X = np.array(X)
        self.y = np.array(y)
        self.dict_y = self.prior_count(self.y)
        self.n_categ = len(set(y))
        N, M = np.shape(self.X) 
        self.dict_prior = {}   # prior probabilities for each category
        self.dict_condi1 = {}   # conditional probabilities for features (use categories as keys)
        self.dict_condi0 = {}
        for categ in self.dict_y:
            self.dict_prior[categ] = self.dict_y[categ]/float(N)
            X_categ = self.X[self.y == categ]
            X_feature = X_categ.T
            n, m = np.shape(X_categ)
            condi_prob1 = [(np.sum(feature)+self.alpha)/(float(n)+self.n_categ*self.alpha) for feature in X_feature]  # P(x=1|y=categ)
            condi_prob0 = [(n-np.sum(feature)+self.alpha)/(float(n)+self.n_categ*self.alpha) for feature in X_feature]  # P(x=0|y=categ)
            self.dict_condi1[categ] = condi_prob1
            self.dict_condi0[categ] = condi_prob0
    
    def predict(self, X):
        Xtest = np.array(X)
        N, M = np.shape(Xtest)
        predicted = []
        for index, abstract in enumerate(Xtest):
            prob_pre = {}
            for categ in self.dict_y:
                condi_prob1 = self.dict_condi1[categ]
                condi_prob0 = self.dict_condi0[categ]
                prob_pre[categ] = self.dict_prior[categ]
                for j, word in enumerate(abstract):
                    if word == 1:
                        prob = condi_prob1[j]
                    else: prob = condi_prob0[j]
                    prob_pre[categ] = prob_pre[categ] * prob
            p = sorted(prob_pre.iteritems(),key=operator.itemgetter(1),reverse=True)
            predicted.append(p[0][0])
        predicted = np.array(predicted)
        return predicted

class BernouNaiveBayes():
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
        self.X = np.array(X)
        self.y = np.array(y)
        self.dict_y = self.prior_count(self.y)
        self.n_categ = len(set(y))
        N, M = np.shape(self.X) 
        self.dict_prior = {}   # prior probabilities for each category
        self.dict_condi1 = {}   # conditional probabilities for features (use categories as keys)
        self.dict_condi0 = {}
        for categ in self.dict_y:
            self.dict_prior[categ] = self.dict_y[categ]/float(N)
            X_categ = self.X[self.y == categ]
            X_feature = X_categ.T
            n, m = np.shape(X_categ)
            condi_prob1 = [(np.sum(feature)+self.alpha)/(float(n)+self.n_categ*self.alpha) for feature in X_feature]  # P(x=1|y=categ)
            condi_prob0 = [(n-np.sum(feature)+self.alpha)/(float(n)+self.n_categ*self.alpha) for feature in X_feature]  # P(x=0|y=categ)
            self.dict_condi1[categ] = condi_prob1
            self.dict_condi0[categ] = condi_prob0
    
    def predict(self, X):
        Xtest = np.array(X)
        N, M = np.shape(Xtest)
        predicted = []
        for index, abstract in enumerate(Xtest):
            prob_pre = {}
            for categ in self.dict_y:
                condi_prob1 = self.dict_condi1[categ]
                condi_prob0 = self.dict_condi0[categ]
                prob_pre[categ] = self.dict_prior[categ]
                for j, word in enumerate(abstract):
                    if word == 1:
                        prob = (condi_prob1[j]**int(word)) * ((1-condi_prob1[j])**int(1-word))
                    else: prob = (condi_prob0[j]**int(word)) * ((1-condi_prob0[j])**int(1-word))
                    prob_pre[categ] = prob_pre[categ] * prob
            p = sorted(prob_pre.iteritems(),key=operator.itemgetter(1),reverse=True)
            predicted.append(p[0][0])
        predicted = np.array(predicted)
        return predicted