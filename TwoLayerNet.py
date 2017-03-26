#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:18:37 2017

@author: Masao Takakuwa
"""
import sys,os
sys.path.append(os.pardir)
import numpy as np

from DPLlayers import FirstAffine,LastAffine
from common.layers import Relu,SoftmaxWithLoss
from common.gradient import numerical_gradient

   
class DPLTwoLayerNet:
    
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['Wt1']=weight_init_std*np.random.randn(hidden_size,input_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        
        # 隠れレイヤーのi,jの擬似乱数生成
        self.i_size = hidden_size * 3 + 1 ;
        self.i_rand= np.random.randint(0,hidden_size,self.i_size)
        self.i = 0
        
       
        self.FirstAffine= FirstAffine(self.params['Wt1'],self.params['b1'])
        self.Relu=       Relu()
        self.LastAffine= LastAffine(self.params['W2'],self.params['b2'])
        self.lastlayers = SoftmaxWithLoss()
    
    def update_i(self):
        self.i = (self.i+1) % self.i_size
        return self.i 
        
    def predict(self,x):
        i = self.i_rand[self.i]
        Wt1b1 = self.FirstAffine.forward(x,i)
        Relu = self.Relu.forward(Wt1b1)
        W2b2 = self.LastAffine.forward(Relu,i)
#       print("x,Wt1b1,Relu,W2b2",x.shape,Wt1b1.shape,Relu.shape,W2b2.shape)
        return W2b2
    
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastlayers.forward(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t=np.argmax(t,axis=1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W:self.loss(x,t)
        
        grads = {}
        grads['Wt1']=numerical_gradient(loss_W,self.params['Wt1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])
        return grads
    
    def gradient(self,x,t):
        # forwad
        self.loss(x,t)
        #backward
        dout = 1
        W2b2 = self.lastlayers.backward(dout)
        Relu = self.LastAffine.backward(W2b2)
        Wt1b1 = self.Relu.backward(Relu)
        self.FirstAffine.backward(Wt1b1)
#        print("x,Wt1b1,Relu,W2b2",x.shape,Wt1b1.shape,Relu.shape,W2b2.shape)
        
        grads = {}
        grads['Wt1']=self.FirstAffine.dW
        grads['b1']=self.FirstAffine.db
        grads['W2']=self.LastAffine.dW
        grads['b2']=self.LastAffine.db
        return grads
