#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:18:37 2017

@author: Masao Takakuwa
"""
import sys,os
sys.path.append(os.pardir)
import numpy as np

from DPLlayers import FirstAffine,LastAffine,Affine,DPLRelu
from common.layers import Relu,SoftmaxWithLoss
from common.gradient import numerical_gradient

   
class DPLTwoLayerNet:
    
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.hidden_size = hidden_size
        self.params = {}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        
        
        # 隠れレイヤーのi,jの擬似乱数生成
        self.i_size = hidden_size * 3 + 1 ;
        self.i_rand= np.random.randint(0,hidden_size,self.i_size)
        self.i = 0
        self.test = 0
        
        if (self.test == 0) :
            self.FirstAffine= FirstAffine(self.params['W1'],self.params['b1'])
            self.Relu=       DPLRelu()
            self.LastAffine= LastAffine(self.params['W2'],self.params['b2'])
            self.lastlayers = SoftmaxWithLoss()
        else :
            self.FirstAffine= Affine(self.params['W1'],self.params['b1'])
            self.Relu=       Relu()
            self.LastAffine= Affine(self.params['W2'],self.params['b2'])
            self.lastlayers = SoftmaxWithLoss()
        
            
    def set_batch(self,x,t):
        self.x = x
        self.t = t
        self.batch_size = x.shape[0]

    def update_i(self):
        self.i = (self.i+1) % self.i_size
        return self.i 
        
    def DPLpredict(self):
        i = self.i_rand[self.i]
        Wt1b1 = self.FirstAffine.DPLforward(self.x,i)
        Relu = self.Relu.forward(Wt1b1,i)
        W2b2 = self.LastAffine.DPLforward(Relu,i)
        #print("DPLpredict x:",self.x.shape,"W1b1:",Wt1b1.shape,"Relu:",Relu.shape,"W2b2:",W2b2.shape)
        return W2b2
    
    def predict(self):
        Wt1b1 = self.FirstAffine.forward(self.x)
        Relu = self.Relu.forward(Wt1b1)
        W2b2 = self.LastAffine.forward(Relu)
#       print("x,Wt1b1,Relu,W2b2",x.shape,Wt1b1.shape,Relu.shape,W2b2.shape)
        return W2b2
    
    def loss(self):
        
        if (self.test == 0):
            y = self.DPLpredict()
        else :
            y = self.predict()
        #print("loss y:",y.shape,"t",self.t.shape)    
        return self.lastlayers.forward(y,self.t)
    
    def accuracy(self):
        y = self.predict()
        y = np.argmax(y,axis=1)
        if self.t.ndim != 1 : t=np.argmax(self.t,axis=1)
        accuracy = np.sum(y == t)/float(self.x.shape[0])
        return accuracy
    
    def numerical_gradient(self):
        loss_W = lambda W:self.loss()
        
        grads = {}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])
        return grads
    
    def gradient(self):
        # forwad
        self.loss()
        #backward
        dout = 1
        W2b2 = self.lastlayers.backward(dout) 
        if (self.test == 0):
            Relu = self.LastAffine.DPLbackward(W2b2)
            #print("gradiant Relu:",Relu.shape,"W2b2",W2b2.shape)
            Wt1b1 = self.Relu.backward(Relu)
        else :
            Relu = self.LastAffine.backward(W2b2)
#            print("gradiant Relu:",Relu.shape,"W2b2",W2b2.shape)
            Wt1b1 = self.Relu.backward(Relu)
            self.FirstAffine.backward(Wt1b1)
#        print("x,Wt1b1,Relu,W2b2",x.shape,Wt1b1.shape,Relu.shape,W2b2.shape)
        
        grads = {}
        grads['W1']=self.FirstAffine.dW
        grads['b1']=self.FirstAffine.db
        grads['W2']=self.LastAffine.dW
        grads['b2']=self.LastAffine.db
        return grads
