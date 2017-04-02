#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:48:03 2017

@author: Masao Takakuwa
"""
import numpy as np
from common.functions import sigmoid
       
class Relu(object) :
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine(object):
    def __init__(self,W,b):
        #print("Affine")
        self.x = None          # N*50
        self.dW = None         #50,10
        self.db = None             #10
        self.W = W             # 50,10 (in,out)
        self.b = b             # 10   (out)        self.db = None             #10

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):    #dout:N,10
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx
 ###---------------------------------------------------------   
class PathIndex(object) :
    def __init__(self,size) :
        #print("PathIndex size:",size)
        self.idx = 0
        self.i = None
        self.pre_i = None
        self.list_size = size*3+1
        self.rand_list = np.random.randint(0,size,self.list_size)
        
    def update(self,pre_i):
        self.pre_i = pre_i
        self.idx = (self.idx+1) % self.list_size
        self.i = self.rand_list[self.idx]
        return self.i

class DPLRelu(Relu,PathIndex) :
    def __init__(self):
        self.mask = None
    def update(self,pre_i) :
        self.i = pre_i
        self.pre_i = pre_i
        return self.i

    def DPLforward(self, x):
        #print("DPLRelu-forward x:",x.shape)
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def DPLbackward(self, dout):
        #dout = dout[:,np.newaxis]
        dout[self.mask] = 0
        dx = dout
        return dx
    
class DPLSigmoid(Sigmoid,PathIndex):
    def __init__(self):
        self.out = None
    def update(self,pre_i) :
        self.i = pre_i
        self.pre_i = pre_i
        return self.i

    def DPLforward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def DPLbackward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class FirstPath(Affine,PathIndex):
    def __init__(self,w,b):
        Affine.__init__(self,w,b)
        PathIndex.__init__(self,b.shape[0])
        # w: 784,50  (in,out) b:50 (out) x: N*784
        self.dW = np.zeros_like(w) #784,50
        self.db = np.zeros_like(b) #50
        
    def DPLforward(self,x):
        #print("FiretPath-forward x:",x.shape,"W:",self.W.shape,"b:",self.b.shape)
        self.x = x
        W = self.W[:,self.i:self.i+1]
        out = np.dot(x,W)+self.b[self.i]
        return out
    
    def DPLbackward(self,dout):    #dout:N*1   
        #print("FirstPath-backward dout:",dout.shape,"X:",self.x.shape,"W:",self.W.shape,"db",self.db.shape)
        dx = np.outer(dout,self.W[self.i].T)
        self.dW[:,self.i:self.i+1] = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        #print("FirstPath-backward dout:",dout.shape,"X:",self.x.shape,"W:",self.W.shape,"db",self.db.shape,"dx:",dx.shape)
        return dx

class LastPath(FirstPath):
    def __init__(self,w,b):
        # w: 100,10 (in,out) b:10 (out) x: N*10
        FirstPath.__init__(self,w,b)
        
    def DPLforward(self,x):
        self.x = x 
        #print("LastPath-forward x:",x.shape,"W:",self.W.shape,"b:",self.b.shape)
        out = np.outer(x,self.W[self.pre_i])+self.b
        return out
    
    def DPLbackward(self,dout):    #dout:N*10  
        #print("LastPath-Bacward dout:",dout.shape,"W:",self.W.shape,"db:",self.db.shape,"x:",self.x.shape)
        dx = np.dot(dout,self.W[self.pre_i])
        dx = dx[:,np.newaxis]
        self.dW[self.pre_i] = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        #print("LastPath-Bacward dout:",dout.shape,"W:",self.W.shape,"db:",self.db.shape,"x:",self.x.shape,"dx:",dx.shape)

        return dx
    
class DPLPath(FirstPath):
    def __init__(self,w,b):
        # w: 50,100 (in,out) b:100 (out) x: N
        FirstPath.__init__(self,w,b)

    def DPLforward(self,x):
        self.x = x 
        #print("DPLPath-forward x:",x.shape,"W:",self.W.shape,"b:",self.b.shape)
        out = x*self.W[self.pre_i][self.i]+self.b[self.i]
        return out
    
    def DPLbackward(self,dout):    #dout:N  
        #print("DPLPath-Bacward dout:",dout.shape,"W:",self.W.shape,"db:",self.db.shape,"x:",self.x.shape)
        dx = dout*self.W[self.pre_i][self.i]
        self.dW[self.pre_i][self.i] = np.dot(self.x.T,dout)
        self.db[self.pre_i] = np.sum(dout,axis=0)
        #print("DPLPath-Bacward dout:",dout.shape,"W:",self.W.shape,"db:",self.db.shape,"x:",self.x.shape,"dx:",dx.shape)
        return dx

