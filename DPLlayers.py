#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:48:03 2017

@author: Masao Takakuwa
"""
import numpy as np
        
class Affine(object):
    def __init__(self,w,b):
        self.W = w             # 50,10 (in,out)
        self.b = b             # 10   (out)
        self.x = None          # N*50
        self.dW = None         #50,10
        self.db = None             #10
        
    def forward(self,x,i=0):
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):    #dout:N,10
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx
    
class DPLRelu :
    def __init__(self):
        self.mask = None

    def DPLforward(self, x,i=0):
        #print("DPLRelu:",x.shape,i)
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def DPLbackward(self, dout):
        dout = dout[:,np.newaxis]
        dout[self.mask] = 0
        dx = dout
        return dx
    
class FirstAffine(Affine):
    def __init__(self,w,b):
        super().__init__(w,b)
        # w: 784,50  (in,out) b:50 (out) x: N*784
        self.i = None
        self.dW = np.zeros_like(w) #784,50
        self.db = np.zeros_like(b) #50
        
    def DPLforward(self,x,i):
        self.i = i
        self.x = x
        W = self.W[:,i:i+1]
        out = np.dot(x,W)+self.b[i]
        return out
    
    def DPLbackward(self,dout):    #dout:N*1   
 #       print("DPLbackward dout:",dout.shape,"X:",self.x.shape,"W:",self.W.shape,"db",self.db.shape)
        dx = np.outer(dout,self.W[self.i].T)
        self.dW[:,self.i:self.i+1] = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx

class LastAffine(Affine):
    def __init__(self,w,b):
        super().__init__(w,b)
        # w: 50,10 (in,out) b:10 (out) x: N
        self.i = None
        self.dW = np.zeros_like(w) #50,10
        self.db = np.zeros_like(b) #10
        
    def DPLforward(self,x,i):
        self.i = i
        self.x = x 
        #print("DpLforward x:",x.shape,"i:",i,"W:",self.W.shape,"b:",self.b.shape)
        out = np.outer(x,self.W[i])+self.b
        return out
    
    def DPLbackward(self,dout):    #dout:N*10  
        #print("DPlBacward dout:",dout.shape,"W:",self.W.shape,"db:",self.db.shape,"x:",self.x.shape)
        dx = np.dot(dout,self.W[self.i])
        self.dW[self.i] = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx
