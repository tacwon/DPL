#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:48:03 2017

@author: Masao Takakuwa
"""
import numpy as np
        
class PathAffine:
    def __init__(self,w,b):
        self.W = w             # 784,50 (in,out)
        self.b = b             # 50     (out)
        self.i = None
        self.x = None         # N*784
        self.dW = np.zeros_like(w) #50,784
        self.db = np.zeros_like(b) #50
        
    def init_Affine(self,x,out):
        self.x = x                  # N*784
        self.out = out              # N*50
        
    def forward(self,i):
        W = self.W[:,i:i+1]
#       print("FirstAffine_Forward x,W,b,W,out", self.x.shape,self.W.shape,self.b.shape,W.shape,self.out.shape)
    
        self.out[self.i] = np.dot(self.x,W)+self.b[self.i]       
        return self.out
    def backward(self,dout):    #dout:N*50
#        print("FirstAffine_Backward dout,W,x", dout.shape,self.W.shape,self.x.shape)
        
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        
        return dx

class Affine:
    def __init__(self,w,b):
        self.W = w             # 50,10 (in,out)
        self.b = b             # 10   (out)
        self.x = None          # N*50
        self.dW = None         #50,10
        self.db = None             #10
 #       print("LastAffine_init",self.W.shape,self.b.shape)
    
    def forward(self,x):
        self.x = x               #N*50
        out = np.dot(self.x,self.W)+self.b
        return out
    
    def backward(self,dout):    #dout:N,10
        dx = np.dot(dout,self.W.T)
 #       print("backward dout,W,dx,x",dout.shape,self.W.shape,dx.shape,self.x.shape)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
 #       print("d db",self.dW.shape,self.db.shape)
        return dx
    
        