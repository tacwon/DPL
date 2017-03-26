#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:48:03 2017

@author: Masao Takakuwa
"""
import numpy as np
        
class FirstAffine:
    def __init__(self,w,b):
        self.W = w             # 50,784 (out,in)
        self.b = b             # 50     (out)
        self.i = None
        self.x = None         # N*784
        self.dW = np.zeros_like(w) #50,784
        self.db = np.zeros_like(b) #50
    def init_Affine(self,dx,i):
        self.dx = dx  # N*50
        self.i = i
        
    def forward(self,x,i):
        self.x = x
        self.i = i             # output index
        out = np.dot(x,self.W[i])+self.b[i]       #DPL
    #    out = np.dot(x,self.W.T)+self.b              #original
        
 #       print("FirstAffine_Forward x,W,out", x.shape,self.W.shape,out.shape)
        return out
    def backward(self,dout):    #dout:N
#        print("FirstAffine_Backward dout,W,x", dout.shape,self.W.shape,self.x.shape)
        
        dx = np.outer(dout,self.W[self.i])
        self.dW[self.i] = np.dot(self.x.T,dout)
        self.db[self.i] = np.sum(dout,axis=0)
        
#        dx = np.dot(dout,self.W)
#       self.dW = np.dot(self.x.T,dout)
#        self.db = np.sum(dout,axis=0)
        return dx

class LastAffine:
    def __init__(self,w,b):
        self.W = w             # 50,10 (in,out)
        self.b = b             # 10   (out)
        self.i = None
        self.x = None          # N
        self.dW = np.zeros_like(w) #50,10
        self.db = None             #10
 #       print("LastAffine_init",self.W.shape,self.b.shape)
    
    def forward(self,x,i):
        self.x = x
        self.i = i         # input index 
        out = np.outer(self.x,self.W[i])+self.b
        return out
    
    def backward(self,dout):    #dout:10
        dx = np.dot(dout,self.W[self.i])
 #       print("backward dout,W,dx,x",dout.shape,self.W.shape,dx.shape,self.x.shape)
        self.dW[self.i] = np.dot(self.x,dout)
        self.db = np.sum(dout,axis=0)
 #       print("d db",self.dW.shape,self.db.shape)
        return dx
    
# PathAffineは、デバッグされていないコードです。
class PathAffine:
    def __init__(self,w,b):
        self.W = w             # 50,100 (in,out)
        self.b = b             # 100    (out)
        self.i = None
        self.j = None
        self.x = None          # N
        self.dW = np.zeros_like(w) #100,50
        self.db = np.zeros_like(b) #100
    
    def forward(self,x,i,j):
        self.x = x
        self.i = i         # input index
        self.j = j         # output index
        out = np.dot(x,self.W[i][j])+self.b[j]
        return out
    
    def backward(self,dout):    #dout:1
        dx = dout*self.W[self.i][self.j]
        self.dW[self.i][self.j] = self.x*dout
        self.db[self.j] = dout
        return dx        
        