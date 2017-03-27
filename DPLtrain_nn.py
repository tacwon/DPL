#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 06:43:00 2017

@author: Masao Takakuwa
"""

import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from TwoLayerNet import DPLTwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

(hidden_size,output_size)=(50,10)
learning_rate = 0.1

epoc_num = hidden_size * output_size
iters_repeat = 10000
iters_num = iters_repeat * epoc_num

train_loss_list = []
train_acc_list = []
test_acc_list = []

#x_batch = x_train[:3]
#t_batch = t_train[:3]
x_batch = x_train
t_batch = t_train

network = DPLTwoLayerNet(x_batch.shape[1], hidden_size, output_size)

for i in range(iters_num):
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    network.set_batch(x_batch,t_batch)
    grad = network.gradient()
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss()
    train_loss_list.append(loss)
    
    if i % output_size == 0 :
        network.set_batch(x_train,t_train)
        train_acc = network.accuracy()
        network.set_batch(x_test,t_test)
        test_acc = network.accuracy()
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
        print("(",i//output_size,network.i,network.i_rand[network.i],"),(",train_acc,test_acc,")")
    
    network.update_i()    
