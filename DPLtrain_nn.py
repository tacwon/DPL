#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 06:43:00 2017

@author: Masao Takakuwa
"""

import sys, os
import numpy as np
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from TwoLayerNet import DPLTwoLayerNet
from common.optimizer import AdaGrad

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

(input_size,hidden_size,output_size)=(784,50,10)
learning_rate = 0.01

epoc_num = hidden_size * output_size
iters_repeat = 10000
iters_num = iters_repeat * epoc_num
batch_size = 1

train_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = t_train.shape[0]
network = DPLTwoLayerNet(input_size, hidden_size, output_size)
optimizer = AdaGrad()

def set_batch() :
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    network.set_batch(x_batch,t_batch)

set_batch()
for i in range(iters_num):
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient()
    params = network.params
    
    # 更新
    optimizer.update(params,grad)
    
    loss = network.loss()
    train_loss_list.append(loss)
    
    if i % hidden_size == 0 :
        network.set_batch(x_train,t_train)
        train_acc = network.accuracy()
        network.set_batch(x_test,t_test)
        test_acc = network.accuracy()
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        set_batch()

    
        print("(",i//hidden_size,network.i,network.i_rand[network.i],"),(",train_acc,test_acc,")")
    
    network.update_i()    
