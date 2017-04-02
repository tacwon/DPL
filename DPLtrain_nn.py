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
from MultiLayerNet import DPLMultiLayerNet
from common.optimizer import AdaGrad

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

(input_size,hidden_size,output_size)=(784,[50],10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 5
inter_per_epoch = max(train_size//batch_size,1)
iters_repeat = iters_num * inter_per_epoch

#epoch_num = hidden_size * output_size * inter_per_epoch
epoch_num = inter_per_epoch
iters_num = iters_repeat * epoch_num

train_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = t_train.shape[0]
network = DPLMultiLayerNet(input_size, hidden_size, output_size,batch_size)
optimizer = AdaGrad()    # very good

def set_batch() :
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    network.set_batch(x_batch,t_batch)

for i in range(iters_num):
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    set_batch()
    grad = network.gradient()
    params = network.params
    # 更新
    optimizer.update(params,grad)
    
    loss = network.loss()
    train_loss_list.append(loss)
    
    if i % epoch_num == 0 :
        network.set_batch(x_train,t_train)
        train_acc = network.accuracy()
        network.set_batch(x_test,t_test)
        test_acc = network.accuracy()
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
        print("(",i//epoch_num,network.update_path.get(),"),(",train_acc,test_acc,")")
