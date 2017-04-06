#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 06:43:00 2017

@author: Masao Takakuwa
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from MultiLayerNet import DPLMultiLayerNet
from common.optimizer import AdaGrad,Adam,SGD

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#x_train = x_train[:300]
#t_train = t_train[:300]
(input_size,hidden_size,output_size)=(784,[100],10)
DPL = 'dpl'
w_d_l = 0
file_name = 'DPLs-[100].png'
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
inter_per_epoch = max(train_size//(batch_size),1)
iters_repeat = iters_num * inter_per_epoch

epoch_num = 1
if (DPL == 'dpl') :
    epoch_num = epoch_num * 100
iters_num = iters_repeat * epoch_num

train_loss_list = []
train_acc_list = []
test_acc_list = []
ratio_list = []

train_size = t_train.shape[0]

# sigmoid better than Relu for DPL

network = DPLMultiLayerNet(input_size, hidden_size, output_size,batch_size,
                           activation='sigmoid',dpl=DPL)
optimizer = AdaGrad()    # very good
#optimizer = Adam()
#optimizer = SGD()

def set_batch() :
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    network.set_batch(x_batch,t_batch)

epoch_cnt = 0
max_epochs = 201
for i in range(iters_num):
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    set_batch()
    grad = network.gradient()
    params = network.params
    # 更新
    optimizer.update(params,grad)
    
    loss = network.loss()
    #train_loss_list.append(loss)
    
    if i % epoch_num == 0 :
        network.set_batch(x_train,t_train)
        train_acc = network.accuracy()
        network.set_batch(x_test,t_test)
        test_acc = network.accuracy()
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        ratio_list.append(test_acc/train_acc*0.5)
    
        if DPL == 'dpl':
            print("(",i//epoch_num,network.update_path.get(),"),(",train_acc,test_acc,")")
        else:
            print("(",i//epoch_num,"),(",train_acc,test_acc,")")

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 3.グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.plot(x, ratio_list, marker='x', label='ratio', markevery=10)

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
#plt.show()
plt.savefig(file_name)
