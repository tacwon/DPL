#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:44:59 2017

@author: tacwon

"""
import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

from common.util import smooth_curve
from MultiLayerNet import DPLMultiLayerNet
from common.optimizer import SGD


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
DPL='dl'
file_name = 'DL-w-i-comp.png'
iter_type= {'dl':1,'dpl': 100} 
iter_scale = iter_type[DPL]
max_iterations = 2000


# 1:実験の設定==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}

for key, weight_type in weight_init_types.items():
    networks[key] = DPLMultiLayerNet(784, [100], 10,
            batch_size,activation='sigmoid',weight_init_std=weight_type,dpl=DPL)
    train_loss[key] = []


# 2:訓練の開始==========

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        networks[key].set_batch(x_batch,t_batch)
        grads = networks[key].gradient()
        optimizer.update(networks[key].params, grads)
    
        for ii in range(iter_scale):
            loss = networks[key].loss()
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            networks[key].set_batch(x_batch,t_batch)
            loss = networks[key].loss()
            print(key + ":" + str(loss))


# 3.グラフの描画==========
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.savefig(file_name)
