#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 07:29:40 2017

@author: tacwon
"""
import numpy as np
import matplotlib.pyplot as plt
from MultiLayerNet import DPLMultiLayerNet

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)

batch_size = 10000
input_size = 100
input_data = np.random.randn(batch_size, input_size)  # 1000個のデータ
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層
activations = {}  # ここにアクティベーションの結果を格納する
    # 初期値の値をいろいろ変えて実験しよう！
    #w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

DPL='dl'
network = DPLMultiLayerNet(input_size, [100,100,100,100,100], 10,
                           batch_size,activation='sigmoid', weight_init_std='he',dpl=DPL)
network.set_batch(input_data,input_data)
network.update_path.update()
iter_num = {'dl':1,'dpl':1}
file_name = 'DL_act-_he.png'
    
i = 0
x = input_data
for layer in network.layers.values() :
    if i != 0:
        x = activations[i-1]
    a = layer.forward(x)
    activations[i] = a * iter_num[DPL]
    i = i+1

# ヒストグラムを描画
for i, a in activations.items():
    if i%2 :continue
    ii = i//2
    if ii >= 5 : break
    plt.subplot(1, len(activations)//2, ii+1)
    plt.title(str(ii+1) + "-layer")
    if ii != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 20, range=(0,1))
plt.savefig(file_name)

