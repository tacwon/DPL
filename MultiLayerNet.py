#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:30:32 2017

@author: Masao Takakuwa
"""
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np

from DPLlayers import DPLSigmoid,DPLRelu,Through,FirstPath,LastPath,DPLPath
from collections import OrderedDict
from common.layers import SoftmaxWithLoss
from common.gradient import numerical_gradient

class update_path :
    def __init__(self,layers) : 
        self.layers = layers
    def update(self) :
        pre_i = 0
        for layer in self.layers.values():   # set layer's path
            pre_i = layer.update(pre_i)
    def get(self):
        i = []
        for layer in self.layers.values():   # get layer's path
            i.append(layer.i)
        return i        
    
class DPLMultiLayerNet:
    """全結合による多層ニューラルネットワーク

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'　or 'through'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        ’linear'または'through'を指定した場合は、「Linearの初期値」を設定
        DLの重み：正規乱数、DPLの重み：正規乱数の絶対値
    """
    def __init__(self, input_size, hidden_size_list, output_size,batch_size=1,
                 activation='sigmoid', weight_init_std='linear',dpl='dpl'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.batch_size = batch_size
        self.dpl = dpl
        self.params = {}
        self.layers = OrderedDict()
        # 重みの初期化
        self.__init_weight(weight_init_std,dpl)
        # レイヤの生成
        self.__init_wb(activation,dpl)
        #print("layers:",self.layers)
        self.update_path = update_path(self.layers)
    
    def __init_wb(self,activation,dpl) :
        """レイヤー生成
            activation : 'relu' or 'sigmoid'　or 'through'
        """
        activation_layer = {'sigmoid': DPLSigmoid, 'relu': DPLRelu,'through':Through}

        #print("_init_wb activation",activation,"dpl",dpl)
        self.layers['Path1'] = FirstPath(self.params['W1'],self.params['b1'],self.batch_size)
        self.layers['Activation_function1'] = activation_layer[activation]()
        
        for idx in range(2, self.hidden_layer_num+1):
            #print("idx:",idx)
            self.layers['Path' + str(idx)] =DPLPath(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Path' + str(idx)] = LastPath(self.params['W' + str(idx)],
            self.params['b' + str(idx)],self.batch_size)

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std,dpl):
        """重みの初期値設定

        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
            ’linear'または'through'を指定した場合は、「Linearの初期値」を設定
            DLの重み：正規乱数、DPLの重み：正規乱数の絶対値
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        #print ("all_size_list:",all_size_list)
        for idx in range(1, len(all_size_list)):
            #print("idx:",idx)
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('linear','through'):
                scale = 1.0/all_size_list[idx - 1]           #DPLで使う場合に推奨される初期値
            rand = np.random.randn(all_size_list[idx-1], all_size_list[idx])
            #if idx == 1: scale/self.batch_size
            if dpl == 'dpl':rand = np.fabs(rand)
            self.params['W' + str(idx)] = scale * rand
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
        #print("param:",self.params)
            
    def set_batch(self,x,t) :
        self.x = x
        self.t = t
        self.batch_size = x.shape[0]

    def predict(self):
        x = self.x
        for layer in self.layers.values():
            x = layer.forward(x)
            #print("Predict x:",x.shape,"self.x",self.x.shape)
        return x
        
    def DPLpredict(self):
        self.update_path.update()
        x = self.x
        for layer in self.layers.values():
            x = layer.DPLforward(x)
            #print("DPLPredict x:",x.shape,"self.x",self.x.shape)
        return x

    def loss(self):
        """損失関数を求める
        Returns
        -------
        損失関数の値
        """
        if self.dpl == 'dpl' :
            y = self.DPLpredict()
        else:
            y = self.predict()
        #print("loss y:",y.shape,"t",self.t.shape)    

        return self.last_layer.forward(y, self.t)

    def accuracy(self):
        y = self.predict()
        y = np.argmax(y, axis=1)
        #print("accracy y:",y.shape,"t:",self.t.shape)
        if self.t.ndim != 1 : t = np.argmax(self.t, axis=1)
        accuracy = np.sum(y == t) / float(self.x.shape[0])
        return accuracy

    def numerical_gradient(self):
        """勾配を求める（数値微分）
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss()

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self):
        """勾配を求める（誤差逆伝搬法）
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss()

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        if self.dpl == 'dpl' :
            for layer in layers:
                dout = layer.DPLbackward(dout)   # Fix comfirmed "backward" was not work for DPLforward
        else:
            for layer in layers:
                dout = layer.backward(dout)   

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Path' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Path' + str(idx)].db

        return grads

