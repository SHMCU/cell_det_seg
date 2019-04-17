from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.ops import losses_ops
from datashape.coretypes import uint8
import glob


def set_lr_by_var_name(grads_and_vars, target_vars, target_lr): # set different learning rate to individual variables, can be used to set different lr to different layers
    # grads_and_vars: a list of tuples, it is the returned value by tf.train.MomentumOptimizer.compute_gradients()
    # target_vars: a list of variable names, for example ['conv5', 'conv4']
    # target_lr: a list of desired learning rate multipliers for the target_vars, for example [0.1, 5]
    
    # return a list of tuples. It has the same structure and format as grads_and_vars
    new_grads_and_vars = []
    for i in grads_and_vars:
        for j in range(len(target_vars)):
            if i[1].name.find(target_vars[j])>-1:
                new_grads_and_vars.append((i[0]*target_lr[j], i[1]))
            else:
                new_grads_and_vars.append(i)
    return new_grads_and_vars
                

def getPretrainedModel(myModelParams,sess, modelName='../hedPreTrainedModel/HED_reproduced.npy'):
    preTrainedModel = np.load(modelName)
    paramNames = preTrainedModel.item().keys()
    for hedName in paramNames:
        id = hedName.find('/')
        hedName_layerName = hedName[:id]
        
        for myParam in myModelParams:
            
            if myParam.name.find(hedName_layerName)>-1 and myParam.name.find('_wt')>-1 and hedName.find('/W')>-1:
                valuePlaceHolder = tf.placeholder(tf.float32, shape=preTrainedModel.item()[hedName].shape)
                assign_op = myParam.assign(valuePlaceHolder)
                sess.run(assign_op, feed_dict={valuePlaceHolder:preTrainedModel.item()[hedName]})
                break
            elif myParam.name.find(hedName_layerName)>-1 and myParam.name.find('_b')>-1 and hedName.find('/b')>-1:
                valuePlaceHolder = tf.placeholder(tf.float32, shape=preTrainedModel.item()[hedName].shape)
                assign_op = myParam.assign(valuePlaceHolder)
                sess.run(assign_op, feed_dict={valuePlaceHolder:preTrainedModel.item()[hedName]})
                break
    
    valuePlaceHolder = tf.placeholder(tf.float32, shape=[1])
    assign_op = myModelParams[-1].assign(valuePlaceHolder)
    sess.run(assign_op, feed_dict = {valuePlaceHolder: [0]})
    return

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
      A weight decay is added only if one is specified.
      Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
      Returns:
        Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def convLayer(inputFeat, name, kernel_shape, stride_shape, actFun='elu'):
    WName = name + '_wt'
    BName = name + '_b'
    preactName = name + 'preact'
    actName = name + '_act'
    weight = tf.get_variable(name=WName, shape=kernel_shape, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    conv1_1 = tf.nn.conv2d(inputFeat, weight, stride_shape, padding='SAME', name='conv1_1')
    bias1_1 = tf.get_variable(name=BName, shape=kernel_shape[-1], initializer=tf.constant_initializer(value=0.0))
    pre_activation = tf.nn.bias_add(conv1_1, bias1_1, name=preactName)
    if actFun == 'elu':
        layerAct=tf.nn.elu(pre_activation, name = actName)
    elif actFun == 'identity':
        layerAct=tf.identity(pre_activation, name = actName)
    elif actFun == 'sigmoid':
        layerAct=tf.sigmoid(pre_activation, name = actName)
        
    return layerAct

def convLayer_upsample(inputFeat, name, kernel_shape, stride_shape, up_ratio, actFun='elu'):
    WName = name + '_wt'
    BName = name + '_b'
    preactName = name + 'preact'
    actName = name + '_act'
    weight = tf.get_variable(name=WName, shape=kernel_shape, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    conv1_1 = tf.nn.conv2d(inputFeat, weight, stride_shape, padding='SAME', name='conv1_1')
    bias1_1 = tf.get_variable(name=BName, shape=kernel_shape[-1], initializer=tf.constant_initializer(value=0.0))
    pre_activation = tf.nn.bias_add(conv1_1, bias1_1, name=preactName)
    if actFun == 'elu':
        layerAct=tf.nn.elu(pre_activation, name = actName)
    elif actFun == 'identity':
        layerAct=tf.identity(pre_activation, name = actName)
    elif actFun == 'sigmoid':
        layerAct = tf.sigmoid(pre_activation, name = actName)
        
    while up_ratio!=1:
        featMapShape = layerAct.shape
        layerAct = tf.image.resize_bilinear(layerAct, size=[2*featMapShape[1].value, 2*featMapShape[2].value])
        up_ratio = up_ratio/2
    return layerAct

def fullConnLayer(inputFeat, name, num_unit, stddev=0.04, wd=0.004):
    WName = name+'_wt'
    BName = name + '_b'
    actName = name + '_act'
    tmp = inputFeat[-1]
    tmp = tf.reshape(tmp, [-1])
    
    dim = tmp.get_shape()[0].value
    reshape = tf.reshape(inputFeat, [-1, dim])
    weights = _variable_with_weight_decay(WName, shape=[dim,num_unit], stddev=stddev, wd=wd)
    biases = tf.get_variable(name=BName, shape=num_unit, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    layerAct = tf.identity(tf.matmul(reshape, weights) + biases, name=actName)
    return layerAct

