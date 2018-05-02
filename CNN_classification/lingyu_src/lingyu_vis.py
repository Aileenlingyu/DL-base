import tensorflow as tf
import pickle
import numpy as np
import argparse
import logging
import os
import time
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib
from PIL.Image import core as image
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import math
from lingyu_train import preprocess, network
from sklearn.metrics import confusion_matrix
import argparse
from io import BytesIO
# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'conv1'
channel = 0 
# start with a gray image with a little noise
img_noise = np.random.uniform(size=(1,32,32,3)) 

def showarray(a,channel):
    a = np.uint8(np.clip(a, 0, 1)*255)
    print('a',a.shape)
    f = BytesIO()
    im = Image.fromarray(a[0,:,:,:])
    im.save('filter_vis/conv1_'+str(channel)+'.png')
    
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def render_naive(layer, channel, img0=img_noise, iter_n=20, step=1.0):
    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 
    saver = tf.train.import_meta_graph(args.checkpoint+'model-6000.meta')
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
    graph = tf.get_default_graph()
    t_input = graph.get_tensor_by_name("inputs:0")
    for channel in range(0,32):
        t_obj = (graph.get_tensor_by_name("%s:0"%layer))[:,:,:,channel]
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
            
        img = img0.copy()
        for i in range(iter_n):
            g, score = sess.run([t_grad, t_score], {t_input:img})
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
            print(score, end = ' ')
        showarray(visstd(img),channel=channel)
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Lingyu for Deep learning programming assignment 3')
    parser.add_argument('--GPUs', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    render_naive(layer, channel)




