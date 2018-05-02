import tensorflow as tf 
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib
from PIL.Image import core as image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import random
import math


def softmax(tmp,k):
    e_x = np.exp(tmp)
    score = e_x[k] / e_x.sum(axis=0) 
    return score



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = dir_path + "/data/train_data/"
    learningrate = 0.01
    lambda_reg = 0.00001
    W = 0.0001*np.ones([785,5])
    label = np.loadtxt("/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/dl/HW/programming_lingyu/P1/data/labels/train_label.txt")
    y = np.zeros([25112,5])
    for i in range(0,25112):
        ind = int(label[i]-1)
        y[i,ind] = 1
    x = np.ones([785,25112])
    i = 0
    for el in sorted(glob.glob(file_path+'*.jpg')):
        print(el)
        img = mpimg.imread(el)
        x[0:784,i]=(img.reshape(784*1))/255;
        i = i+1

    file_path_test = dir_path + "/data/test_data/"
    label_test = np.loadtxt("/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/dl/HW/programming_lingyu/P1/data/labels/test_label.txt")
    y_test = np.zeros([4982,5])
    for i in range(0,4982):
        ind = int(label[i]-1)
        y_test[i,ind] = 1
    x_test = np.ones([785,4982])
    i = 0
    for el_test in sorted(glob.glob(file_path_test+'*.jpg')):
        print(el_test)
        img_test = mpimg.imread(el_test)
        x_test[0:784,i]=(img_test.reshape(784*1))/255;
        i = i+1


    batch_size = 100
    max_iter = 5000
    loss = np.zeros([max_iter,5])
    loss_test = np.zeros([max_iter,5])
    for k in range(0,5):
        iter_num = 0
        while(iter_num<max_iter):
            theta = W[:,k]
            batch = random.sample(range(0, 25112), batch_size)
            # for batch in batch_array_list: # We have multiple batches, each time we randomly select a batch
            for m in batch: # Within one batch, we do the update
                delta = np.zeros([785,])
                tmp = np.matmul(np.transpose(W), x[:,m])
                h = softmax(tmp,k)
                delta = delta + x[:,m]*(y[m,k]-h)
            W[:,k] = W[:,k] - learningrate*(-delta+lambda_reg*2*W[:,k])
            for m in batch:
                print("training loss",'iteration',iter_num,'class', k)
                loss[iter_num,k] = loss[iter_num,k] + y[m,k]*(np.matmul(np.transpose(W[:,k]), x[:,m])-math.log(np.exp(np.matmul(np.transpose(W), x[:,m])).sum(axis=0) ))
            # for m in range(0,4982):
            #     print("testing loss", 'iteration',iter_num,'class', k)
            #     loss_test[iter_num,k] = loss_test[iter_num,k] + y_test[m,k]*(np.matmul(np.transpose(W[:,k]), x_test[:,m])-math.log(np.exp(np.matmul(np.transpose(W), x_test[:,m])).sum(axis=0) ))
            # # theta = theta + learningrate*delta # Within one batch, we do the update
            # W[:,k] = theta
            iter_num = iter_num + 1
    print("W",W.shape, W)
    filehandler = open("/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/dl/HW/programming_lingyu/P1/output/multiclass_parameters.txt","wb")
    pickle.dump(W, filehandler)
    filehandler.close()
    
    for k in range(0,5):
        W_k = W[0:784,k]
        Img = W_k.reshape(28,28)
        plt.imshow(Img)
        plt.colorbar()
        plt.show()

    overall_loss = -(loss[:,0] + loss[:,1] +loss[:,2]+loss[:,3]+loss[:,4])
    plt.plot(overall_loss)
    plt.ylabel('training loss during the iteration')
    plt.show()
    
    # overall_loss_test = -(loss_test[:,0] + loss_test[:,1] +loss_test[:,2]+loss_test[:,3]+loss_test[:,4])
    # plt.plot(overall_loss_test)
    # plt.ylabel('testing loss during the iteration')
    # plt.show()








