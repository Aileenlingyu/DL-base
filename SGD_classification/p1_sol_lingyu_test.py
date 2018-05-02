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
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
	dir_path = os.path.dirname(os.path.realpath(__file__))
	file_path = dir_path + "/data/test_data/"
	label = np.loadtxt("/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/dl/HW/programming_lingyu/P1/data/labels/test_label.txt")
	x = np.ones([785,4982])
	i = 0
	for el in sorted(glob.glob(file_path+'*.jpg')):
		print(el)
		img = mpimg.imread(el)
		x[0:784,i]=(img.reshape(784*1))/255;
		i = i+1
	W = np.load("/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/dl/HW/programming_lingyu/P1/output/multiclass_parameters.txt",encoding='ASCII')
	print(type(W),W.shape)
	y_pred = np.zeros([4982,])
	for m in range(0,4982):
		tmp = np.matmul(np.transpose(W), x[:,m])
		y_pred[m] = np.argmax(tmp)+1
	
	C_matrix = confusion_matrix(label, y_pred)
	
	print(C_matrix)