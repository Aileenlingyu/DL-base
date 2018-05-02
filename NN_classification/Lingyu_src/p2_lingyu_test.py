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
	test_file_path = dir_path + "/Prog3_data/test_data/"
	test_label = np.loadtxt(dir_path + "/Prog3_data/labels/test_label.txt")
	test_np_label = np.zeros([5000,10])
	for i in range(0,5000):
		test_ind = int(test_label[i])
		test_np_label[i,test_ind] = 1
		test_image = np.ones([5000,784])
		test_idx = 0
	for test_el in sorted(glob.glob(test_file_path+'*.jpg')):
		print(test_el)
		test_img = mpimg.imread(test_el)
		test_image[test_idx,0:784]=(test_img.reshape(784*1))/255
		test_idx = test_idx+1
	Theta = np.load("/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/dl/HW/programming_lingyu/P2/P2_lingyu_sol/output/nn_parameters.txt",encoding='ASCII')
	print(type(Theta),len(Theta))


	W1 = tf.Variable(Theta[0])
	b1 = tf.Variable(Theta[1])
	W2 = tf.Variable(Theta[2])
	b2 = tf.Variable(Theta[3])
	W3 = tf.Variable(Theta[4])
	b3 = tf.Variable(Theta[5])

	x = tf.placeholder(tf.float32, [None, 784]) # input has 784 nodes
	y = tf.placeholder(tf.float32, [None, 10]) # output has 10 nodes
	# forward propagation
	z1 = tf.add(tf.matmul(x, W1),b1)
	print('z1',z1)
	h1 = tf.nn.relu(z1)

	print('h1',h1)

	z2 = tf.add(tf.matmul(h1, W2),b2)
	print('z2,Batchx100',z2)
	h2 = tf.nn.relu(z2)

	print('h2',h2)

	z3 = tf.add(tf.matmul(h2, W3),b3)
	print('z3',z3)
	pred = tf.nn.softmax(z3)
	#pred = tf.nn.relu(z3)
	# pred = tf.nn.softmax(tf.matmul(x, W))
	print('pred',pred)
	print('y',y)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		pred_test_val = sess.run([pred], feed_dict={x: test_image, y: test_np_label})
		print(len(pred_test_val))
		print('pred',pred_test_val[0].shape)
		print('gd', test_np_label.shape)
		y_pred = np.argmax(pred_test_val[0], axis = 1)
		print('pred', y_pred.shape)
		label = np.argmax(test_np_label, axis = 1)
		print('gd', label.shape)
		C_matrix = confusion_matrix(label, y_pred)
		print(C_matrix)
		print(tf.confusion_matrix(
    label,
    y_pred,
    num_classes=10,
    dtype=tf.int32))

	  






