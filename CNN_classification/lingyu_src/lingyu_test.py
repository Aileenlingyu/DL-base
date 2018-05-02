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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import math
from lingyu_train import preprocess, network
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Lingyu for Deep learning programming assignment 3')
	parser.add_argument('--GPUs', type=int, default=1)
	parser.add_argument('--checkpoint', type=str, default='')
	args = parser.parse_args()

	pklfile = "cifar_10_tf_train_test.pkl"
	train_x, train_y, test_x, test_y = preprocess(pklfile)
	test_y = np.asarray(test_y)
	with tf.device('/gpu:0'):
		sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 
		saver = tf.train.import_meta_graph(args.checkpoint+'model-6000.meta')
		saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
		graph = tf.get_default_graph()
		optim_vars = tf.get_collection('validation_nodes')
		print([v.name for v in optim_vars])
		x = graph.get_tensor_by_name("inputs:0")
		pred = graph.get_tensor_by_name("pred/pred:0")

		pred_test_val = sess.run([pred], feed_dict={x: test_x})
	print(len(pred_test_val))
	print('pred',pred_test_val[0].shape)
	print('gd', test_y.shape)
	y_pred = np.argmax(pred_test_val[0], axis = 1)
	print('pred', y_pred.shape)
	label = test_y
	print('gd', label.shape)
	C_matrix = confusion_matrix(label, y_pred)
	print(C_matrix)