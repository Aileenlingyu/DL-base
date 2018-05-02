import tensorflow as tf
import os
import numpy as np
import glob
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib
from PIL.Image import core as image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import random
import math
import pickle


# def sigma(h):
# 	return tf.nn.softmax(h)
    
def sigmader(z_bef):
    
	# print('x',x)
	z = tf.nn.softmax(z_bef)
	x_re1 = tf.expand_dims(z, 2)
	x_re2 = tf.transpose(x_re1, perm=[0, 2, 1])
	# print('x_re1',x_re1)
	# print('x_re2',x_re2)
	tmp1 = tf.matmul(x_re1,x_re2)
	#print('tmp1',tmp1)
	tmp2 = tf.matrix_diag(z)
	#print('tmp2',tmp2)
	der_sigma= tmp2 - tmp1

	# print('BatchxKxK,der_sigma',der_sigma)
	return der_sigma

# def sigmader(z):
# 	# print('x',x)
# 	tmp1 = tf.nn.relu(z)
# 	tmp2 = tf.sign(tmp1)
# 	# tmp1 = tf.sign(z)
# 	# tmp2 = tf.nn.relu(tmp1)
# 	# print('tmp1',tmp1)
# 	# print('tmp2',tmp2)
# 	der_relu= tf.matrix_diag(tmp2)

# 	# print('Batchx100x100,der_relu',der_relu)
# 	return der_relu
	
def reluder(z):
	# print('x',x)
	tmp1 = tf.nn.relu(z)
	tmp2 = tf.sign(tmp1)
	# tmp1 = tf.sign(z)
	# tmp2 = tf.nn.relu(tmp1)
	# print('tmp1',tmp1)
	# print('tmp2',tmp2)
	der_relu= tf.matrix_diag(tmp2)

	# print('Batchx100x100,der_relu',der_relu)
	return der_relu
	

	
# load data
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = dir_path + "/Prog3_data/train_data/"
label = np.loadtxt(dir_path + "/Prog3_data/labels/train_label.txt")
train_label = np.zeros([50000,10])
for i in range(0,50000):
    ind = int(label[i])
    train_label[i,ind] = 1
train_image = np.ones([50000,784])
idx = 0
for el in sorted(glob.glob(file_path+'*.jpg')):
    print(el)
    img = mpimg.imread(el)
    train_image[idx,0:784]=(img.reshape(784*1))/255
    idx = idx+1

#########################
##########testing data
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
###########################


# Parameters
learning_rate = 0.2
training_epochs = 1
batch_size = 50
display_step = 1

# define the nodes
x = tf.placeholder(tf.float32, [None, 784]) # input has 784 nodes
y = tf.placeholder(tf.float32, [None, 10]) # output has 10 nodes


# define the parameters
W1 = tf.Variable(tf.random_normal([784, 100],mean=0.0,stddev=0.1))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.random_normal([100, 100],mean=0.0,stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.random_normal([100, 10],mean=0.0,stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))




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



#Back propagation

# W_grad =  - tf.matmul ( tf.transpose(x) , y - pred) 
# b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)
# new_W = W.assign(W - learning_rate * W_grad)
# new_b = b.assign(b - learning_rate * b_grad)


y_grad = pred - y
print('y_grad,Batchx10',y_grad)
y_grad_exd = tf.expand_dims(y_grad, 2)
print('y_grad_exd,Batchx10x1',y_grad_exd)
#define the loss function
loss = tf.reduce_mean(tf.matmul(tf.transpose(y_grad_exd,perm=[0, 2, 1]),y_grad_exd), axis=[0])





z3_grad = tf.matmul(sigmader(z3),y_grad_exd)
print('z3_grad,Batchx10x1',z3_grad)
b3_grad = tf.reshape(z3_grad,[50,10])
print('b3_grad,Batchx10',b3_grad)
# W3_grad = tf.matmul(tf.expand_dims(h2, 2), z3_grad)

#######################
#W3_grad
s_3_tmp = tf.reshape(sigmader(z3),[50,1,100])
h2_tmp = tf.reshape(h2,[50,100,1])
mut_tmp = tf.matmul(h2_tmp,s_3_tmp)
W3_grad_tmp = tf.reshape(mut_tmp,[50,1000,10])
W3_grad = tf.reshape(tf.matmul(W3_grad_tmp,y_grad_exd),[50,100,10])
#######################





print('W3_grad,Batchx100x10',W3_grad)
W3_expand_tmp = tf.expand_dims(W3,0)
print('W3_expand_tmp',W3_expand_tmp)
h2_grad = tf.matmul(tf.multiply(tf.ones([50,1,1]),W3_expand_tmp),z3_grad)
print('h2_grad, Batchx100x1', h2_grad)






print('z2_relu, Batchx100x100',reluder(z2))
z2_grad = tf.matmul(reluder(z2),h2_grad)
print('z2_grad,Batchx100x1',z2_grad)
b2_grad = tf.reshape(z2_grad,[50,100])
print('b2_grad,Batchx100',b2_grad)
#W2_grad = tf.matmul(tf.expand_dims(h1, 2), z2_grad)

#####################################
###W2_grad
s_2_tmp = tf.reshape(reluder(z2),[50,1,10000])
h1_tmp = tf.reshape(h1,[50,100,1])
mut_2_tmp = tf.matmul(h1_tmp,s_2_tmp)
W2_grad_tmp = tf.reshape(mut_2_tmp,[50,10000,100])
W2_grad = tf.reshape(tf.matmul(W2_grad_tmp,h2_grad),[50,100,100])
##################################



print('W2_grad,Batchx100x100',W2_grad)
W2_expand_tmp = tf.expand_dims(W2,0)
h1_grad = tf.matmul(tf.multiply(tf.ones([50,1,1]),W2_expand_tmp),z2_grad)
print('h1_grad, Batchx100x1', h1_grad)



z1_grad = tf.matmul(reluder(z1),h1_grad)
print('z1_grad,Batchx100x1',z1_grad)
b1_grad = tf.reshape(z1_grad,[50,100])
#W1_grad = tf.matmul(tf.expand_dims(x, 2), z1_grad)

#####################################
###W1_grad
s_1_tmp = tf.reshape(reluder(z1),[50,1,10000])
x_tmp = tf.reshape(x,[50,784,1])
mut_3_tmp = tf.matmul(x_tmp,s_1_tmp)
W1_grad_tmp = tf.reshape(mut_3_tmp,[50,78400,100])
W1_grad = tf.reshape(tf.matmul(W1_grad_tmp,h1_grad),[50,784,100])
##################################





print('W1_grad,Batchx784x100',W1_grad)



new_W3 = W3.assign(W3 - learning_rate * tf.reduce_mean(W3_grad, axis=[0]))
new_b3 = b3.assign(b3 - learning_rate * tf.reduce_mean(b3_grad, axis=[0]))
new_W2 = W2.assign(W2 - learning_rate * tf.reduce_mean(W2_grad, axis=[0]))
new_b2 = b2.assign(b2 - learning_rate * tf.reduce_mean(b2_grad, axis=[0]))
new_W1 = W1.assign(W1 - learning_rate * tf.reduce_mean(W1_grad, axis=[0]))
new_b1 = b1.assign(b1 - learning_rate * tf.reduce_mean(b1_grad, axis=[0]))

plot_accu = np.zeros([int(50000/batch_size),1])
plot_loss = np.zeros([int(50000/batch_size),1])
plot_test = np.zeros([int(50000/batch_size),1])

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

    # Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(50000/batch_size)
        
        # Loop over all batches
		for i in range(total_batch):
			# batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			batch_xs, batch_ys = train_image[i*batch_size:(i+1)*batch_size,:], train_label[i*batch_size:(i+1)*batch_size,:]

			# Fit training using batch data

			W3_latest, b3_latest, W2_latest, b2_latest, W1_latest, b1_latest, c,pred_val= sess.run([new_W3, new_b3, new_W2, new_b2, new_W1, new_b1, loss,pred], feed_dict={x: batch_xs, y: batch_ys})
			# Compute average loss
			avg_cost += c / total_batch
			# record the loss
			plot_loss[i] = c
			print('iteration i', i, 'iteration cost', plot_loss[i])
			print(np.argmax(pred_val, axis = 1), np.argmax(batch_ys, axis = 1))
			# record the prediction accuracy
			correct_prediction_iter = tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y, axis = 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction_iter, tf.float32))
			plot_accu[i] = accuracy.eval({x: batch_xs, y: batch_ys})
			print('iteration i', i, 'training accuracy', plot_accu[i])
			# record the prediction accuracy of testing set
			#test_batch_xs, test_batch_ys = test_image[i*batch_size:(i+1)*batch_size,:], train_label[i*batch_size:(i+1)*batch_size,:]

			plot_test[i] = accuracy.eval({x: test_image, y: test_np_label})
			print('iteration i', i, 'testing accuracy', plot_test[i])
			





        # Display logs per epoch step
		if (epoch+1) % display_step == 0:
			print ("Epoch:", '%04d' % (epoch+1), "cost=", avg_cost)

	print ("Optimization Finished!")
	filehandler = open(dir_path + "/output/nn_parameters.txt","wb")
	Theta = [W1_latest, b1_latest, W2_latest, b2_latest, W3_latest, b3_latest]
	pickle.dump(Theta, filehandler, protocol=2)
	filehandler.close()

	plt.figure(1)
	plt.plot(plot_loss, label = "loss")
	plt.xlabel('iteration')
	plt.ylabel('loss during training')
	plt.title('Loss during training, Lingyu Zhang')


	plt.figure(2)
	plt.plot(plot_accu, label = "training accuracy")
	plt.plot(plot_test, label = "testing accuracy")
	plt.xlabel('Iteration')
	plt.ylabel('Accuracy')
	plt.title('Accuracy during training, Lingyu Zhang')
	plt.legend()

	plt.figure(3)
	plt.plot(1-plot_accu, label = "training error")
	plt.plot(1-plot_test, label = "testing error")
	plt.xlabel('Iteration')
	plt.ylabel('Classification error')
	plt.title('Classification error during training, Lingyu Zhang')
	plt.legend()
	plt.show()


	






