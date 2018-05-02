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
import os
import random
import math
import pickle


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def c_network(images):
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE) as scope:
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[5, 5, 3, 32], name='conv1_weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name= 'conv1_biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name='conv1')

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE) as scope:
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[5, 5, 32, 32], name='conv2_weights')
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name='conv2_biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name='conv2')

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE) as scope:
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[3, 3, 32, 64], name='conv3_weights')
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='conv3_biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name='conv3')
    
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
    

    with tf.variable_scope('img_pred') as scope:
        shape = norm3.get_shape().as_list()
        reshape = tf.reshape(norm3,[-1,shape[1]*shape[2]*shape[3]])
        dim = reshape.get_shape()[1].value
        img_pred = reshape
        print('dim',dim)
        # pred is [batchsize, dim], each pred is a vector.
        # Actually, pred should be [batchsize, seqlength, dim]        
        # dim = reshape.get_shape()[1].value
        # weights = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[dim, 10], name='pred_weights')
        # biases = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32),trainable=True, name='pred_biases')
        # pred = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='pred')

    return img_pred,dim

def r_network(pred):
    num_units = 64
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    #pred should be [batchsize, seqlength, dim]
    h_val, _ = tf.nn.dynamic_rnn(lstm_cell, pred, dtype=tf.float32)
    final_output = tf.zeros(shape=[tf.shape(pred)[0], 0, 14])
    w_fc = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[num_units, 14], name='pred_weights')
    b_fc = tf.Variable(tf.constant(0.1, shape=[14], dtype=tf.float32),trainable=True, name='pred_biases')
        
    for i in np.arange(seq_length):
        temp = tf.reshape(h_val[:, i, :], [-1, num_units])
        output = tf.matmul(temp, w_fc) + b_fc
        output = tf.reshape(output, [-1, 1, 14])
        final_output = tf.concat([final_output, output], axis=1)
    return final_output

def network(x):
    dim = 7744
    pred = tf.zeros(shape=[tf.shape(x)[0], 0, dim])
    for i in range(0,10):
        img = x[:,i,:,:,:]
        img_pred,dim = c_network(img)
        img_pred = tf.reshape(img_pred, [-1, 1,dim])
        pred = tf.concat([pred, img_pred], axis=1)
    final_output = r_network(pred)
    final_output = tf.reshape(final_output, [-1, 10,7, 2])
    return final_output

def loss(predict_op, y):
    loss_l2 = tf.nn.l2_loss(predict_op - y, name='mse_loss')
    loss_l2 = loss_l2/(batch_size*10*7)
    tf.add_to_collection('losses', loss_l2)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def percent_plot(pred_record, gd_record, num):
    dist = np.zeros([num,10,7])
    dist_hist = np.zeros([7,20])
    dist_hist_sum = np.zeros([8,20])
    for j in range(0,7):
        pred_j = pred_record[:,:,j,:]
        gd_j = gd_record[:,:,j,:]
        tmp = np.square(pred_j- gd_j)
        dist[:,:,j] = np.sqrt(tmp[:,:,0] + tmp[:,:,1])
        dist_hist[j,:] = (np.histogram(dist[:,:,j], bins=20)[0])/(num*10)
        sum_val = 0
        for i in range(0,20):
            sum_val = sum_val + dist_hist[j,i]
            dist_hist_sum[j,i] = sum_val
    dist_hist_sum[7,:] = np.mean(dist_hist_sum[0:7,:],axis=0)
    return dist_hist_sum

if __name__ == '__main__':
    batch_size = 64
    learning_rate_init = 0.01
    seq_length = 10
    parser = argparse.ArgumentParser(description='Lingyu for Deep learning programming assignment 4')
    parser.add_argument('--GPUs', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--modelpath', type=str, default='saved_models_v2/')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()


    pklfile = open('youtube_train_data.pkl', 'rb')
    datafile, labelfile = pickle.load(pklfile) 
    orifile = datafile
    pklfile.close()
    datafile = datafile.astype(np.float32)

    datafile -= np.mean(datafile, axis=(2, 3, 4), keepdims=True)
    datafile /= np.std(datafile, axis=(2, 3, 4), keepdims=True)
    indices = np.random.permutation(datafile.shape[0])
    training_idx, validation_idx = indices[:7000], indices[7000:]
    train_data, test_data = datafile[training_idx,:,:,:,:], datafile[validation_idx,:,:,:,:]
    train_label, test_label = labelfile[training_idx,:,:,:], labelfile[validation_idx,:,:,:]
    print('Number of training samples', train_data.shape[0])
    print('Number of training label nodes', train_label.shape)
    print('Number of validation samples', test_data.shape[0])
    print('Number of validation label nodes', test_label.shape)
    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        x = tf.placeholder(shape=[None, 10, 64, 64, 3], dtype=tf.float32)
        y = tf.placeholder(shape=[None, 10, 7, 2], dtype=tf.float32)
        predict_op = network(x)
        total_loss = loss(predict_op,y)
        step_per_epoch = 7000 // batch_size
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate_init,global_step,decay_steps=35000,decay_rate=0.1,staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
    	train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        tf.get_collection('validation_nodes')
        tf.add_to_collection('validation_nodes', x)
        tf.add_to_collection('validation_nodes', y)
        tf.add_to_collection('validation_nodes', predict_op)
    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    plot_loss = np.zeros([args.max_epoch*int(7000/batch_size),1])
    # plot_test = np.zeros([args.max_epoch,1])
    plot_loss_epoch = np.zeros([args.max_epoch*4,1])
    plot_loss_epoch_test = np.zeros([args.max_epoch*4,1])
    plot_error_epoch = np.zeros([args.max_epoch*4,1])
    plot_error_epoch_test = np.zeros([args.max_epoch*4,1])
    indx = np.zeros([args.max_epoch*4,1])

    with tf.Session(config=config) as sess:
        training_name = 'batch:{}_lr:{}'.format(batch_size,learning_rate_init)
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())
        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        logger.info('Training Started.')
        time_started = time.time()
        initial_gs_num = sess.run(global_step)
        cnt = 0
        for e in range(args.max_epoch):
            total_batch = int(7000/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = train_data[i*batch_size:(i+1)*batch_size,:,:,:,:], train_label[i*batch_size:(i+1)*batch_size,:,:,:]
                _, gs_num, train_loss, lr_val, pred_val = sess.run([train_op, global_step, total_loss, learning_rate, predict_op],feed_dict={x: batch_x, y:batch_y})
                logger.info('epoch=%.2f step=%d, lr=%f, loss=%g' % (gs_num / step_per_epoch, gs_num, lr_val, train_loss))
                
                # record the loss
                # train_loss = np.sqrt(train_loss)
                plot_loss[e*total_batch+i] = train_loss
                print('epoch e=',e+1,'iteration i=', i, 'training loss', train_loss)
                if (i == int(total_batch/4)-1) or (i == int(2*total_batch/4)-1) or (i == int(3*total_batch/4)-1) or (i == int(4*total_batch/4)-1):
                    train_error = np.sqrt(train_loss)
                    test_loss, pred_test = sess.run([total_loss,predict_op],feed_dict={x: test_data, y: test_label})
                    test_loss = np.asarray(test_loss)
                    test_loss = test_loss*batch_size/1000
                    test_error = np.sqrt(test_loss)
                    print('validation loss', test_loss)
                    print('validation error', test_error)
                    plot_loss_epoch[cnt] = train_loss
                    plot_loss_epoch_test[cnt] = test_loss
                    plot_error_epoch[cnt] = train_error
                    print('training error', train_error)
                    plot_error_epoch_test[cnt] = test_error
                    indx[cnt] = e*total_batch+i
                    cnt = cnt+1
                   



            #saver.save(sess, os.path.join(args.modelpath, 'model'), global_step=global_step)     
        saver.save(sess, os.path.join(args.modelpath, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
    
        

    plt.figure(1)
    plt.plot(indx,plot_loss_epoch, label = "Loss on training set")
    plt.plot(indx,plot_loss_epoch_test, label = "Loss on testing set")
    plt.xlabel('Global step (iterations)')
    plt.ylabel('Loss')
    plt.title('Mean Squared loss during training, Lingyu Zhang')
    plt.legend()


    plt.figure(2)
    plt.plot(indx,plot_error_epoch, label = "Error on training set")
    plt.plot(indx,plot_error_epoch_test, label = "Error on testing set")
    plt.xlabel('Global step (iterations)')
    plt.ylabel('Distance error (Pixel)')
    plt.title('Distance error during training, Lingyu Zhang')
    plt.legend()


    dist_hist_test = percent_plot(pred_test, test_label, 1000)
    plt.figure(3)
    for j in range(0,7):
        plt.plot(dist_hist_test[j,:], label = "joint" + str(j))
    plt.plot(dist_hist_test[7,:], label = "Average over joints")
    plt.xlabel('Pixel Distance')
    plt.ylabel('Percentage')
    plt.title('Percentage of error on testing dataset, Lingyu Zhang')
    plt.legend()


    dist_hist_train = percent_plot(pred_val, batch_y, batch_size)
    plt.figure(4)
    for j in range(0,7):
        plt.plot(dist_hist_train[j,:], label = "joint" + str(j))
    plt.plot(dist_hist_train[7,:], label = "Average over joints")
    plt.xlabel('Pixel Distance')
    plt.ylabel('Percentage')
    plt.title('Percentage of error on training dataset, Lingyu Zhang')
    plt.legend()

    plt.figure(5)
    plt.plot(plot_loss, label = "Loss on training set")
    plt.xlabel('Global step (iterations)')
    plt.ylabel('Loss')
    plt.title('Mean Squared loss during training over iterations, Lingyu Zhang')
    plt.legend()

    plt.figure(6)
    img = (orifile[validation_idx[1],5,:,:,:])
    print(img.shape)
    plt.imshow(img)
    #print(test_data[1,1,:,:])
    plt.scatter(x=pred_test[1,5,:,0], y=pred_test[1,5,:,1], c='r', s=80)
    plt.scatter(x=test_label[1,5,:,0], y=test_label[1,5,:,1], c='b', s=80)
    plt.show()






