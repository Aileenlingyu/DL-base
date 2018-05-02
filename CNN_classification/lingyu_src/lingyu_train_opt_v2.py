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



def preprocess(pklfile):
	data_file = open(pklfile, "rb")
	train_x, train_y, test_x, test_y = pickle.load(data_file,encoding='bytes')
	data_file.close()
	train_x = train_x.astype(np.float32)
	test_x = test_x.astype(np.float32)
	train_x = train_x/255
	test_x = test_x/255
	train_x = train_x - np.mean(train_x,axis=0)
	test_x = test_x - np.mean(test_x,axis=0)
	print(np.mean(train_x,axis=0).shape)
	return train_x, train_y, test_x, test_y

def network(images):
    with tf.name_scope('conv1') as scope:
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[5, 5, 3, 32], name='conv1_weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name= 'conv1_biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

    with tf.name_scope('conv2') as scope:
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[5, 5, 32, 32], name='conv2_weights')
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name='conv2_biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

    with tf.name_scope('conv3') as scope:
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[3, 3, 32, 64], name='conv3_weights')
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='conv3_biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)


    #dropout = tf.nn.dropout(conv3,0.5)       
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
    #pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding='VALID', name='pool3')


    with tf.variable_scope('pred') as scope:
        shape = norm3.get_shape().as_list()
        reshape = tf.reshape(norm3,[-1,shape[1]*shape[2]*shape[3]])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[dim, 10], name='pred_weights')
        biases = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32),trainable=True, name='pred_biases')
        pred = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='pred')

    return pred

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Lingyu for Deep learning programming assignment 3')
    parser.add_argument('--GPUs', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--modelpath', type=str, default='saved_models/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--lr', type=str, default='0.000001')
    args = parser.parse_args()
    batch_size = args.batchsize
    learning_rate_init = float(args.lr)
    pklfile = "cifar_10_tf_train_test.pkl"
    train_x, train_y, test_x, test_y = preprocess(pklfile)
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    print('Number of training samples', train_x.shape[0])
    print('Number of training output nodes', train_y.shape)
    print('Number of testing samples', test_x.shape[0])
    print('Number of testing output nodes', train_y.shape)
    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32,name='inputs')
        y = tf.placeholder(shape=[None], dtype=tf.int64)      
    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    #         with tf.variable_scope(tf.get_variable_scope()):
    #             pred = network(x)

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        pred = network(x)
        total_loss = loss(pred,y)
        step_per_epoch = 50000 // batch_size
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate_init,global_step,decay_steps=3500,decay_rate=0.1,staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
    	train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)

    # valid_loss = tf.placeholder(tf.float32, shape=[])
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        tf.get_collection('validation_nodes')
        tf.add_to_collection('validation_nodes', x)
        tf.add_to_collection('validation_nodes', pred)
    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    plot_loss = np.zeros([args.max_epoch*int(50000/batch_size),1])
    plot_accu = np.zeros([args.max_epoch,1])
    plot_test = np.zeros([args.max_epoch,1])
    plot_loss_epoch = np.zeros([args.max_epoch,1])

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
        for e in range(args.max_epoch):
            total_batch = int(50000/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = train_x[i*batch_size:(i+1)*batch_size,:,:,:], train_y[i*batch_size:(i+1)*batch_size]
                _, gs_num, train_loss, lr_val, pred_val = sess.run([train_op, global_step, total_loss, learning_rate, pred],feed_dict={x: batch_x, y:batch_y})
                logger.info('epoch=%.2f step=%d, lr=%f, loss=%g' % (gs_num / step_per_epoch, gs_num, lr_val, train_loss))
                
                # record the loss
                plot_loss[e*total_batch+i] = train_loss
                print('epoch e=',e,'iteration i=', i, 'iteration cost', plot_loss[e*total_batch+i])
                print(np.argmax(pred_val, axis = 1), batch_y)
            # For each epoch record the prediction accuracy
            correct_prediction_iter = tf.equal(tf.argmax(pred, axis = 1), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction_iter, tf.float32))
            plot_accu[e] = accuracy.eval({x: batch_x, y: batch_y})
            print('epoch e=', e, 'training accuracy', plot_accu[e])
            plot_test[e] = accuracy.eval({x: test_x, y: test_y})
            print('epoch e=', e, 'testing accuracy', plot_test[e])
            plot_loss_epoch[e] = train_loss


            saver.save(sess, os.path.join(args.modelpath, 'model'), global_step=global_step)     
        saver.save(sess, os.path.join(args.modelpath, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
    plt.figure(1)
    plt.plot(plot_loss, label = "loss")
    plt.xlabel('iteration')
    plt.ylabel('loss during training')
    plt.title('Loss during training, Lingyu Zhang')


    plt.figure(2)
    plt.plot(plot_accu, label = "training accuracy")
    plt.plot(plot_test, label = "testing accuracy")
    plt.plot(plot_loss_epoch, label = "Cost during training")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy or cost')
    plt.title('Accuracy and cost during training, Lingyu Zhang')
    plt.legend()

    plt.figure(3)
    plt.plot(1-plot_accu, label = "training error")
    plt.plot(1-plot_test, label = "testing error")
    plt.xlabel('Iteration')
    plt.ylabel('Classification error')
    plt.title('Classification error during training, Lingyu Zhang')
    plt.legend()
    plt.show()






