import os
mnist_path = "/data_public/mnist"
os.path.exists(mnist_path)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/")
tf.reset_default_graph()

height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")#[28*28]
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])#[n,28,28,1]
    y = tf.placeholder(tf.int32, shape=[None], name="y")#[n,1]

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,#kernel.shape=[3,3,1,32],strides=[1,1,1,1]==>[n,28,28,32]
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,#kernel.shape=[3,3,32,64],strides=[1,2,2,1]==>[n,14,14,64]
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")#==>[n,7,7,64]
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])# [n,7*7*64]

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")#[64]

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")#[10]
    Y_proba = tf.nn.softmax(logits, name="Y_proba")#每一个类别的概率

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
with tf.name_scope('summary'):
    writer = tf.summary.FileWriter('/output', tf.get_default_graph())#文件目录，要保存的计算图（默认）
    mse_summary = tf.summary.scalar('loss', loss)#计算mse 并将结果保存在当前节点
    acc_summary = tf.summary.histogram('accuracy', accuracy)#计算mse 并将结果保存在当前节点
    
    
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
n_epochs = 1000
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for batch_index in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            _,loss_summary_str,train_acc_summary_str = sess.run([training_op,mse_summary,acc_summary], feed_dict={X: X_batch, y: y_batch})
            test_acc_summary_str = sess.run(acc_summary,feed_dict={X: mnist.test.images, y: mnist.test.labels})
            step = epoch * n_batches + batch_index#计算出step
            writer.add_summary(loss_summary_str,step)
            writer.add_summary(train_acc_summary_str,step)
            writer.add_summary(test_acc_summary_str,step)
        save_path = saver.save(sess, "model/my_mnist_model")