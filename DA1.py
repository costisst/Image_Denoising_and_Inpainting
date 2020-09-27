# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:21:33 2017

@author: c.stefanopoulos
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
 
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
 
def Denosing_Autoencoder_1(num_epochs,np_clean,np_cor):
#    num_epochs = 10
    #    im_list = []
    #    key_list = []
    #    cor_im_list = []
    #    cor_key_list = []
    #klaiw = []
    collect_hy = []
    collect_hx = []
    #   LOADING ALL THE IMAGE SETS
#    clean_set = []
#    clean_set = glob.glob('/home/costis/mathenw_tensorflow/peiramata/peirama_lifeordeath/cifar_10/*.jpg')
#    
#    corrupted_set = []
#    corrupted_set = glob.glob('/home/costis/mathenw_tensorflow/peiramata/peirama_lifeordeath/1024_cifars/*.jpg') 
    
#    np_clean = []
#    np_cor = []
#    for path1, path2 in zip(clean_set,corrupted_set):
#        img1 = Image.open( path1 )
#        data1 = np.asarray( img1, dtype="float32" )
#        data1 = data1.reshape(1,-1)
#        np_clean.append(data1)
#        img2 = Image.open( path2 )
#        data2 = np.asarray( img2, dtype="float32" )
#        data2 = data2.reshape(1,-1)
#        np_cor.append(data2)
    
    #loading data to queue
    #clean_queue = tf.train.string_input_producer(clean_set, shuffle=False)
    #cor_queue = tf.train.string_input_producer(corrupted_set, shuffle=False)
    # 
    #reader = tf.WholeFileReader()
    #key, value = reader.read(clean_queue)
    #cor_key, cor_value = reader.read(cor_queue)
    # 
    #data = tf.image.decode_jpeg(value, channels = 3)
    #cor_data = tf.image.decode_jpeg(cor_value, channels = 3)
    #
    #data.set_shape([32, 32,3])
    #data = tf.to_float(data)  # convert uint8 to float32
    #data = tf.reshape(data, [-1])
    #
    #cor_data.set_shape([32, 32,3])
    #cor_data = tf.to_float(cor_data)  # convert uint8 to float32
    #cor_data = tf.reshape(cor_data, [-1])
    #
    #
    #
    #min_after_dequeue = 10240
    #capacity = min_after_dequeue + 3 * 64
    #data_batch, cor_data_batch = tf.train.shuffle_batch([data, cor_data], batch_size=64, capacity=capacity,min_after_dequeue=min_after_dequeue,allow_smaller_final_batch=True)
    
    #x = data_batch
    #x.set_shape([64,1024])
    #y_ = cor_data_batch
    #y_.set_shape([64,1024])
    ##  VARIABLES
    
    x = tf.placeholder(tf.float32,[64,1024])
    y_ = tf.placeholder(tf.float32,[64,1024])
     
     
    # DA 1
    W = tf.Variable(tf.random_normal([1024,5120],stddev=0.1))
    b = tf.Variable(tf.zeros([5120]))
     
    W_ = tf.Variable(tf.random_normal([5120,1024],stddev=0.1))
    b_ = tf.Variable(tf.zeros([1024]))
     
     
     
     
     
    #   DENOISING AUTO-ENCODER 1
     
    #   DA layer 1
     
#    x_reg = tf.divide(x,255)
#    y_reg = tf.divide(y_,255)
    y = tf.matmul(x,W) + b   
    h_x_s = tf.nn.sigmoid(y)
     
    #   DA layer 2
    h_x= tf.matmul(h_x_s,W_) + b_
    y_xi = tf.nn.sigmoid(h_x) 
     
    #   h(yi)
    h_yy = tf.matmul(y_,W) + b
    h_y = tf.nn.sigmoid(h_yy)
     
     
    
     
    #   LOSS FUNCTION L1
     
    #   1o meros tis sxesis loss
    dif_y = tf.subtract(y_xi,y_)
    sqr_dif_y = tf.square(dif_y)
    oug = tf.reduce_sum(sqr_dif_y)
    error_1 = tf.divide(oug,2*64)
    
    #  2o meros tis sxesis loss
    # r kapelo
    summed_hx = tf.reduce_sum(h_x_s,0)
    ro_hat = tf.divide(summed_hx, 64)
     
    # KL part 1
    r_temp = tf.divide(0.01,ro_hat) # r / r_hat
    r_log = tf.log(r_temp)
    r_log_r = tf.multiply(0.01,r_log)
    part1_KL = tf.reduce_sum(r_log_r)
     
    #KL part 2
    ena_r = 1 - 0.01
    ena_ro_hat = tf.subtract(1.0,ro_hat)
    r_temp1 = tf.divide(ena_r,ena_ro_hat)
    r_log1 = tf.log(r_temp1)
    r_log_r1 = tf.multiply(ena_r,r_log1)
    part2_KL = tf.reduce_sum(r_log_r1)
    KL_part = tf.add(part1_KL, part2_KL)
    error_2 = tf.multiply(0.01, KL_part)
    
    #  3o meros tis sxesis loss
#    temp = tf.matmul(tf.transpose(W),W)
    fro_W = tf.norm(W, ord ='fro',axis=(0,1))
    fro_W_ = tf.norm(W_, ord ='fro',axis=(0,1))
    W_trace_sqr = tf.square(fro_W)
    WT_trace_sqr = tf.square(fro_W_)
    W_mid_trace = tf.reduce_sum(W_trace_sqr)
    WT_mid_trace = tf.reduce_sum(WT_trace_sqr)
    sum_trace = W_mid_trace + WT_mid_trace 
    error_3 = tf.multiply((0.0001/2), sum_trace) 
     
     
    #   Loss + Training Step
    loss = error_1 + error_2 + error_3
    train_step = tf.train.AdamOptimizer().minimize(loss)
#    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
#    
    sess = tf.InteractiveSession()
     
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
     
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
     
    
    #sam = 0
    for i in range(num_epochs):
        sam = 0
        for j in range(30):
            batch_x = np_cor[(j*64):((j+1)*64)]
            batch_y = np_clean[(j*64):((j+1)*64)]
#            derp1 = np.concatenate(batch_x, axis = 0)
#            derp2 = np.concatenate(batch_y, axis = 0)
    #        derp1.astype(float)
    #        derp2.astype(float)
            _, DA1_error, DA1_W, DA1_W_, DA1_b, DA1_b_, DA1_hx, DA1_hy = sess.run([train_step, loss, W, W_, b, b_, h_x_s, h_y],feed_dict={x: batch_x ,y_: batch_y })
    #       k = sess.run([x])
    #       print len(k)
            if (i == num_epochs - 1):
                collect_hx.append(DA1_hx)
                collect_hy.append(DA1_hy)
#            print er1,er2,er3
            sam += DA1_error
    #        print 'Epoch ' + str(i) + ',' + str(j) + ' had error: '+ str(DA1_error)+ '\n'
        print 'Epoch ' + str(i)
        print sam/30
        print '-----------------------------------'
        
    DA1_hx = np.asarray(collect_hx)
    DA1_hy = np.asarray(collect_hy)
    coord.request_stop()
    coord.join(threads)
     
    sess.close()
    print '\nFinished DA1\n'
    return  DA1_W, DA1_W_, DA1_b, DA1_b_, DA1_hx, DA1_hy
 