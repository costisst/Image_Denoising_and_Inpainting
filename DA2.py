#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:49:24 2017

@author: costis
"""
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# 
import tensorflow as tf
import numpy as np

def Denosing_Autoencoder_2(num_epochs,np_clean, np_cor ):
    
    
    collect_hy = []
    collect_hx = []

    x = tf.placeholder(tf.float32,[64,5120])
    y_ = tf.placeholder(tf.float32,[64,5120])
        
        
        # DA 1
    W = tf.Variable(tf.random_normal([5120,5120],stddev=0.1))
    b = tf.Variable(tf.zeros([5120]))
        
    W_ = tf.Variable(tf.random_normal([5120,5120],stddev=0.1))
    b_ = tf.Variable(tf.zeros([5120]))
     
     
     
     
     
    #   DENOISING AUTO-ENCODER 1
     
    #   DA layer 1
     
#    x_reg = x/255
#    y_reg = y_/255
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
#    temp_ = tf.matmul(tf.transpose(W_),W_)
    fro_W_ = tf.norm(W_, ord ='fro',axis=(0,1))
    W_trace_sqr = tf.square(fro_W)
    WT_trace_sqr = tf.square(fro_W_)
    W_mid_trace = tf.reduce_sum(W_trace_sqr)
    WT_mid_trace = tf.reduce_sum(WT_trace_sqr)
    sum_trace = W_mid_trace + WT_mid_trace 
    error_3 = tf.multiply((0.0001/2), sum_trace) 
     
     
    #   Loss + Training Step
    loss = error_1 + error_2                                                                                                                                                                                                     + error_3
    train_step = tf.train.AdamOptimizer().minimize(loss)
#    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
     
    
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
            derp1 = np_cor[j]
            derp2 = np_clean[j]
    #        derp1.astype(float)
    #        derp2.astype(float)
            _, DA1_error, DA2_W, DA2_W_, DA2_b, DA2_b_, DA2_hx, DA2_hy = sess.run([train_step, loss, W, W_, b, b_, h_x_s, h_y],feed_dict={x: derp1 ,y_: derp2 })
    #       k = sess.run([x])
    #       print len(k)
            if (i == num_epochs - 1):
                collect_hx.append(DA2_hx)
                collect_hy.append(DA2_hy)
            sam += DA1_error
    #        print 'Epoch ' + str(i) + ',' + str(j) + ' had error: '+ str(DA1_error)+ '\n'
        print 'Epoch ' + str(i)
        print sam/30
        print '-----------------------------------'

    DA2_hx = np.asarray(collect_hx)
    DA2_hy = np.asarray(collect_hy)
    coord.request_stop()
    coord.join(threads)
     
    sess.close()
    print '\nFinished DA2\n'
    return  DA2_W, DA2_W_, DA2_b, DA2_b_, DA2_hx, DA2_hy