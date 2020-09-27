#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:12:27 2017

@author: costis
"""

from DA1 import Denosing_Autoencoder_1
from datetime import datetime
from DA2 import Denosing_Autoencoder_2
import tensorflow as tf
import glob
import numpy as np
from PIL import Image

startTime = datetime.now()
print 'gia pame'
my_times = 75
num_epochs = 75


error_plot = []
ta_w = []
np_clean = []
np_cor = []
np_test = []
np_test1 = []
np_test3 = []
np_test2 = []
clean_set = []
testing_set = glob.glob('/corrupted_set_1/*.jpg')

testing_set1 = glob.glob('/testing_set/*.jpg')

corrupted_set = glob.glob('/corrupted_set/*.jpg')

clean_set = glob.glob('/clean_set/*.jpg')

testing_set2 = glob.glob('/testing_set1/*.jpg')

testing_set3 = glob.glob('/testing_set2/*.jpg')

for path1, path2 in zip(clean_set,corrupted_set):
    img1 = Image.open( path1 )
    data1 = np.asarray( img1, dtype="float32" )
    data1 = data1/255.0
#    data1 = (data1 - np.mean(data1))
    data1 = data1.reshape(-1)
    np_clean.append(data1)
    img2 = Image.open( path2 )
    data2 = np.asarray( img2, dtype="float32" )
    data2 = data2/255.0
#    data2 = (data2 - np.mean(data2))
    data2 = data2.reshape(-1)
    np_cor.append(data2)
    
for test_temp in testing_set:
    img3 = Image.open( test_temp )
    data3 = np.asarray( img3, dtype="float32" )
    data3 = data3/255.0
#    data1 = (data1 - np.mean(data1))
    data3 = data3.reshape(-1)
    np_test.append(data3)    
#    
for test_temp1 in testing_set1:
    img4 = Image.open( test_temp1 )
    data4 = np.asarray( img4, dtype="float32" )
    data4 = data4/255.0
#    data1 = (data1 - np.mean(data1))
    data4 = data4.reshape(-1)
    np_test1.append(data4)    
    
for test_temp2 in testing_set2:
    img5 = Image.open( test_temp2 )
    data5 = np.asarray( img5, dtype="float32" )
    data5 = data5/255.0
#    data1 = (data1 - np.mean(data1))
    data5 = data5.reshape(-1)
    np_test2.append(data5)    

for test_temp3 in testing_set3:
    img6 = Image.open( test_temp3 )
    data6 = np.asarray( img6, dtype="float32" )
    data6 = data6/255.0
#    data1 = (data1 - np.mean(data1))
    data6 = data6.reshape(-1)
    np_test3.append(data6)    



DA1_W, DA1_W_, DA1_b, DA1_b_, DA1_hx, DA1_hy = Denosing_Autoencoder_1(50, np_clean,np_cor)
DA2_W, DA2_W_, DA2_b, DA2_b_, DA2_hx, DA2_hy = Denosing_Autoencoder_2(num_epochs, DA1_hy, DA1_hx)



collect_output = []
    ##  VARIABLES
x = tf.placeholder(tf.float32,[64,1024])
y_ = tf.placeholder(tf.float32,[64,1024])
    
    
    # DA 1
W1 = tf.Variable(DA1_W)
b1 = tf.Variable(DA1_b)
    
W1_ = tf.Variable(DA1_W_)
b1_ = tf.Variable(DA1_b_)
    
    # DA 2
W2 = tf.Variable(DA2_W)
b2 = tf.Variable(DA2_b)
    
W2_ = tf.Variable(DA2_W_)
b2_ = tf.Variable(DA2_b_)

#nx = tf.divide(x,255)

# NN layer 1
x1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
# NN layer 2
x2 = tf.nn.sigmoid(tf.matmul(x1,W2) + b2)
# NN layer 3
x3 = tf.nn.sigmoid(tf.matmul(x2,W2_) + b2_)
## NN output layer
y = tf.nn.sigmoid(tf.matmul(x3,W1_) + b1_)

# ERROR 1
#y_reg = tf.divide(y_,255)
dif_y = tf.subtract(y,y_)
sqr_dif_y = tf.square(dif_y)
oug = tf.reduce_sum(sqr_dif_y)
error_1 = tf.divide(oug,2*64)

# ERROR 2
#temp1 = tf.matmul(tf.transpose(W1),W1)
fro_W1 = tf.norm(W1, ord ='fro',  axis = (0,1))
W_trace_sqr1 = tf.square(fro_W1)
W_mid_trace1 = tf.reduce_sum(W_trace_sqr1)

#temp1_ = tf.matmul(tf.transpose(W1_),W1_)
fro_W1_ = tf.norm(W1_, ord ='fro',  axis = (0,1))
W_trace_sqr2 = tf.square(fro_W1_)
W_mid_trace2 = tf.reduce_sum(W_trace_sqr2)

#temp2 = tf.matmul(tf.transpose(W2),W2)
fro_W2 = tf.norm(W2, ord ='fro',  axis = (0,1))
W_trace_sqr3 = tf.square(fro_W2)
W_mid_trace3 = tf.reduce_sum(W_trace_sqr3)

#temp2_ = tf.matmul(tf.transpose(W2_),W2_)
fro_W2_ = tf.norm(W2_, ord ='fro',  axis = (0,1))
W_trace_sqr4 = tf.square(fro_W2_)
W_mid_trace4 = tf.reduce_sum(W_trace_sqr4)

sum_sqr = W_mid_trace1 + W_mid_trace2 + W_mid_trace3 + W_mid_trace4
error_2 = tf.multiply((0.0001/2), sum_sqr) 

# LOSS
loss = error_1 + error_2

train_step = tf.train.AdamOptimizer(0.00005).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

sess = tf.InteractiveSession()
    
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
    
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for i in range(my_times):
    sam = 0
    for j in range(30):
        batch_x = np_cor[(j*64):((j+1)*64)]
        batch_y = np_clean[(j*64):((j+1)*64)]
#        derp1 = np.concatenate(batch_x, axis = 0)
#        derp2 = np.concatenate(batch_y, axis = 0)
#        derp1.astype(float)
#        derp2.astype(float)
        _, NN_error,psi,er1 = sess.run([train_step, loss,y,error_1],feed_dict={x: batch_x ,y_: batch_y })
#       k = sess.run([x])
#       print len(k)\
#        print NN_error
        if (i == my_times - 1):
            collect_output.append(psi)
#        print er1,        er2
#        ta_w.append(er2)
        sam += NN_error
#        print 'Epoch ' + str(i) + ',' + str(j) + ' had error: '+ str(DA1_error)+ '\n'
#    print er1, er2
    print 'Epoch ' + str(i)
    print sam/30
    error_plot.append(sam)
    print '-----------------------------------'
tapsi = np.concatenate(collect_output, axis = 0)
j = 0
batch_test = np_test[(j*64):((j+1)*64)]
teesstt = sess.run([y], feed_dict={x: batch_test })
j = 0
batch_test1 = np_test1[(j*64):((j+1)*64)]
teesstt1 = sess.run([y], feed_dict={x: batch_test1 })

j = 0
batch_test2 = np_test2[(j*64):((j+1)*64)]
teesstt2 = sess.run([y], feed_dict={x: batch_test2 })

j = 0
batch_test3 = np_test3[(j*64):((j+1)*64)]
teesstt3 = sess.run([y], feed_dict={x: batch_test3 })

coord.request_stop()
coord.join(threads)
 
print '\nDone\n'

print datetime.now() - startTime