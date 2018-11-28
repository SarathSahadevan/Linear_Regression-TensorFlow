# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:04:07 2018

@author: Sarath.Sahadevan
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
sess = tf.Session()



path = os.getcwd() + '\Iris.csv'  
data = pd.read_csv(path, header=None, names=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])  
data.head()

data=data.drop(data.index[[0]])


data_1 = data['SepalLengthCm']
data_2 = data['SepalWidthCm']
data_3 = data['PetalLengthCm']

data_4 = data['PetalWidthCm']

data_5 = data['Species']

data_5 = data['Species'].map({'Iris-setosa':'1', 'Iris-versicolor':'2', 'Iris-virginica':'3'})

#Y = w*x+b

w1 = tf.Variable([-3],dtype = np.float32)
w2 = tf.Variable([-3],dtype = np.float32)

w3 = tf.Variable([-3],dtype = np.float32)
w4 = tf.Variable([-3],dtype = np.float32)

b = tf.Variable([3],dtype = np.float32)

x1 = tf.placeholder(dtype = np.float32)
x2 = tf.placeholder(dtype = np.float32)
x3 = tf.placeholder(dtype = np.float32)
x4 = tf.placeholder(dtype = np.float32)

act_y = tf.placeholder(dtype = np.float32)

linear_model = w1*x1+w2*x2+w3*x3+w4*x4+b

squared_error = tf.square(linear_model-act_y)


loss = tf.reduce_sum(squared_error)

optimizer_train = tf.train.GradientDescentOptimizer(0.0001)

train = optimizer_train.minimize(loss)


init  = tf.global_variables_initializer()
sess.run(init)


sess.run(loss,{x1:data_1,x2:data_2,x3:data_3,x4:data_4,act_y:data_5})

sess.run([w1,w2,w3,w4]),sess.run([b])


for i in range (0,100000):
    sess.run(train,{x1:data_1,x2:data_2,x3:data_3,x4:data_4,act_y:data_5})
    print('loss >>>',sess.run(loss,{x1:data_1,x2:data_2,x3:data_3,x4:data_4,act_y:data_5}))




