#importing the required components
import tensorflow as tf,numpy as np
from random import shuffle
import time,sys,random
#learning rate of the optimizer
learning_rate=0.01
#dataset to train the network
dataset=[[np.array([1,0]).reshape(2,1),0],
[np.array([0,1]).reshape(2,1),0],
[np.array([1,1]).reshape(2,1),1],
[np.array([0,0]).reshape(2,1),0],
]
#giving input at the runtime and target value to identify the error
x=tf.placeholder(tf.float32,shape=(None))
target=tf.placeholder(tf.float32,shape=(None))

#layer->1
#weight and bias of the layer1
weight=tf.Variable(tf.random_normal([1,2]))
bias=tf.Variable(tf.random_normal([1,1]))
#predicting the output for the given input
layer1=tf.add(tf.matmul(weight,x),bias)
#squeeze the predicted the ouput into 0 and 1
y=tf.nn.sigmoid(layer1)
#finding the cost of the model
cost=tf.reduce_sum(tf.pow(y-target,2))
#gradient descent optimizer is used to reduce the error of the model
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#iteration of dataset trained throught the model
epochs=50000
#initializing the tf variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        #shuffling the dataset to get the better result
        shuffle(dataset)
        for data in dataset:
            sess.run(optimizer,feed_dict={x:data[0],target:data[1]})
        #displying the no of epochs is finished
        sys.stdout.write("\r"+str(e))
    x1=np.array([1,0]).reshape(2,1)
    x2=np.array([0,0]).reshape(2,1)
    x3=np.array([1,1]).reshape(2,1)
    x4=np.array([0,1]).reshape(2,1)
    #checking the model is working as expected
    print(sess.run(y,feed_dict={x:x1}))
    print(sess.run(y,feed_dict={x:x2}))
    print(sess.run(y,feed_dict={x:x3}))
    print(sess.run(y,feed_dict={x:x4}))
