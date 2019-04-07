#importing the requied components
import tensorflow as tf
import numpy as np
from random import shuffle
import time,sys,random

#learning rate of the optimizer
learning_rate=0.01
#dataset to train the network
dataset=[[np.array([1,0]).reshape(2,1),1],
[np.array([0,1]).reshape(2,1),1],
[np.array([1,1]).reshape(2,1),1],
[np.array([0,0]).reshape(2,1),0],
]

#to give input value in the run time
x=tf.placeholder(tf.float32,shape=(None))
#to give the target value in the run time for the optimizer to fing the error
target=tf.placeholder(tf.float32,shape=(None))
#input layer->1
#weight and bias of the layer1
weightIH=tf.Variable(tf.random_normal([2,2]),name='weightsIH')
biasIH=tf.Variable(tf.random_normal([2,1]))
#matrix multiplying the weight with the input and adding with the bias
layer1=tf.add(tf.matmul(weightIH,x),biasIH)
#squeeze the predicted value between 0 and 1
yIH=tf.nn.sigmoid(layer1)

#input layer->2
#weight and bias of the layer2
weightHO=tf.Variable(tf.random_normal([1,2]),name='weightsHO')
biasHO=tf.Variable(tf.random_normal([1,1]))
#matrix multiplying the weight of the second layer by the input from the first layer
layer2=tf.add(tf.matmul(weightHO,yIH),biasHO)
#squeeze the predictd value between 0 and 1
yHO=tf.nn.sigmoid(layer2)

#finding the cost of the model
#tf.pow is to ensure that the value is positive
costHO=tf.reduce_sum(tf.pow(yHO-target,2))
#reducing the cost using the gradient descent optimizer of tensorflow
optimizerIO=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(costHO)
#no of times the dataset is trained in the model
epochs=50000
#initizlizing the tf variables
init=tf.global_variables_initializer()
#creating a session
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        #for every iteration shuffling the dataset gives better results
        shuffle(dataset)
        for data in dataset:
            sess.run(optimizerIO,feed_dict={x:data[0],target:data[1]})
        #displaying the no of epochs is finished
        sys.stdout.write("\r"+str(e))
    x1=np.array([1,0]).reshape(2,1)
    x2=np.array([0,0]).reshape(2,1)
    x3=np.array([1,1]).reshape(2,1)
    x4=np.array([0,1]).reshape(2,1)
    #check the model is working as expected
    print(sess.run(yHO,feed_dict={x:x1}))
    print(sess.run(yHO,feed_dict={x:x2}))
    print(sess.run(yHO,feed_dict={x:x3}))
    print(sess.run(yHO,feed_dict={x:x4}))
