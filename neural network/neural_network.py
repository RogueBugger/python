import random,math
import numpy as np
from random import shuffle

def RElu(a):
    ar=[]
    for v in a:
        ar.append([1 if v>0.5 else 0])
    return ar

def lsf(a):
    return 1/(1+np.exp(-a))

class Neuron:
    def __init__(self,input_layer,hidden_layer,output_layer):
        self.input_layer=input_layer
        self.hidden_layer=hidden_layer
        self.output_layer=output_layer
        self.hidden_input=0
        self.output=0
        self.input_layer_weight=[]
        for i in range(self.hidden_layer):
            self.input_layer_weight.append([random.uniform(0.0,1.0) for i in range(self.input_layer)])
        self.hidden_layer_weight=[]
        for i in range(self.output_layer):
            self.hidden_layer_weight.append([random.uniform(0.0,1.0) for i in range(self.hidden_layer)])
        self.bias_input_weight=[]
        for i in range(self.hidden_layer):
            self.bias_input_weight.append([random.uniform(0.0,1.0) for i in range(1)])
        self.bias_hidden_weight=[]
        for i in range(output_layer):
            self.bias_hidden_weight.append([random.uniform(0.0,1.0) for i in range(1)])
        self.bias=np.array([[-0.5],[-0.6]])
        self.biasHidden=np.array([-0.5])
        self.learning_rate=1.5
        self.acc=0

    def gradient(self,x):
        k=np.subtract(1,x)
        x=np.multiply(k,x)
        return x


    def guess(self,inputs):
        sum=0
        sum=np.dot(self.input_layer_weight,inputs)
        sumBias=np.multiply(self.bias_input_weight,self.bias)
        v=np.add(sumBias,sum)
        hd=lsfcd (v)
        self.hidden_input=hd
        sum1=np.dot(self.hidden_layer_weight,hd)
        sumBias1=np.multiply(self.bias_hidden_weight,self.biasHidden)
        v1=np.add(sumBias1,sum1)
        hd1=lsf(v1)
        self.output=hd1
        return hd1

    def train(self,inputs,target):

        error=np.subtract(target,self.guess(inputs))
        hiddenError=np.dot(np.transpose(self.hidden_layer_weight),error)
        #gradiet of the output and hidden to output weight
        gd=self.gradient(self.output)
        gradientHidden=np.multiply(error,gd)
        deltaHW=np.multiply(gradientHidden,self.learning_rate)
        changeHO=np.dot(deltaHW,np.transpose(self.output))
        self.hidden_layer_weight=np.add(self.hidden_layer_weight,deltaHW)
        #bias delta weight
        self.bias_hidden_weight=np.add(deltaHW,self.bias_hidden_weight)
        inputGradient=np.multiply(hiddenError,self.gradient(self.hidden_input))
        inputErLr=np.multiply(inputGradient,self.learning_rate)
        deltaIH=np.multiply(inputErLr,np.transpose(inputs))
        self.input_layer_weight=np.add(self.input_layer_weight,deltaIH)
        #bias of input_layer
        self.bias=np.add(self.bias,inputErLr)

dataset=[[np.array([1,0]).reshape(2,1),0],
[np.array([0,1]).reshape(2,1),0],
[np.array([1,1]).reshape(2,1),1],
[np.array([0,0]).reshape(2,1),1],
]
a=Neuron(2,2,1)
for i in range(1000):
    shuffle(dataset)
    for v in dataset:
        a.train(v[0],v[1])

print(a.guess(np.array([0,0]).reshape(2,1)))
print(a.guess(np.array([0,1]).reshape(2,1)))
print(a.guess(np.array([1,0]).reshape(2,1)))
print(a.guess(np.array([1,1]).reshape(2,1)))
