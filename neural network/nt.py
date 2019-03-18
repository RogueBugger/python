import random,math
import numpy as np
from random import shuffle

def RElu(a):
    ar=[]
    for v in a:
        ar.append([1 if v>0.5 else 0])
    return ar

def lsf(a):
    ar=[]
    for x in a:
        ar.append([1/(1+math.exp(-x))])
    return ar

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
            self.bias_input_weight.append([random.uniform(0.0,1.0) for i in range(self.input_layer)])
        self.bias_hidden_weight=[]
        for i in range(output_layer):
            self.bias_hidden_weight.append([random.uniform(0.0,1.0) for i in range(1)])
        self.bias=np.array([[1],[1]])
        self.biasHidden=np.array([1])
        self.learning_rate=0.1
        self.acc=0

    def gradient(self,x):
        k=np.subtract(1,x)
        x=np.multiply(k,x)

        return x


    def guess(self,inputs):
        sum=0
        sum=np.dot(self.input_layer_weight,inputs)
        sumBias=np.dot(self.bias_input_weight,self.bias)
        v=np.add(sumBias,sum)
        hd=lsf(v)
        self.hidden_input=hd
        sum1=np.dot(self.hidden_layer_weight,hd)
        sumBias1=np.dot(self.bias_hidden_weight,self.biasHidden)
        v1=np.add(sumBias1,sum1)
        self.output=v1
        hd1=lsf(v1)
        return hd1

    def train(self,inputs,target):
        Guess=self.guess(inputs)

        #output error
        error=np.subtract(target,Guess)

        #gradiet of the output and hidden to output weight
        gd=self.gradient(self.output)
        gradientHidden=np.multiply(gd,np.transpose(self.hidden_input))
        changeHO=np.multiply(error,self.learning_rate)
        deltaHW=np.add(gradientHidden,changeHO)
        self.hidden_layer_weight=np.add(self.hidden_layer_weight,deltaHW)
        #bias delta weight
        deltaBH=np.multiply(self.learning_rate,error)
        self.bias_hidden_weight=np.add(deltaBH,self.bias_hidden_weight)

        #calculating hidden error
        hiddenError=np.dot(np.transpose(self.input_layer_weight),self.hidden_input)
        inputGradient=np.dot(self.gradient(self.hidden_input),np.transpose(inputs))
        inputErLr=np.multiply(hiddenError,self.learning_rate)
        deltaIH=np.multiply(inputErLr,inputGradient)
        self.input_layer_weight=np.add(self.input_layer_weight,deltaIH)

        #bias of input_layer
        deltaBI=np.multiply(self.learning_rate,hiddenError)
        self.bias=np.add(self.bias,deltaBI)
        #print(Guess,target)


class points:
    def __init__(self):
        self.dataset=[[np.array([1,0]).reshape(2,1),1],
        [np.array([0,0]).reshape(2,1),0],
        [np.array([1,1]).reshape(2,1),0],
        [np.array([0,0]).reshape(2,1),1],
        ]

a=Neuron(2,2,1)
o=[]
for i in range(1000 ):
    o.append(points())


for o in o:
    shuffle(o.dataset)
    for v in o.dataset:
        a.train(v[0],v[1])

print(a.guess(np.array([1,0]).reshape(2,1)))
print(a.guess(np.array([0,1]).reshape(2,1)))
print(a.guess(np.array([0,0]).reshape(2,1)))
print(a.guess(np.array([1,1]).reshape(2,1)))
