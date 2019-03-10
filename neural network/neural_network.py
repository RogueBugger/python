import random

def activation(value):
    return 1 if value > 1 else 0

weight = float
learning_rate=0.1

weight=[random.uniform(0.0,1.9) for i in range(0,2)]

def guess(inputv):
  sum=0
  for w,i in weight,inputv:
    sum+=w*i
  neuron=activation(sum)
  return neuron



def train(inputv, target):
  error=int
  gues=guess(inputv)
  error=target-gues
  for w,i in weight,inputv:
    k=w+error*i*learning_rate
  print(weight)
  

'''def trainn(a,b):
  obj.train(a,b)
inp=[([1,1],1),([1,0],1),([0,0],0),([0,1],1)]
for i in range(len(inp)):
  trainn((inp[i][0]),inp[i][1])
'''

  

train([1,1],1)
train([0,1],1)
train([1,0],1)
train([0,0],0)