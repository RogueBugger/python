import random
acc=0
def sign_activation(ws):
    return 1 if ws>=1 else -1
class perceptron:
    weight=[]
    learning_rate=1
    acc=0
    def __init__(self):
        self.weight=[random.uniform(0.0,1.9) for i in range(0,2)]
        #print(f"weight before change{self.weight}")
        
    def guess(self,inputs):
        sum=0
        for w, i in self.weight,inputs:
            sum=sum+w*i
            
        return sign_activation(sum)

    def train(self, inputs, target):
        gues=self.guess(inputs)
        wk=[]
        k=0
        error= target - gues
        
        self.acc+=error/100
        for w, i  in self.weight, inputs:
            self.weight[k]=w+(error*i*self.learning_rate)
            k+=1
        #print(self.weight)
         
            



class points:
    def __init__(self):
        self.x,self.y=random.uniform(0.0,1.4),random.uniform(0.0,1.4)
        self.lable= 1 if self.x>self.y else -1

        #print("x=",self.x,"y=",self.y,"lable=",self.lable)
pr=perceptron()
p=[points() for i in range(0,100)]
for o in p:
    pr.train([o.x,o.y],o.lable)


v=pr.guess([0,1])
print(v)
print(f"weight after change{pr.weight}")
print(pr.acc)
