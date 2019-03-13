import random
acc=0
def RElu(ws):
    return 1 if ws>=1 else 0

class perceptron:
    weight=[]
    learning_rate=0.1
    acc=0
    def __init__(self):
        self.weight=[random.uniform(0.0,1.5) for _ in range(0,3)]

    def guess(self,inputs):
        sum=0
        for w, i in zip(self.weight,inputs):
            sum=sum+w*i
        return RElu(sum)

    def train(self, inputs, target):
        gues=self.guess(inputs)
        wk=[]
        k=0
        error= target - gues
        self.acc+= 1 if target==gues else 0
        for w, i  in zip(self.weight, inputs):
            self.weight[k]=w+(error*i*self.learning_rate)
            k+=1


class points:
    def __init__(self):
        a,b=random.randint(0,5),random.randint(0,5)
        self.x= 1 if a>=3 else 0
        self.y= 1 if b>=3 else 0
        self.lable= 1 if self.x==self.y and self.x==self.y!=0 else 0

pr=perceptron()
p=[points() for i in range(0,10000)]
for o in p:
    pr.train([o.x,o.y,1],o.lable)


v=pr.guess([1,1,1])
print(v)
print(pr.acc,'10000')
