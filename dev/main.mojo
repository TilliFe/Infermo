from module import Module
from Tensor import Tensor, shape
from random import rand, random_si64, seed
from math import sin

# define one layer of an MLP
fn Linear(inout nn: Module, inout x: Tensor, numNeurons: Int) -> Tensor:
    let x_dim = x.getShape(x.num_dims - 2)
    var W = Tensor(shape(numNeurons,x_dim))
    W.initRandom(-0.1,0.1)
    x = nn.mul(W,x)
    x = nn.ReLU(x)
    return x

# define the model and its behaviour
struct model:
    var nn: Module
    var input: Tensor
    var trueVals: Tensor
    var logits: Tensor
    var loss: Tensor

    fn __init__(inout self):
        self.input = Tensor(shape(1,64))
        self.input.requiresGradient = False
        self.trueVals = Tensor(shape(1,64))
        self.trueVals.requiresGradient = False
        self.nn = Module()

        # define model architecture
        var x = Linear(self.nn,self.input,16)
        for i in range(1):
            x = Linear(self.nn,x,128)
        self.logits = Linear(self.nn,x,1)
        self.loss = self.nn.MSE(self.trueVals,self.logits)
        
    fn forward(inout self, _input: DTypePointer[DType.float32], _trueVals: DTypePointer[DType.float32]) -> Tensor:
        self.nn.Tensors[1].setData(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.trueVals.setData(_trueVals)
        self.nn.forward(self.logits)
        return self.logits

    fn backward(inout self):
        self.nn.backward(self.loss)

    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.1, momentum = 0.9)


# Data Generator for a simple regression problem
struct DataGenerator:
    var size: Int
    var x: DTypePointer[DType.float32]
    var y: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.size = size
        self.x = DTypePointer[DType.float32].alloc(self.size)
        self.y = DTypePointer[DType.float32].alloc(self.size)

    fn random(self, it: Int):
        seed(it)
        rand(self.x, self.size)
        let min = -1
        let max = 1
        for i in range(self.size):
            let x_rand = self.x.load(i) * (max - min) + min
            self.x.store(i, x_rand)
            let res = 0.5 + 0.5*sin(10*x_rand)
            self.y.store(i, res) 


# train the model
fn main():

    let dataset = DataGenerator(64)
    var model = model()
    let num_epochs = 10000

    var lossSum: Float32 = 0
    let every = 100

    for epoch in range(1,num_epochs):
        dataset.random(epoch)
        let logits = model.forward(dataset.x,dataset.y)
        model.backward()
        model.step()

        lossSum += model.loss.getData(0)
        if( epoch % every == 0):
            print("\nEpoch", epoch,", AvgLoss = ", lossSum / every)
            lossSum = 0      
            # logits.printData()
            # model.nn.printTensors()


