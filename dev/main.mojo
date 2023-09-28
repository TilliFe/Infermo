from module import Module
from Tensor import Tensor, shape
from abstractions import Linear
from random import rand, random_si64, seed
from math import sin

# define the model and its behaviour
struct model:
    var nn: Module
    var input: Tensor
    var trueVals: Tensor
    var logits: Tensor
    var loss: Tensor

    fn __init__(inout self):
        self.input = Tensor(shape(1,512))
        self.input.requiresGradient = False
        self.trueVals = Tensor(shape(1,512))
        self.trueVals.requiresGradient = False
        self.nn = Module()

        # define model architecture
        var x = Linear(self.nn,self.input, num_neurons=8, addBias=True, activation='ReLU')
        for i in range(2):
            x = Linear(self.nn,x, num_neurons=16, addBias=True, activation='ReLU')
        self.logits = Linear(self.nn,x,1,True,'none')
        self.loss = self.nn.MSE(self.trueVals,self.logits)

    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _trueVals: DTypePointer[DType.float32]) -> Tensor:
        self.nn.Tensors[1].setData(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.trueVals.setData(_trueVals)
        self.nn.forward(self.logits)
        return self.logits

    @always_inline
    fn backward(inout self):
        self.nn.backward(self.loss)

    @always_inline
    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.01, momentum = 0.9)


# Data Generator for a simple regression problem
struct DataGenerator:
    var size: Int
    var x: DTypePointer[DType.float32]
    var y: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.size = size
        self.x = DTypePointer[DType.float32].alloc(self.size)
        self.y = DTypePointer[DType.float32].alloc(self.size)

    @always_inline
    fn random(self, it: Int):
        seed(it)
        rand(self.x, self.size)
        let min = -1
        let max = 1
        for i in range(self.size):
            let x_rand = self.x.load(i) * (max - min) + min
            self.x.store(i, x_rand)
            let res = 0.5 + 0.5*sin(5*x_rand)
            self.y.store(i, res) 


# train the model
fn main():

    let dataset = DataGenerator(512)
    var model = model()
    let num_epochs = 1000

    var lossSum: Float32 = 0
    let every = 100

    for epoch in range(0,num_epochs):
        dataset.random(epoch)
        let logits = model.forward(dataset.x,dataset.y)
        model.backward()
        model.step()

        lossSum += model.loss.getData(0)
        if( epoch % every == 0 and epoch > 0):
            print("Epoch", epoch,", AvgLoss = ", lossSum / every)
            lossSum = 0      
            # logits.printData()
            # model.trueVals.printData()
            # model.nn.printTensors()


# fn main():
#     # init
#     var nn = Module()
#     var A = Tensor(shape(2,3))
#     var B = Tensor(shape(3,4))

#     # specify tensor entries
#     A.setDataAll(2)
#     B.setDataAll(3)

#     # perform computation and print result
#     var C = nn.mul(A,B)
#     var D = nn.sum(C)

#     # perform computation and print result
#     nn.forward(C)
#     C.printData()

#     # compute gradients of A and B
#     nn.backward(D)
#     A.printGradient()
#     B.printGradient()
