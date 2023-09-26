from module import Module
from Tensor import Tensor, shape
from random import rand

# define one layer of an MLP
fn Linear(inout nn: Module, inout x: Tensor) -> Tensor:
    var W = Tensor(shape(1,5,5))
    W.initRandom(-1,1)
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
        self.input = Tensor(shape(1,5,5))
        self.input.requiresGradient = False
        self.trueVals = Tensor(shape(1,5,5))
        self.trueVals.requiresGradient = False
        self.nn = Module()

        # define model architecture
        var x = Linear(self.nn,self.input)
        for i in range(2):
            x = Linear(self.nn,x)
        self.logits = x
        self.loss = self.nn.MSE(self.logits,self.trueVals)
        self.loss.setGradientAll(1)
        
    fn forward(inout self, _input: DTypePointer[DType.float32], _trueVals: DTypePointer[DType.float32]) -> Tensor:
        self.nn.Tensors[1].setData(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.trueVals.setData(_trueVals)
        self.nn.forward(self.logits)
        return self.logits

    fn backward(inout self):
        self.nn.Tensors[11].setGradientAll(1)
        self.nn.backward(self.loss)


# Data Genrator for doing some linear approximation
struct DataGenerator:
    var size: Int
    var x: DTypePointer[DType.float32]
    var y: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.size = size
        self.x = DTypePointer[DType.float32].alloc(self.size)
        self.y = DTypePointer[DType.float32].alloc(self.size)
        rand(self.x, size)
        let min = 0
        let max = 1
        for i in range(size):
            let x_rand = self.x.load(i) * (max - min) + min
            self.x.store(i, x_rand)
            let res = x_rand * 2
            self.y.store(i, res) 


# train the model
fn main():

    let dataset = DataGenerator(25)
    var model = model()

    for i in range(1000):
        let logits = model.forward(dataset.x,dataset.y)
        model.backward()
        model.nn.optimize('sgd', 0.1)
        print(model.loss.getData(0))
        # logits.printData()
        # model.nn.printTensors()