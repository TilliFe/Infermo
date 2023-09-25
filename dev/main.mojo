from module import Module
from Tensor import Tensor, shape

fn Linear(inout nn: Module, inout x: Tensor) -> Tensor:
    var W = Tensor(shape(16,4,4))
    W.initRandom(-1,1)
    x = nn.mul(W, x)
    x = nn.ReLU(x)
    return x

struct model:
    var nn: Module
    var input: Tensor
    var trueVals: Tensor
    var logits: Tensor
    var loss: Tensor

    fn __init__(inout self):
        self.input = Tensor(shape(16,4,4))
        self.trueVals = Tensor(shape(16,4,4))
        self.input.setDataAll(2)
        self.trueVals.setDataAll(1)
        self.nn = Module()

        # define model architecture
        var x = Linear(self.nn,self.input)
        for i in range(0):
            x = Linear(self.nn,x)
        self.logits = x
        self.loss = self.nn.MSE(self.logits,self.trueVals)
        
    fn forward(inout self) -> Tensor:
        self.nn.forward(self.logits)
        return self.logits

    fn backward(inout self):
        self.nn.backward(self.loss)

fn main():

    # init
    var model = model()
    var input = Tensor(shape(16,4,4))
    input.requiresGradient = False
    input.setDataAll(2)

    # training loop
    for i in range(1):
        let logits = model.forward()
        model.backward()
        model.nn.optimize('sgd', 0.04)
        # logits.printData()
        model.nn.printTensors()