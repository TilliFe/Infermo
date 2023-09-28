# MojoGrad🔥

A small AutoDiff library in Mojo for Neural Network training.

- **Warning**: Please note that MojoGrad is still a Proof of Concept for a Differentiable Programming Engine and is not quite usable yet.

## Features

- **No External Libraries**: MojoGrad is built entirely in Mojo, without the use of any external libraries written in other languages like C++ or Python.
- **Easy Model Definition**: Define your models dynamically in an object oriented way just like in Pytorch.
- **Automatic Differentiation**: Compute gradients automatically

## Example Code

### Simple Example

Let's start with a basic multiplication between two tensors and their respective gradient computation.

```python
from MojoGrad import Module, Tensor, shape

fn main():
    # init
    var nn = Module()
    var A = Tensor(shape(5,3))
    var B = Tensor(shape(3,4))

    # specify tensor entries
    A.setDataAll(2)
    B.setDataAll(3)

    # perform computation and print result
    var C = nn.mul(A,B)
    var D = nn.sum(C)

    # perform computation and print result
    nn.forward(C)
    C.printData()

    # compute gradients of A and B
    nn.backward(D)
    A.printGradient()
    B.printGradient()
```

### Build a Neural Network

Let's now look at a slightly more elaborate example. Here we define a struct called 'model' which defines the architectur of our Neural Network.

We also define a function Linear , which is basic basic MLP building block. Last but not least, we define a Data Generator, which provides data for a simple regression task, i.e. the Network should learn the sine curve on a given interval. (I.e. given the x values between 0 and 1, the model should learn to predict the proper y values.)

Import the necessary parts from MojoGrad and the Mojo standard library

```python
from MojoGrad import Module, Tensor, shape, Linear
from random import rand, random_si64, seed
from math import sin
```

Define the model architecture,the number of layers and neurons in each layer.

```python
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
        var x = Linear(self.nn,self.input, num_neurons=32, addBias=True, activation='ReLU')
        for i in range(1):
            x = Linear(self.nn,x, num_neurons=128, addBias=True, activation='ReLU')
        self.logits = Linear(self.nn,x,1,True,'none')
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
```

Define the Data Generator

```python
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
```

Call the model and define the number of epochs, then let it train on a randomly generated batch of data.

```python
fn main():

    let dataset = DataGenerator(512)
    var model = model()
    let num_epochs = 10000

    var lossSum: Float32 = 0
    let every = 100

    for epoch in range(0,num_epochs):
        dataset.random(epoch)
        let logits = model.forward(dataset.x,dataset.y)
        model.backward()
        model.step()

        lossSum += model.loss.getData(0)
        if( epoch % every == 0 and epoch > 0):
            print("\nEpoch", epoch,", AvgLoss = ", lossSum / every)
            lossSum = 0      
            # logits.printData()
            # model.nn.printTensors()
```

## TODO 🚀
- More basic Operators/Loss functions/Activations
- Enhance backprop with Vectorization and Paralleization
- Link to Numpy for Data handling