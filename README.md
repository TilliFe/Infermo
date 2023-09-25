# MojoGrad ❄️
A tiny Automatic Differentiation library in Mojo for building some cool stuff.
- **Warning**: Please note that MojoGrad is a Proof of Concept and not quite usable yet. It’s an early approach to building a differentiable programming engine with Mojo, a very new language. I am excited about the possibilities and am actively working on improving it. Stay tuned for updates!

## Features 🌟

- **No External Libraries**: MojoGrad is built entirely in Mojo, without the use of any external libraries written in other languages like C++ or Python.
- **Easy Model Definition**: Define your models dynamically in an object oriented way just like Pytorch. 
- **Automatic Differentiation**: Compute gradients automatically

## Example Code 📜

Here's an example of how you can define your own model in MojoGrad:

```python
from MojoGrad import Module, Tensor, shape

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

    fn step(inout self):
        self.nn.optimize('sgd', 0.04)

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
        model.step()
        # logits.printData()
        # model.nn.printTensors()
```
In this code snippet, we first define a Linear function that performs a matrix multiplication and applies a ReLU activation function. We then define a model struct that includes the definition of our model architecture. The forward method computes the forward pass of the model and the backward method computes the gradients, which are used in the step method to optimize the networ. Finally, we have a main function that initializes our model and runs the training loop.


## Getting Started with MojoGrad 🚀

To get started with MojoGrad, you can find all the code in the `MojoGrad.mojo` file. This file contains all the necessary objects and functions of MojoGrad. You can simply copy this file into your project.

The actual development files are located in the `dev` directory. This is where you'll find the most up-to-date and in-progress versions of my code.