# Infermo🔥

### AutoDiff with Tensors in a thousand lines of pure Mojo code!

**Heads up**: Infermo, as a Differentiable Programming Engine, is currently in its proof-of-concept stage and is not fully operational yet. It will be ready for some basic tests in a couple of days. 

## Features

- **No External Libraries**: Infermo is built entirely in Mojo, without the use of any external libraries written in other languages like C++ or Python.
- **Easy Model Definition**: Define your models dynamically in an object oriented way just like in Pytorch. Train Neural Network on classification and regression tasks.
- **Automatic Differentiation**: Compute gradients automatically

## Example Code

### Train a Neural Network on the **MNIST** dataset

Import the necessary parts from Infermo

```python
from dv import Module, Tensor, shape, Linear, oneHot, accuracy
```

Define the model architecture (simple MLP with ReLU activations and biases)

```python
struct model:
    var nn: Module
    var input: Tensor
    var trueVals: Tensor
    var logits: Tensor
    var loss: Tensor
    var avgAcc: Float32

    fn __init__(inout self):
        self.input = Tensor(shape(64,784))
        self.input.requiresGradient = False
        self.trueVals = Tensor(shape(64,10))
        self.trueVals.requiresGradient = False
        self.nn = Module()
        self.avgAcc = 0

        # define model architecture
        var x = Linear(self.nn,self.input, num_neurons=64, addBias=True, activation='ReLU')
        for i in range(1):
            x = Linear(self.nn,x, num_neurons=128, addBias=True, activation='ReLU')
        x = Linear(self.nn,x,10,True,'none')
        self.logits = self.nn.softmax(x)
        self.loss = self.nn.MSE(self.trueVals,self.logits)

    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _trueVals: DTypePointer[DType.float32]) -> Tensor:

        # fill the input and trueVals Tensors with theri data
        self.nn.Tensors[0].setData(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.trueVals.setData(_trueVals)

        # one forward pass through the network
        self.nn.forward(self.logits)

        # some additional ops, not necessary for the training, just for showing the accuracy
        let oneHots = oneHot(self.logits)
        self.avgAcc = accuracy(oneHots,self.trueVals)

        return self.logits

    @always_inline
    fn backward(inout self):
        self.nn.backward(self.loss)

    @always_inline
    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.05, momentum = 0.9)
```

Read in the MNIST dataset from a file, initialize the model and define the number of epochs, then let it train on a randomly generated batch of data. 

```python
fn main()raises:

    # init
    var dl = DataLoader('./datasets/mnist.txt')
    var model = model()

    let num_epochs = 1000
    var lossSum: Float32 = 0
    var avgAcc: Float32 = 0
    let every = 100

    for epoch in range(1,num_epochs+1):
        # load a batch of images into the model
        let inputs = dl.load(
            batchSize=64,
            start=1, # regarding the columns of the dataset
            end=785,
            scalingFactor=Float32(1)/Float32(255)
        )
        # load the labels for the images (oneHot encded from 0 to 9)
        let labels = dl.oneHot(
            batchSize=64,
            index=0, # regarding the columm of the labels in the dataset
            ndims=10
        )
        let logits = model.forward(inputs,labels)
        model.backward()
        model.step()

        lossSum += model.loss.getData(0)
        avgAcc += model.avgAcc
        if( epoch % every == 0):
            print("Epoch", epoch,", AvgLoss =", lossSum / every, ", AvgAccuracy =", avgAcc / every)
            lossSum = 0      
            avgAcc = 0
            # logits.printData()
            # model.trueVals.printData()
```

### Simple Example

If that was a bit too much, here is a simpler example of a basic multiplication between two tensors and their respective gradient computation.

```python
from Infermo import Module, Tensor, shape

fn main():
    # init
    var nn = Module()
    var A = Tensor(shape(5,3))
    var B = Tensor(shape(3,4))

    # specify tensor entries
    A.setDataAll(2)
    B.setDataAll(3)

    # perform computation 
    var C = nn.mul(A,B)
    var D = nn.sum(C)
    nn.forward(C)

    # print result of matrix multiplication
    C.printData()

    # compute gradients of A and B
    nn.backward(D)
    A.printGradient()
    B.printGradient()
```

## TODO 🚀
- More basic Operators/Loss-functions/Activations
- more speedups via Vectorization and Paralleization

## Usage
Make sure you have installed and [configured mojo on your environment](https://docs.modular.com/mojo/manual/get-started/index.html)

If you have problems with the Python integration, follow these steps: https://gist.github.com/trevorhobenshield/6bca58f947ad6115a113a97072df1a73

Clone the repository
```
git clone https://github.com/TilliFe/Infermo.git
```
Navigate to the cloned repository
```
cd Infermo
```
Once this is set up, you can directly try out the MNIST training setup with the following command
```
mojo trainMNIST.mojo
```
or the regression task
```
mojo regressionTask.mojo
```