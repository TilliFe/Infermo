# Infermo🔥

### AutoDiff with Tensors in pure Mojo!

* Independence from external libraries.
* Dynamic, object-oriented model definition
* Automatic gradient computation.

Infermo is currently a Proof-of-Concept. While it’s mainly functional, it’s still under optimization and currently operates on CPU only.

## Overview

**High Level Operators**:
- MLP, Linear Layer with ReLU activation
- (masked) Transformer Block
- 2d Convolution, Max Pooling (still naive implementation)

**General Structs**:
- **Tensor**: A Tensor is a multidimensional array. It is either a separate data holder or part of a computation graph (see Module).
- **Module**: The Module is the Computation Graph and stores the relations between the Tensors (nodes).
- **DataLoader**: The DataLoader reads in data from a .txt file (.csv coming soon). The read-in data is stored in a buffer. Batches of data get generated based on the specified sub-rows and sub-columns.

**Tensor methods**:
- **initialization**: Via the shape(*Ints) function, e.g., 'var new_tensor = Tensor(shape(2,3,4))'
- **getData(index: Int)**: Returns the element at the index position. Attention: Does not take the shape of the Tensor into consideration.
- **setData(index: Int, value: Float32)**: Sets the entry at the index position.
- **setData(new_data: DTypeFloat32[DType.float32])**: If the capacities of the Tensors data and new_data are the same, the data in the Tensor gets efficiently replaced by new_data.
- **etc.** (We recommend checking out the Tensor struct in the graph directory)

**Module Operators**:
Operators include mul, add, sum, ReLU, softmax, reshape (broadcast), MSE, CE, transpose. Each operator takes one or two Tensors and performs a specific operation. For more details on each operator, please refer to the source code. \
*Important:* First, an object of type Module needs to be initialized, like so: 'var nn = Module()'. Then we can call the methods for example as follows: 'var res = nn.mul(tensor1,tensor2).
- **forward(last_node: Tensor)**: Takes in one Tensor, computes one forward pass through the Network till the specified Tensor (compute node). Returns none.
- **backward(last_node: Tensor)**: Takes in one Tensor, computes the gradients of the last Tensor (here: last_node) with respect to all prior Tensors (compute nodes). Returns none.
- **optimize("optimization_algorithm",learning_rate,momentum)**: Optimizes the entire network where the gradients are known. Returns none.

A more detailed overview is on its way. Stay tuned! 😊


## Example Code

### Train a Neural Network on the **MNIST** dataset

Import the necessary parts from Infermo

```python
from infermo import Module, Tensor, shape, Linear, max, accuracy
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
        for i in range(2):
            x = Linear(self.nn,x, num_neurons=64, addBias=True, activation='ReLU')
        x = Linear(self.nn,x,10,True,'none')
        self.logits = self.nn.softmax(x)
        self.loss = self.nn.CE(self.trueVals,self.logits)

    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _trueVals: DTypePointer[DType.float32]) -> Tensor:

        # fill the input and trueVals Tensors with theri data
        self.nn.Tensors[0].setData(_input) # bug!
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
        self.nn.optimize('sgd_momentum', lr = 0.0001, momentum = 0.9)
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
from infermo import Module, Tensor, shape

fn main():
    # init
    var nn = Module()
    var A = Tensor(shape(2,5,3))
    var B = Tensor(shape(2,3,4))

    # specify tensor entries
    A.setDataAll(2)
    B.setDataAll(3)

    # perform computation 
    var C = nn.mul(A,B)
    var D = nn.sum(C) # compute sum, since the gradient can only be computed of a scalar value
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
- build a simple Transformer with Infermo

## Usage
Make sure you have installed and [configured mojo on your environment](https://docs.modular.com/mojo/manual/get-started/index.html)

If you have problems with the Python integration (i.e. using numpy to read txt file), follow these steps: https://gist.github.com/trevorhobenshield/6bca58f947ad6115a113a97072df1a73

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
or the regression task via
```
mojo regressionTask.mojo
```