# Infermo🔥

### AutoDiff with Tensors in pure Mojo!

- Independence from external libraries.
- Dynamic, object-oriented Model definition
- Automatic gradient computation.

Infermo is currently a Proof-of-Concept. While it’s mainly functional, it’s still under optimization and currently operates on CPU only.

## Overview

**High Level Operators**:

- mlp, linear Layer with relu activation
- (masked) Transformer Block
- 2d Convolution, Max Pooling (still naive implementation)

**General Structs**:

- **Tensor**: A Tensor is a multidimensional array. It is either a separate data holder or part of a computation graph (see Module).
- **Module**: The Module is the Computation Graph and stores the relations between the Tensors (nodes).
- **DataLoader**: The DataLoader reads in data from a .txt file (.csv coming soon). The read-in data is stored in a buffer. Batches of data get generated based on the specified sub-rows and sub-columns.

**Tensor methods**:

- **initialization**: Via the shape(\*Ints) function, e.g., 'var new_tensor = Tensor(shape(2,3,4))'
- **data.load(index: Int)**: Returns the element at the index position. Attention: Does not take the shape of the Tensor into consideration.
- **set_data(index: Int, value: Float32)**: Sets the entry at the index position.
- **set_data(new_data: DTypeFloat32[DType.float32])**: If the capacities of the Tensors data and new_data are the same, the data in the Tensor gets efficiently replaced by new_data.
- **etc.** (We recommend checking out the Tensor struct in the graph directory)

**Module Operators**:
Operators include mul, add, sum, relu, softmax, reshape (broadcast), mse, CE, transpose. Each operator takes one or two Tensors and performs a specific operation. For more details on each operator, please refer to the source code. \
_Important:_ First, an object of type Module needs to be initialized, like so: 'var nn = Module()'. Then we can call the methods for example as follows: 'var res = nn.mul(tensor1,tensor2).

- **forward(last_node: Tensor)**: Takes in one Tensor, computes one forward pass through the Network till the specified Tensor (compute node). Returns none.
- **backward(last_node: Tensor)**: Takes in one Tensor, computes the gradients of the last Tensor (here: last_node) with respect to all prior Tensors (compute nodes). Returns none.
- **optimize("optimization_algorithm",learning_rate,momentum)**: Optimizes the entire network where the gradients are known. Returns none.

A more detailed overview is on its way. Stay tuned! 😊

## Example Code

### Train a Neural Network on the **MNIST** dataset

Import the necessary parts from Infermo

```python
from infermo import Module, Tensor, shape, linear, max, accuracy
```

Define the Model architecture (simple mlp with relu activations and biases)

```python
struct Model:
    var nn: Module
    var input: Tensor
    var true_vals: Tensor
    var logits: Tensor
    var loss: Tensor
    var avg_acc: Float32

    fn __init__(inout self):
        self.input = Tensor(shape(64,784))
        self.input.requires_grad = False
        self.true_vals = Tensor(shape(64,10))
        self.true_vals.requires_grad = False
        self.nn = Module()
        self.avg_acc = 0

        # define Model architecture
        var x = linear(self.nn,self.input, num_neurons=64, addBias=True, activation='relu')
        for i in range(2):
            x = linear(self.nn,x, num_neurons=64, addBias=True, activation='relu')
        x = linear(self.nn,x,10,True,'none')
        self.logits = self.nn.softmax(x)
        self.loss = self.nn.CE(self.true_vals,self.logits)

    @always_inline
    fn forward(inout self, _input: DTypePointer[DType.float32], _true_vals: DTypePointer[DType.float32]) -> Tensor:

        # fill the input and true_vals Tensors with theri data
        self.nn.Tensors[0].set_data(_input) # bug!
        self.true_vals.set_data(_true_vals)

        # one forward pass through the network
        self.nn.forward(self.logits)

        # some additional ops, not necessary for the training, just for showing the accuracy
        let one_hots = one_hot(self.logits)
        self.avg_acc = accuracy(one_hots,self.true_vals)

        return self.logits

    @always_inline
    fn backward(inout self):
        self.nn.backward(self.loss)

    @always_inline
    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.0001, momentum = 0.9)
```

Read in the MNIST dataset from a file, initialize the Model and define the number of epochs, then let it train on a randomly generated batch of data.

```python
fn main()raises:

    # init
    var dl = DataLoader('./datasets/mnist.txt')
    var model = Model()

    let num_epochs = 1000
    var loss_sum: Float32 = 0
    var avg_acc: Float32 = 0
    let every = 100

    for epoch in range(1,num_epochs+1):
        # load a batch of images into the Model
        let inputs = dl.load(
            batch_size=64,
            start=1, # regarding the columns of the dataset
            end=785,
            scalingFactor=Float32(1)/Float32(255)
        )
        # load the labels for the images (one_hot encded from 0 to 9)
        let labels = dl.one_hot(
            batch_size=64,
            index=0, # regarding the columm of the labels in the dataset
            ndims=10
        )
        let logits = model.forward(inputs,labels)
        model.backward()
        model.step()

        loss_sum += model.loss.data.load(0)
        avg_acc += model.avg_acc
        if( epoch % every == 0):
            print("Epoch", epoch,", AvgLoss =", loss_sum / every, ", AvgAccuracy =", avg_acc / every)
            loss_sum = 0
            avg_acc = 0
            # logits.print_data()
            # model.true_vals.print_data()
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
    A.fill(2)
    B.fill(3)

    # perform computation
    var C = nn.mul(A,B)
    var D = nn.sum(C) # compute sum, since the gradient can only be computed of a scalar value
    nn.forward(C)

    # print result of matrix multiplication
    C.print_data()

    # compute gradients of A and B
    nn.backward(D)
    A.print_grad()
    B.print_grad()
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
