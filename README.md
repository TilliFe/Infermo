# Infermo🔥

### Tensors and dynamic Neural Networks in Mojo

Infermo is a Mojo library that provides two high-level features:
- Tensor computation
- Deep neural networks built on a tape-based autograd system

Mojo currently operates on CPU only. GPU support will come soon! Infermo is currently still a Proof-of-Concept, if you encounter any bugs, feel free to create an issue or a PR. Thank you for your contribution. :)

## Available Operators
The operators listed below are methods of the `Module` class, which orchestrates both forward and backward computations. Each operator accepts one or two `Tensor` objects as input.

- **matmul**: Performs matrix multiplication of two tensors.
- **conv_2d**: Applies a 2D convolution over an input signal composed of several input planes.
- **max_pool_2d**: Applies a 2D max pooling over an input signal composed of several input planes.
- **sum**: Returns the sum of all elements in the input tensor.
- **softmax**: Applies a softmax function. It is applied to all slices along dim, and will re-scale them so that the element-wise sum is 1.
- **mse**: Calculates the mean squared error between each element in the input x and target.
- **ce**: Computes cross entropy loss, often used for classification problems.
- **reshape**: Returns a tensor with the same data and number of elements as input, but with the specified shape.
- **transpose**: Returns a tensor that is a transposed version of input. The given dimensions are swapped.
- **mean**: Computes the mean value along a list of dimensions. (TODO: backward)
- **variance**: Computes the variance value along a list of dimensions. (TODO: backward)
- **std**: Computes the standard deviation along a list of dimensions. (TODO: backward)
- **mul**: Performs element-wise multiplication of two tensors.
- **add**: Performs element-wise addition of two tensors.
- **sub**: Performs element-wise subtraction of two tensors.
- **div**: Performs element-wise division of two tensors.
- **sqrt**: Returns a new tensor with the square-root of the elements of input.
- **abs**: Computes the absolute value of each element in input.
- **exp2**: Computes 2 raised to the power of each element in input.
- **exp**: Computes exponential of each element in input.
- **log2**: Computes logarithm base 2 of each element in input.
- **log**: Computes natural logarithm ln(x) of each element in input.
- **sin, cos, tan, asin, acos, atan, sinh, cosh, tanh**: Trigonometric functions. Each computes trigonometric function of each element in input.
- **relu**: Applies the rectified linear unit function element-wise. It replaces all negative values in the tensor with zero.
- **copy**: Performs a deep copy of the input Tensor.

**Note**: All binary operators in this library are designed to handle tensors of different `shape` through broadcasting.

## Advanced Operators

- **linear**: This operator represents a dense layer of neurons. It can be used with or without the ReLU activation function, providing flexibility in network design.
- **mlp**: Similar to the dense operator, but specifically tailored for use within a transformer block.
- **conv2d**: Executes a convolution operation with a specified tensor, commonly used in convolutional neural networks.
- **transformer_block, embed, unembed, pos_embed**: These are the fundamental building blocks of a Transformer model, providing the necessary operations for effective sequence transduction.
- **DataLoader**: A utility for handling data. It reads, initializes, and loads data from a given .txt file, simplifying the data preparation process.


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
        var x = linear(self.nn,self.input, num_neurons=64, add_bias=True, activation='relu')
        for i in range(2):
            x = linear(self.nn,x, num_neurons=64, add_bias=True, activation='relu')
        x = linear(self.nn,x,10,True,'none')
        self.logits = self.nn.softmax(x)
        self.loss = self.nn.ce(self.true_vals,self.logits)

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
    var C = nn.matmul(A,B)
    var D = nn.sum(C) # compute sum, since the gradient can only be computed of a scalar value
    nn.forward(C)

    # print result of matrix multiplication
    C.print_data()

    # compute gradients of A and B
    nn.backward(D)
    A.print_grad()
    B.print_grad()
```


## Usage

Make sure you have installed and [configured the latest version of mojo on your environment](https://docs.modular.com/mojo/manual/get-started/index.html)


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
