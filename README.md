# Infermo🔥

### Tensors and dynamic Neural Networks in Mojo

Infermo is a Mojo library that provides two high-level features:
- Tensor computation
- Deep neural networks built on a tape-based autograd system

Mojo currently operates on CPU only. GPU support will come soon! Infermo is currently still a Proof-of-Concept, if you encounter any bugs, feel free to create an issue or a PR. Thank you for your contribution. :)


## Available operators
The operators listed below are methods of the `Module` class, which orchestrates the gradient computation. Each operator accepts one or two `Tensor` objects as input. All binary operators accept differently shaped Tensors via broadcasting. The result of each operation is only temporarily stored as part of the dynamic computation graph. Call the `clear_cache` method, in order to clear all temporarily stored results of the graph.

- **matmul**: Performs matrix multiplication of two tensors.
- **conv_2d**: Applies a 2D convolution over an input signal composed of several input planes.
- **max_pool_2d**: Applies a 2D max pooling over an input signal composed of several input planes.
- **sum**: Computes the sum of all elements in the input tensor.
- **softmax**: Applies a softmax function along the last dimension.
- **mse**: Calculates the mean squared error between each element in the input x and target.
- **ce**: Computes cross entropy loss, often used for classification problems.
- **reshape**: Returns a tensor with the same data and number of elements as input, but with the specified shape.
- **transpose**: Transposes a Tensor along the last two dimensions.
- **mean**: Computes the mean value along a list of dimensions.
- **variance**: Computes the variance value along a list of dimensions.
- **std**: Computes the standard deviation along a list of dimensions. (TODO: backward)
- **mul**: Performs element-wise multiplication of two tensors.
- **add**: Performs element-wise addition of two tensors.
- **sub**: Performs element-wise subtraction of two tensors.
- **div**: Performs element-wise division of two tensors.
- **sqrt**: Elemtwise square root computation.
- **abs**: Computes the absolute value of each element in input.
- **pow**: Elementwise pow operation between two Tensors, or elemtwise raise to the power of some number.
- **exp2**: Computes 2 raised to the power of each element in input.
- **exp**: Computes exponential of each element in input.
- **log2**: Computes logarithm base 2 of each element in input.
- **log**: Computes natural logarithm ln(x) of each element in input.
- **sin, cos, tan, asin, acos, atan, sinh, cosh, tanh**: Elementwise trigonometric functions. 
- **relu**: Applies the rectified linear unit function element-wise. 
- **copy**: Performs a deep copy of the input Tensor.

## Advanced submodules

Submodules are abstractions of more complex computations. Each submodule has its own `forward` method which returns a Tensor.

- **Linear**: This submodule represents a dense layer of neurons. 
- **Mlp**: Similar to the dense layer, but specifically tailored for use within a transformer block.
- **Conv2d**: Executes a convolution operation with a specified tensor and adds a bias if necessary.
- **TransformerBlock, Embed, Unembed, PosEmbed**: These are the fundamental building blocks of a Transformer model. (Currently under construction)
- **DataLoader**: A utility for handling data. It reads, initializes, and loads data from a given .txt file. (TODO: dataset splitting, read from csv)


## Example code

### Train a neural network on the **MNIST** dataset

Import the necessary parts from Infermo

```python
from infermo import Module, Tensor, shape, Linear, max, accuracy
```

Define the Model architecture (simple mlp with relu activations and biases)

```python
struct Model:
    var nn: Module
    var input: Tensor
    var true_vals: Tensor
    var loss: Tensor
    var l1: Linear
    var l2: Linear
    var l3: Linear
    var l4: Linear

    fn __init__(inout self):
        self.nn = Module()
        let batch_size = 64
        self.input = self.nn.tensor(shape(batch_size,784),requires_grad=False)
        self.true_vals = self.nn.tensor(shape(batch_size,10),requires_grad=False)
        self.loss = Tensor(shape(1))
        self.l1 = Linear(self.nn,784,64,batch_size,add_bias=True,activation="relu")
        self.l2 = Linear(self.nn,64,64,batch_size,add_bias=True,activation="relu")
        self.l3 = Linear(self.nn,64,64,batch_size,add_bias=True,activation="relu")
        self.l4 = Linear(self.nn,64,10,batch_size,add_bias=True,activation="none")

    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _true_vals: DTypePointer[DType.float32]) -> Tuple[Float32,Float32,Tensor]:
        
        # important clearing methods, since our compute graph is dynamic
        self.nn.clear_cache()
        self.nn.zero_grads()

        # fill the input and true_vals tensors with data
        self.input.set_data(_input)
        self.true_vals.set_data(_true_vals)

        # define forward pass
        var x = self.l1.forward(self.nn,self.input)
        x = self.l2.forward(self.nn,x)
        x = self.l3.forward(self.nn,x)
        x = self.l4.forward(self.nn,x)
        var logits = self.nn.softmax(x)
        self.loss = self.nn.ce(self.true_vals,logits)

        # compute accuracy
        let one_hots = max(logits)
        let avg_acc = accuracy(one_hots,self.true_vals)
        return Tuple(self.loss.data.load(0),avg_acc,one_hots)   
```

Read in the MNIST dataset from a file, initialize the Model and define the number of epochs, then let it train on a randomly generated batch of data.

```python
fn main() raises:

    # init
    var dl = DataLoader('./infermo/datasets/mnist.txt')
    var model = Model()

    let num_epochs = 10000
    var loss_sum: Float32 = 0
    var avg_acc: Float32 = 0
    let every = 500

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
            index=0,
            ndims=10
        )
        let res = model.forward(inputs,labels)
        model.nn.backward(model.loss)
        model.nn.optimize('sgd_momentum', lr = 0.003, momentum = 0.9, weight_decay=0.001)

        loss_sum += res.get[0,Float32]()
        avg_acc += res.get[1,Float32]()
        if( epoch % every == 0):
            print("Epoch", epoch,", avgLoss =", loss_sum / every, ", avg_accuracy =", avg_acc / every)
            loss_sum = 0      
            avg_acc = 0
```

### Super simple example

If that was a bit too much, here is a simpler example of a basic multiplication between two tensors and their respective gradient computation.

```python
from infermo import Module, Tensor, shape

fn main():
    # init
    var nn = Module()
    var a = nn.tensor(shape(2,3))
    var b = nn.tensor(shape(2,2,3,4))

    # specify tensor entries
    a.fill(2.0)
    b.fill(3.0)

    # perform computation 
    var c = nn.matmul(a,b)
    var D = nn.sum(c) # compute sum, since the grad can only be computed of a scalar value

    # print result of matrix multiplication
    c.print_data()

    # compute grads of a and b
    nn.backward(D)
    a.print_grad()
    b.print_grad()
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

Once this is set up, you can directly try out one of the tests e.g. the MNIST training setup with the following command

```
mojo train_MNIST.mojo
```