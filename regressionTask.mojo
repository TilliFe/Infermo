from dv import Tensor,Module,shape,linear
from math import sin
from random import rand, seed

###################### Simple mlp for a dd regression task ########################################################

# define the Model and its behaviour
struct Model:
    var nn: Module
    var input: Tensor
    var true_vals: Tensor
    var logits: Tensor
    var loss: Tensor

    fn __init__(inout self):
        self.input = Tensor(shape(512,1))
        self.input.requires_grad = False
        self.true_vals = Tensor(shape(512,1))
        self.true_vals.requires_grad = False
        self.nn = Module()

        # define Model architecture
        var x = linear(self.nn,self.input, num_neurons=16, addbias=True, activation='relu')
        for i in range(2):
            x = linear(self.nn,x, num_neurons=32, addbias=True, activation='relu')
        self.logits = linear(self.nn,x,1,True,'none')
        self.loss = self.nn.mse(self.true_vals,self.logits)

    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _true_vals: DTypePointer[DType.float32]) -> Tensor:
        self.nn.tensors[0].set_data(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.true_vals.set_data(_true_vals)
        self.nn.forward(self.logits)
        return self.logits

    @always_inline
    fn backward(inout self):
        self.nn.backward(self.loss)

    @always_inline
    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.1, momentum = 0.9, weight_decay=0.0001)


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


# train the Model
fn main() raises:

    let dataset = DataGenerator(512)
    var model = Model()
    let num_epochs = 1000

    var loss_sum: Float32 = 0
    let every = 50

    for epoch in range(0,num_epochs):
        dataset.random(epoch)
        let logits = model.forward(dataset.x,dataset.y)
        model.backward()
        model.step()

        loss_sum += model.loss.data.load(0)
        if( epoch % every == 0 and epoch > 0):
            print("Epoch", epoch,", avgLoss = ", loss_sum / every)
            loss_sum = 0      
            # logits.print_data()
            # model.true_vals.print_data()