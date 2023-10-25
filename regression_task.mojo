from infermo import Tensor,Module,shape,list,Mlp
from math import sin
from random import rand, seed

###################### Simple mlp for a dd regression task ########################################################

# define the Model and its behaviour
struct Model:
    var nn: Module
    var input: Tensor
    var true_vals: Tensor
    var mlp: Mlp
    var loss: Tensor

    fn __init__(inout self, batch_size: Int):
        self.nn = Module()
        self.input = self.nn.tensor(shape(batch_size,1),requires_grad=False)
        self.true_vals = self.nn.tensor(shape(batch_size,1),requires_grad=False)
        self.mlp = Mlp(self.nn,list(1,32,32,32,1))
        self.loss = Tensor(shape(1))

    fn forward(inout self, _input: DTypePointer[DType.float32], _true_vals: DTypePointer[DType.float32]):
        
        # clear dynamic sub graph
        self.nn.clear_cache()
        self.nn.zero_grads()

        # model architecture
        self.input.set_data(_input)
        self.true_vals.set_data(_true_vals)
        var logits = self.mlp.forward(self.nn,self.input) 
        self.loss = self.nn.mse(self.true_vals,logits)


# Data Generator for a simple regression problem, the (nonlinear) function looks like: /\/\/\/ and the network shall predict the proper y-values for some random x in [0,1]
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
        let min = 0
        let max = 1
        for i in range(self.size):
            let x_rand = self.x.load(i) * (max - min) + min
            self.x.store(i, x_rand)
            let res = 0.5 + 0.5*sin(15*x_rand)
            self.y.store(i, res) 


# train the Model
fn main() raises:

    let batch_size=64
    let dataset = DataGenerator(batch_size)
    var model = Model(batch_size)
    let num_epochs = 100000

    var loss_sum: Float32 = 0
    let every = 1000

    for epoch in range(0,num_epochs):
        dataset.random(epoch)
        model.forward(dataset.x,dataset.y)
        model.nn.backward(model.loss)
        model.nn.optimize('sgd_momentum', lr = 0.001, momentum = 0.9, weight_decay=0.001)

        loss_sum += model.loss.data.load(0)
        if( epoch % every == 0 and epoch > 0):
            print("Epoch", epoch,", avgLoss = ", loss_sum / every)
            loss_sum = 0      

    model.nn.print_backward_durations()