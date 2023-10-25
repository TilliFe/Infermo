from infermo import Module, Tensor, shape, Linear, max, accuracy, DataLoader

######################## basic mlp for handwritten digit recognition (MNIST) ################################################

# define the Model and its behaviour
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
        self.input.set_data(_input) # self.nn.leaf_nodes[0].set_data(_input) # self.input.set_data(_input)
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

# train the Model
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