from dv import Module, Tensor, shape, linear, max, accuracy, DataLoader

######################## basic mlp for handwritten digit recognition (MNIST) ################################################

# define the Model and its behaviour
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

        # fill the input and true_vals tensors with theri data
        self.nn.tensors[0].set_data(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.true_vals.set_data(_true_vals)

        # one forward pass through the network
        self.nn.forward(self.logits)

        # some additional ops, not necessary for the training, just for showing the accuracy
        let one_hots = max(self.logits)
        self.avg_acc = accuracy(one_hots,self.true_vals)

        return self.logits

    @always_inline
    fn backward(inout self):
        self.nn.backward(self.loss)

    @always_inline
    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.01, momentum = 0.9, weight_decay=0.01)


# train the Model
fn main() raises:

    # init
    var dl = DataLoader('./dv/datasets/mnist.txt')
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
            index=0,
            ndims=10
        )
        let logits = model.forward(inputs,labels)
        model.backward()
        model.step()

        loss_sum += model.loss.data.load(0)
        avg_acc += model.avg_acc
        if( epoch % every == 0):
            print("Epoch", epoch,", avgLoss =", loss_sum / every, ", avg_accuracy =", avg_acc / every)
            loss_sum = 0      
            avg_acc = 0
            # logits.print_data()
            # model.true_vals.print_data()

    model.nn.print_forward_durations()
    model.nn.print_backward_durations()