from infermo import Tensor,Module,shape,conv2d,max_pool_2d,linear,max,accuracy,DataLoader

# ######################### convolutional Neural Network (trained on MNIST dataset) ###########################################################

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
        var x = self.nn.reshape(self.input,shape(self.input.shape[0],1,28,28))                                  # 64,1,28,28
        x = conv2d(self.nn,x,out_channels=4,kernel_width=5,kernel_height=5,stride=1,padding=0,use_bias=True)   # 64,4,24,24
        x = self.nn.max_pool_2d(x,kernel_width=4,kernel_height=4,stride=4,padding=0)                            # 64,4,6,6
        x = self.nn.reshape(x,shape(64,144))                                                                    # 64,144
        x = linear(self.nn,x,64,True,'relu')                                                                    # 64,32
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
        self.nn.optimize('sgd_momentum', lr = 0.003, momentum = 0.9, weight_decay=0.001,threshold=1.0)


# train the Model
fn main() raises:

    # init
    var dl = DataLoader('./infermo/datasets/mnist.txt')
    var model = Model()

    let num_epochs = 1000
    var loss_sum: Float32 = 0
    var avg_acc: Float32 = 0
    let every = 50

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

        # forward pass through the network
        let logits = model.forward(inputs,labels)

        # for progress measures only
        loss_sum += model.loss.data.load(0)
        avg_acc += model.avg_acc
        if( epoch % every == 0):
            print("Epoch", epoch,", avgLoss =", loss_sum / every, ", avg_accuracy =", avg_acc / every)
            loss_sum = 0      
            avg_acc = 0
            # logits.print_data()
            # model.true_vals.print_data()

        # compute the grads
        model.backward()

        # take an optimization step
        model.step()

    model.nn.print_forward_durations()
    model.nn.print_backward_durations()