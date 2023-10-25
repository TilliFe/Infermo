from infermo import Module, Tensor, shape, Linear, max, accuracy, DataLoader, Conv2d

########################## Convolutional Neural Network (trained on MNIST dataset) ###########################################################

# define the Model and its behaviour
struct Model:
    var nn: Module
    var input: Tensor
    var true_vals: Tensor
    var loss: Tensor
    var conv2d: Conv2d
    var l1: Linear
    var l2: Linear

    fn __init__(inout self):
        self.nn = Module()
        let batch_size = 64
        self.input = self.nn.tensor(shape(batch_size,784),requires_grad=False)
        self.true_vals = self.nn.tensor(shape(batch_size,10),requires_grad=False)
        self.loss = Tensor(shape(1))
        self.conv2d = Conv2d(self.nn,in_channels=1,out_channels=4,kernel_width=5,kernel_height=5,stride=1,padding=0,use_bias=True)
        self.l1 = Linear(self.nn,144,32,batch_size,add_bias=True,activation="relu")
        self.l2 = Linear(self.nn,32,10,batch_size,add_bias=True,activation="relu")
        
    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _true_vals: DTypePointer[DType.float32]) -> Tuple[Float32,Float32,Tensor]:
        
        # important clearing methods, since our compute graph is dynamic
        self.nn.clear_cache()
        self.nn.zero_grads()

        # fill the input and true_vals tensors with data
        self.input.set_data(_input) # self.nn.leaf_nodes[0].set_data(_input) # self.input.set_data(_input)
        self.true_vals.set_data(_true_vals)

        # define forward pass
        var x = self.nn.reshape(self.input,shape(self.input.shape[0],1,28,28))              # 64,1,28,28
        x = self.conv2d.forward(self.nn,x)                                                  # 64,4,24,24
        x = self.nn.max_pool_2d(x,kernel_width=4,kernel_height=4,stride=4,padding=0)        # 64,4,6,6
        x = self.nn.reshape(x,shape(64,144))                                                # 64,144
        x = self.l1.forward(self.nn,x)                                                      # 64,32
        x = self.l2.forward(self.nn,x)
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
        let res = model.forward(inputs,labels)
        model.nn.backward(model.loss)
        model.nn.optimize('sgd_momentum', lr = 0.01, momentum = 0.9, weight_decay=0.001)

        loss_sum += res.get[0,Float32]()
        avg_acc += res.get[1,Float32]()
        if( epoch % every == 0):
            print("Epoch", epoch,", avgLoss =", loss_sum / every, ", avg_accuracy =", avg_acc / every)
            loss_sum = 0      
            avg_acc = 0

    model.nn.print_forward_durations()
    model.nn.print_backward_durations()