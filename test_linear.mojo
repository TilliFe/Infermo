from autograd import (
    Tensor,
    Linear,
    sin,
    mse,
    max_pool_2d,
)
from autograd.utils.shape import shape


fn main() raises:

    # init params
    let l1 = Linear(1,64)
    let l2 = Linear(64,64)
    let l3 = Linear(64,1,activation='none')
    
    # training
    var avg_loss = Float32(0.0)
    let every = 1000
    let num_epochs = 20000

    for epoch in range(1,num_epochs+1):

        # set input and true values
        let input = Tensor(shape(32,1)).randu(0,1).dynamic()
        let true_vals = sin(15.0 * input)

        # define model architecture
        var x = l1.forward(input)
        x = l2.forward(x)
        x = l3.forward(x)
        let loss = mse(x,true_vals)

        # print progress
        avg_loss += loss[0]
        if epoch%every == 0:
            print("Epoch:",epoch," Avg Loss: ",avg_loss/every)
            avg_loss = 0.0
            

        # compute gradients and optimize
        loss.backward()
        loss.optimize(0.01,"sgd")

        # clear graph
        loss.clear()
        input.free()