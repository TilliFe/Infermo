# from autograd import Tensor, add, sin, relu
from autograd import Tensor, cos, sin, tan, relu, mse, cross_entropy, softmax, tanh, add, sub, mul, div, pow, mmul, sum, log, exp, sqrt, abs, reshape, transpose
from autograd.utils.shape import shape

#######################################################################################################################
# main function: Testing functionality of the Engine... Dynamic Computation Graph (with conditional model architecture)
########################################################################################################################

fn main() raises:

    # init params
    let W1 = Tensor(shape(1,64)).randhe().requires_grad()
    let W2 = Tensor(shape(64,64)).randhe().requires_grad()
    let W3 = Tensor(shape(64,1)).randhe().requires_grad()
    let W_opt = Tensor(shape(64,64)).randhe().requires_grad()
    let b1 = Tensor(shape(64)).randhe().requires_grad()
    let b2 = Tensor(shape(64)).randhe().requires_grad()
    let b3 = Tensor(shape(1)).randhe().requires_grad()
    let b_opt = Tensor(shape(64)).randhe().requires_grad()


    # training
    var avg_loss = Float32(0.0)
    let every = 1000
    let num_epochs = 20000

    for epoch in range(1,num_epochs+1):

        # set input and true values
        let input = Tensor(shape(32,1)).randu(0,1).dynamic()
        let true_vals = sin(15.0 * input)

        # define model architecture
        var x = relu(input @ W1 + b1)
        x = relu(x @ W2 + b2)
        if epoch < 100:
            x = relu(x @ W_opt + b_opt)
        x = x @ W3 + b3
        let loss = mse(x,true_vals).forward()

        # print progress
        avg_loss += loss[0]
        if epoch%every == 0:
            print("Epoch:",epoch," Avg Loss: ",avg_loss/every)
            avg_loss = 0.0   
       
        # # compute gradients and optimize
        loss.backward()
        loss.optimize(0.01,"sgd")

        # clear graph
        loss.clear() 
        input.free()