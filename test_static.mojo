# from autograd import Tensor, add, sin, relu
from autograd import (
    Tensor,
    cos,
    sin,
    tan,
    relu,
    mse,
    cross_entropy,
    softmax,
    tanh,
    add,
    sub,
    mul,
    div,
    pow,
    mmul,
    sum,
    log,
    exp,
    sqrt,
    abs,
    reshape,
    transpose,
)
from autograd.utils.shape import shape
import math

###############################################################################################################################
# main function: Testing functionality of the Engine... Static Computation Graph
###############################################################################################################################


fn main() raises:
    # init params
    let W1 = Tensor(shape(1, 64)).randhe().requires_grad()
    let W2 = Tensor(shape(64, 64)).randhe().requires_grad()
    let W3 = Tensor(shape(64, 1)).randhe().requires_grad()
    let b1 = Tensor(shape(64)).randhe().requires_grad()
    let b2 = Tensor(shape(64)).randhe().requires_grad()
    let b3 = Tensor(shape(1)).randhe().requires_grad()

    # training
    var avg_loss = Float32(0.0)
    let every = 1000
    let num_epochs = 20000

    # set input and true values
    let input = Tensor(shape(32, 1)).randu(0, 1)
    let true_vals = Tensor(shape(32, 1))

    # statically define model architecture
    var x = relu(input @ W1 + b1)
    x = relu(x @ W2 + b2)
    x = x @ W3 + b3
    let loss = mse(x, true_vals)

    # train the model
    for epoch in range(1, num_epochs + 1):
        # get new input data and compute true values
        for i in range(input.randu(0, 1).capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        # print progress
        avg_loss += loss.forward_static()[0]
        if epoch % every == 0:
            print("Epoch:", epoch, " Avg Loss: ", avg_loss / every)
            avg_loss = 0.0

        # compute gradients and optimize
        loss.backward()
        loss.optimize(0.01, "sgd")
