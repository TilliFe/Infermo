from infermo import Module, Tensor, shape

################### Random example: compute Gradients automatically ##############################################
fn main():
    # init
    var nn = Module()
    var a = Tensor(shape(2,5,3))
    var b = Tensor(shape(2,3,4))

    # specify tensor entries
    a.fill(2)
    b.fill(3)

    # perform computation 
    var c = nn.matmul(a,b)
    var D = nn.sum(c) # compute sum, since the grad can only be computed of a scalar value
    nn.forward(c)

    # print result of matrix multiplication
    c.print_data()

    # compute grads of a and b
    nn.backward(D)
    a.print_grad()
    b.print_grad()

    nn.print_forward_durations()
    nn.print_backward_durations()