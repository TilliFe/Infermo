from infermo import Module, Tensor, shape

################### Random example: compute Gradients automatically ##############################################
fn main():
    # init
    var nn = Module()
    var a = nn.tensor(shape(2,2,2,3))
    var b = nn.tensor(shape(2,2,3,4))

    # specify tensor entries
    a.fill(2.0)
    b.fill(3.0)

    # perform computation 
    var c = nn.matmul(a,b)
    var D = nn.sum(c) # compute sum, since the grad can only be computed of a scalar value

    # print result of matrix multiplication
    c.print_data()

    # compute grads of a and b
    nn.backward(D)
    a.print_grad()
    b.print_grad()