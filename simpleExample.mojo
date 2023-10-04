from infermo import Module, Tensor, shape, Linear, max, accuracy, DataLoader

fn main():
    # init
    var nn = Module()
    var A = Tensor(shape(2,5,3))
    var B = Tensor(shape(2,3,4))

    # specify tensor entries
    A.setDataAll(2)
    B.setDataAll(3)

    # perform computation 
    var C = nn.mul(A,B)
    var D = nn.sum(C) # compute sum, since the gradient can only be computed of a scalar value
    nn.forward(C)

    # print result of matrix multiplication
    C.printData()

    # compute gradients of A and B
    nn.backward(D)
    A.printGradient()
    B.printGradient()