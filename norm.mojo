from infermo import *

# test function for normalizing along a given dimensions
fn main():

    var nn = Module()

    let dims = list(0)

    var a = Tensor(shape(2,3,4))
    for i in range(a.cap):
        a.data.store(i,Float32(i))

    var mean = nn.mean(a,dims)
    var std = nn.std(a,dims)
    var diff = nn.sub(a,mean)
    var norm = nn.div(diff,std)
    nn.forward(norm)

    print("input:")
    a.print_shape()
    a.print_data()

    print("mean:")
    mean.print_shape()
    mean.print_data()

    print("std")
    std.print_shape()
    std.print_data()

    print("diff:")
    diff.print_shape()
    diff.print_data()

    print("norm:")
    norm.print_shape()
    norm.print_data()
    