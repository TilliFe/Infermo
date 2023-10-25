from infermo import *

# test function for normalizing along a given dimensions
fn main():

    var nn = Module()

    let dims = list(0)

    var a = nn.tensor(shape(2,3,4))
    for i in range(a.cap):
        a.data.store(i,Float32(i))

    var mean = nn.mean(a,dims)
    var std = nn.std(a,dims)
    var diff = nn.sub(a,mean)
    let norm = nn.div(diff,std)

    print("input:")
    a.print_data()

    print("mean:")
    mean.print_data()

    print("std")
    std.print_data()

    print("diff:")
    diff.print_data()

    print("norm:")
    norm.print_data()
    