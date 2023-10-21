from infermo import *

fn main():

    var nn = Module()

    let dims = list(0)

    var a = Tensor(shape(3,4))
    for i in range(a.cap):
        a.data.store(i,Float32(i))

    var b = Tensor(shape(2,3,4))
    for i in range(b.cap):
        b.data.store(i,0.1*Float32(i))

    var c = nn.pow(a,b)
    var d = nn.sum(c)
    nn.forward(d)
    nn.backward(d)

    a.print_data()
    b.print_data()
    c.print_data()
    a.print_grad()
    b.print_grad()