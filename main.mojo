from MojoGrad import nn

fn main():
    var nn = nn()
    var A = nn.tensor(10,200,300)
    var x = nn.tensor(10,300,400)
    var C = nn.tensor(10,200,400)

    A.setDataAll(2)
    x.setDataAll(3)
    C.setData(4,-4)

    var D = nn.mul(A,x)
    var E = nn.add(C,D)
    let F = nn.ReLU(E)
    
    for i in range(10):
        nn.forward()
        nn.backward()
    nn.printNodes()
