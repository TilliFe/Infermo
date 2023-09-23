from MojoGrad import nn

fn main():
    var nn = nn()
    var A = nn.tensor(1,2,3)
    var x = nn.tensor(1,3,4)
    var C = nn.tensor(1,2,4)

    A.setDataAll(2)
    x.setDataAll(3)
    C.setData(4,-40)

    var D = nn.mul(A,x)
    var E = nn.add(C,D)
    let F = nn.ReLU(E)
    
    for i in range(1):
        x.setDataAll(i)
        nn.forward()
        nn.backward()
        nn.printNodes()
