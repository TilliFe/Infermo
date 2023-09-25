from MojoGrad import nn
from vector import Vec, shape
from node import Node

fn MLP(inout nn: nn, inout x: Node) -> Node:
    var W = Node(shape(1,4,4))
    var B = Node(shape(1,4,4))
    W.setDataAll(3)
    B.setDataAll(-1)
    var a = nn.mul(W, x)
    var b = nn.add(a,B)
    let c = nn.ReLU(b)
    return c

struct model:
    var nn: nn
    var input: Node

    fn __init__(inout self):
        self.input = Node(shape(1,4,4))
        self.input.setDataAll(2)
        self.nn = nn()

        # model architecture
        var B = MLP(self.nn,self.input)
        for i in range(2):
            B = MLP(self.nn,B)

fn main():

    var model = model()

    for _ in range(1):
        model.nn.forward()
        model.nn.backward()
        model.nn.printNodes()
