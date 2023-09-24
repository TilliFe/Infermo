from MojoGrad import nn
from vector import Vec, shape
from node import Node

# @always_inline
# fn MLP(inout nn: nn, inout x: Node) -> Node:
#     var fw1 = nn.mul(self.A, x)
#     return nn.ReLU(fw1)

@always_inline
fn MLP(inout nn: nn, inout x: Node) -> Node:
    var W = Node(shape(10,4,4))
    return nn.mul(W, x)

# struct model:
#     var nn: nn
#     var A: Node

#     @always_inline
#     fn __init__(inout self):
#         self.nn = nn()
#         self.A = Node(shape(10,4,4))
#         self.A.setDataAll(2)

#     # @always_inline
#     # fn init(inout self):
#     #     var res = MLP(self.nn,res)

fn main():

    var nn = nn()

    var res = nn.tensor(10,4,4)
    for i in range(2):
        var x = Node(shape(10,4,4))
        var res = nn.ReLU(x)

    nn.forward()

    nn.printNodes()

    # var model = model()
    # var x = Node(shape(10,4,4))
    # model.init()
    # for i in range(1):
    #     # x.setDataAll(3)
    #     model.nn.forward()
    #     # logits.printData()
    #     # model.nn.backward()
    #     model.nn.printNodes()



    # var nn = nn()
    # var K = Node(shape(10,2,2,3))
    # var A = nn.reshape(K,Vec(10,4,3))
    # var x = Node(shape(10,3,4))
    # var C = Node(shape(10,4,4))

    # A.setDataAll(2)
    # x.setDataAll(3)
    # C.setData(4,-4)

    # var D = nn.mul(A,x)
    # var E = nn.add(C,D)
    # var F = nn.ReLU(E)
    
    # for i in range(1):
    #     nn.forward()
    #     nn.backward()
    # nn.printNodes()
