from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp

from ..graph.tensor import Tensor
from ..graph. module import Module
from ..helpers.shape import Vec, shape

@always_inline
fn Linear(inout nn: Module, inout x: Tensor, num_neurons: Int, addBias : Bool = True, activation: String = 'ReLU') -> Tensor:
    let x_rows = x.getShape(x.num_dims - 2)
    let x_cols = x.getShape(x.num_dims - 1)
    var W = Tensor(shape(x_cols,num_neurons))

    W.initRandomHe()
    if(addBias):
        var ones = Tensor(shape(x_rows,1))
        var bias = Tensor(shape(1,num_neurons))
        ones.setDataAll(1)
        ones.requiresGradient = False
        bias.setDataAll(0.1)
        var xW = nn.mul(x,W)    
        var ob = nn.mul(ones,bias)
        x = nn.add(xW,ob)
        if(activation == 'ReLU'):
            x = nn.ReLU(x)
    else:
        x = nn.mul(x,W)
        if(activation == 'ReLU'):
            x = nn.ReLU(x)
    return x


@always_inline
fn max(inout A: Tensor) -> Tensor:

    let num_dims = A.getNum_dims()
    var new_shape = DynamicVector[Int]()

    for i in range(num_dims):
        new_shape.push_back(A.getShape(i))

    let B = Tensor(new_shape)

    let A_matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    for s in range(A.getCap() // N):
        let offset = s * N
        var max: Float32 = 0
        var argMax: Int = 0
        var idx: Int = 0
        for n in range(N):
            if(A.getData(offset + n) > max):
                max = A.getData(offset + n)
                argMax = idx
            idx += 1
        for n in range(N):
            if(n == argMax):
                B.setData(offset + n, 1)
            else:
                B.setData(offset + n, 0)
                
    return B

# compute similarity accuracy between tweo tensors along the last dimension
fn accuracy(logits: Tensor, trueVals: Tensor) -> Float32:
    var avgAcc: Float32 = 0
    let M = trueVals.getShape(trueVals.num_dims-2)
    let N = trueVals.getShape(trueVals.num_dims-1)
    for i in range(trueVals.getCap()):
        if( logits.getData(i) == Float32(1.0) and trueVals.getData(i) == Float32(1.0) ):
            avgAcc += 1
    return avgAcc / M
