from Tensor import Tensor, shape
from module import Module

# define one layer of an MLP
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