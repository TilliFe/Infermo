from Tensor import Tensor, shape
from module import Module

# define one layer of an MLP
fn Linear(inout nn: Module, inout x: Tensor, num_neurons: Int, addBias : Bool = True, activation: String = 'ReLU') -> Tensor:
    let x_rows = x.getShape(x.num_dims - 2)
    let x_cols = x.getShape(x.num_dims - 1)
    var W = Tensor(shape(num_neurons,x_rows))

    W.initRandom(-0.1,0.1)
    if(addBias):
        var bias = Tensor(shape(num_neurons,1))
        var ones = Tensor(shape(1,x_cols))
        bias.initRandom(-0.1,0.1)
        ones.setDataAll(1)
        ones.requiresGradient = False
        var wx = nn.mul(W,x)    
        var bo = nn.mul(bias,ones)
        x = nn.add(wx,bo)
        if(activation == 'ReLU'):
            x = nn.ReLU(x)
    else:
        x = nn.mul(W,x)
        if(activation == 'ReLU'):
            x = nn.ReLU(x)
    return x