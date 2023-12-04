from .tensor import Tensor, relu, tanh, softmax, conv_2d
from .utils.shape import shape


struct Linear:
    var W: Tensor
    var bias: Tensor
    var activation: String
    var add_bias: Bool

    fn __init__(
        inout self,
        in_neurons: Int,
        out_neurons: Int,
        add_bias: Bool = True,
        activation: String = "relu",
    ) raises:
        self.W = Tensor(shape(in_neurons, out_neurons)).randhe().requires_grad()
        self.bias = Tensor(shape(out_neurons)).randhe().requires_grad()
        self.activation = activation
        self.add_bias = add_bias

    fn forward(self, x: Tensor) raises -> Tensor:
        var res = x @ self.W
        if self.add_bias:
            res = res + self.bias
        if self.activation == "relu":
            return relu(res)
        elif self.activation == "tanh":
            return tanh(res)
        elif self.activation == "softmax":
            return softmax(res)
        else:
            return res


struct Conv2d:
    var kernels: Tensor
    var bias: Tensor
    var padding: Int
    var stride: Int
    var in_channels: Int
    var out_channels: Int
    var use_bias: Bool

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_width: Int,
        kernel_height: Int,
        stride: Int,
        padding: Int,
        use_bias: Bool = False,
    ) raises:
        self.kernels = (
            Tensor(shape(out_channels, in_channels, kernel_width, kernel_height))
            .randhe()
            .requires_grad()
        )
        self.bias = Tensor(shape(out_channels, 1, 1)).randhe().requires_grad()
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias

    fn forward(self, x: Tensor) raises -> Tensor:
        var res = conv_2d(x, self.kernels, self.padding, self.stride)
        if self.use_bias:
            return res + self.bias
        return res
