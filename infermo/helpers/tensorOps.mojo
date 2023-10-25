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


struct Conv2d:
    var kernels: Tensor
    var padding: Int
    var stride: Int
    # var in_channels: Int
    # var out_channels: Int
    # var bias_raw: Tensor
    # var ones: Tensor
    # var use_bias: Bool
    
    fn __init__(inout self, inout nn: Module, in_channels: Int, out_channels: Int, kernel_width: Int, kernel_height: Int, stride: Int, padding: Int, use_bias: Bool = True):
        self.kernels = nn.tensor(shape(out_channels,in_channels,kernel_width,kernel_height))
        self.kernels.randHe()
        self.padding = padding
        self.stride = stride
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.bias_raw = nn.tensor(shape(out_channels,1))
        # self.bias_raw.randn(0.001)
        # self.use_bias = use_bias
        # self.ones = nn.tensor(shape(1,28*28), requires_grad=False)
        # self.ones.fill(1.0)
    
    fn forward(inout self, inout nn: Module, inout x: Tensor) -> Tensor:
        return nn.conv_2d(x,self.kernels,self.padding,self.stride)
        # if(self.use_bias):
        #     var bias_ones = nn.matmul(self.bias_raw,self.ones)
        #     var bias_reshaped = nn.reshape(bias_ones,shape(self.out_channels,conv.shape[2],conv.shape[3]))
        #     return nn.add(conv,bias_reshaped)
        # else:
        # return conv


struct Linear:
    var W: Tensor
    var bias: Tensor
    var activation: String

    fn __init__(inout self, inout nn: Module, in_neurons: Int, out_neurons: Int, batch_size: Int, add_bias : Bool = True, activation: String = 'relu'):
        self.W = nn.tensor(shape(in_neurons,out_neurons))
        self.W.randHe()
        self.bias = nn.tensor(shape(1,out_neurons))
        self.bias.fill(0.01)
        self.activation = activation

    fn forward(inout self, inout nn: Module, inout x: Tensor) -> Tensor:
        var xW = nn.matmul(x,self.W)    
        var added = nn.add(xW,self.bias)
        if(self.activation == 'relu'):
            return nn.relu(added)
        return added

struct Mlp:
    var neurons_per_layer: DynamicVector[Int]
    var W_id_list: DynamicVector[Int]
    var bias_id_list: DynamicVector[Int]
    var activation: String
    var num_layers: Int

    fn __init__(inout self, inout nn: Module, neurons_per_layer: DynamicVector[Int], add_bias : Bool = True, activation: String = 'relu'):
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = DynamicVector[Int]()
        self.W_id_list = DynamicVector[Int]()
        self.bias_id_list = DynamicVector[Int]()

        for i in range(self.num_layers-1):
            let W = nn.tensor(shape(neurons_per_layer[i],neurons_per_layer[i+1]))
            W.randHe()
            self.W_id_list.push_back(W.id)
            let bias = nn.tensor(shape(1,neurons_per_layer[i+1]))
            bias.fill(0.01)
            self.bias_id_list.push_back(bias.id)
            if(i == self.num_layers-1): break
            
        self.activation = activation


    @always_inline
    fn forward(inout self, inout nn: Module, inout _x: Tensor) -> Tensor:
        var x = _x
        for i in range(self.num_layers-1):
            if(i == self.num_layers-1): break
            var W = nn.nodes[self.W_id_list[i]]
            var bias = nn.nodes[self.bias_id_list[i]]
            var xW = nn.matmul(x,W)    
            var added = nn.add(xW,bias)
            if(self.activation == 'relu'):      
                x = nn.relu(added)
        return x

@always_inline
fn max(inout a: Tensor) -> Tensor:

    let num_dims = a.num_dims
    var new_shape = DynamicVector[Int]()

    for i in range(num_dims):
        new_shape.push_back(a.shape[i])

    let b = Tensor(new_shape)

    let a_matrix_size = a.shape[num_dims-2] * a.shape[num_dims-1]
    let M = a.shape[num_dims-2]
    let N = a.shape[num_dims-1]

    for s in range(a.cap // N):
        let offset = s * N
        var max: Float32 = 0
        var argMax: Int = 0
        var idx: Int = 0
        for n in range(N):
            if(a.data.load(offset + n) > max):
                max = a.data.load(offset + n)
                argMax = idx
            idx += 1
        for n in range(N):
            if(n == argMax):
                b.set_data(offset + n, 1)
            else:
                b.set_data(offset + n, 0)
                
    return b

# compute similarity accuracy between tweo tensors along the last dimension
fn accuracy(logits: Tensor, true_vals: Tensor) -> Float32:
    var avg_acc: Float32 = 0
    let M = true_vals.shape[true_vals.num_dims-2]
    let N = true_vals.shape[true_vals.num_dims-1]
    for i in range(true_vals.cap):
        if( logits.data.load(i) == Float32(1.0) and true_vals.data.load(i) == Float32(1.0) ):
            avg_acc += 1
    return avg_acc / (Float32(logits.cap) / N)

@always_inline
fn scale(inout nn: Module, inout a: Tensor, scalar: Float32) -> Tensor:
    let num_dims = a.num_dims 
    let N = a.shape[num_dims-1]
    var new_shape = DynamicVector[Int]()
    for i in range(num_dims-1):
        new_shape.push_back(a.shape[i])
    new_shape.push_back(N)

    var S = nn.tensor(new_shape)
    for s in range(a.cap // (N*N)):
        for i in range(N):
            S.set_data(s*N*N + i*N+i, scalar)
    S.requires_grad = False

    return nn.matmul(a,S)

@always_inline
fn mask(inout nn: Module, inout a: Tensor, l_value: Float32 = 0.0, u_value: Float32 = 1.0, dim: Int = 0) -> Tensor:

    let num_dims = a.num_dims
    var new_shape = DynamicVector[Int]()
    let M = a.shape[a.num_dims-2]
    let N = a.shape[a.num_dims-1]

    for i in range(num_dims-2,num_dims):
        new_shape.push_back(a.shape[i])

    var Mask = nn.tensor(new_shape)

    for i in range(M):
        for j in range(N):
            if(j >= i + dim):
                Mask.set_data(i * N + j, u_value)
            else:
                Mask.set_data(i * N + j, l_value)

    Mask.requires_grad = False

    return nn.add(a,Mask)


@always_inline
fn embed(inout nn: Module, d_vocab: Int, d_Model: Int, batch_size: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    var W_E = nn.tensor(shape(d_vocab,d_Model))
    W_E.randn(init_std)
    return nn.matmul(x,W_E)


@always_inline
fn unembed(inout nn: Module, d_vocab: Int, d_Model: Int, batch_size: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    var W_U = nn.tensor(shape(d_Model,d_vocab))
    W_U.randn(init_std)
    return nn.matmul(x, W_U)


@always_inline
fn posembed(inout nn: Module, max_ctx: Int, d_Model: Int, batch_size: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    var W_pos = nn.tensor(shape(max_ctx,d_Model))
    W_pos.randn(init_std)
    return nn.matmul(W_pos, x)


@always_inline
fn attention(inout nn: Module, d_Model: Int, num_heads: Int, d_head: Int, n_ctx: Int, batch_size: Int, seq_len: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    
    # init    
    var x_t_unreshaped = nn.transpose(x)
    var x_t = nn.reshape(x_t_unreshaped,shape(batch_size,num_heads,d_Model,seq_len))

    var W_K = nn.tensor(shape(num_heads, d_head, d_Model))
    var W_Q = nn.tensor(shape(num_heads, d_head, d_Model))
    var W_V = nn.tensor(shape(num_heads, d_head, d_Model))
    var W_O = nn.tensor(shape(num_heads*d_head, d_Model))
    W_K.randn(init_std)
    W_Q.randn(init_std) 
    W_V.randn(init_std)
    W_O.randn(init_std)  

    # attention heads
    var k = nn.matmul(W_K,x_t)     # batch_size,num_heads,d_head,seq_len
    var q = nn.matmul(W_Q,x_t)     # batch_size,num_heads,d_head,seq_len
    var v = nn.matmul(W_V,x_t)     # batch_size,num_heads,d_head,seq_len
    var k_t = nn.transpose(k)   # batch_size,num_heads,seq_len,d_head

    x = nn.matmul(k_t,q)           # batch_size,num_heads,seq_len,seq_len
    x = mask(nn, x, 0.0, -10000000.0, 1) 
    x = scale(nn, x, Float32(1.0)/(sqrt(Float32(d_head))))
    x = nn.softmax(x)           # batch_size,num_heads,seq_len,seq_len
    x = nn.matmul(v,x)             # batch_size,num_heads,d_head,seq_len

    # concatenate results of all heads
    x = nn.reshape(x,shape(batch_size, num_heads * d_head, seq_len)) 
    x = nn.transpose(x)     	# batch_size,seq_len,num_heads*d_head
    return nn.matmul(x,W_O)        # batch_size, seq_len, d_Model


# fn mlp(inout nn: Module, d_Model: Int, d_mlp: Int, batch_size: Int, seq_len: Int, init_std: Float32, inout x: Tensor) -> Tensor:

#     # init
#     var x_t = nn.transpose(x) # shape: batch_size,d_Model,seq_len

#     var W_in = nn.tensor(shape(d_mlp,d_Model))
#     W_in.randn(init_std)

#     var b_in_flat = nn.tensor(shape(d_mlp,1))
#     var ones_in = nn.tensor(shape(1,seq_len))
#     ones_in.fill(1)
#     ones_in.requires_grad = False
#     var b_in = nn.matmul(b_in_flat,ones_in)

#     var W_out = nn.tensor(shape(d_Model,d_mlp))
#     W_out.randn(init_std)

#     var b_out_flat = nn.tensor(shape(d_Model,1))
#     var ones_out = nn.tensor(shape(1,seq_len))
#     ones_out.fill(1)
#     ones_out.requires_grad = False
#     var b_out = nn.matmul(b_out_flat,ones_out)

#     # architecture
#     x = nn.matmul(W_in, x_t)     # batch_size,d_mlp,seq_len
#     x = nn.add(x,b_in)        # batch_size,d_mlp,seq_len
#     x = nn.relu(x)            # batch_size,d_mlp,seq_len
#     x = nn.matmul(W_out,x)       # batch_size,d_Model,seq_len
#     x = nn.add(x,b_out)       # batch_size,d_Model,seq_len
#     return nn.transpose(x)    # batch_size,seq_len,d_model = shape(input)



fn transformer_block(inout nn: Module, d_Model: Int, num_heads: Int, d_head: Int, n_ctx: Int, d_mlp: Int, batch_size: Int, seq_len: Int, use_attn: Bool, use_mlp: Bool, init_std: Float32, inout x: Tensor) -> Tensor:
    pass
    # var res_stream = nn.copy(x)
    # if(use_attn):
    #     x = attention(
    #         nn=nn,
    #         d_Model=d_Model,
    #         num_heads=num_heads,
    #         d_head=d_head,
    #         n_ctx=seq_len,
    #         batch_size=batch_size,
    #         seq_len=seq_len,
    #         init_std=init_std,
    #         x=x
    #     )
    # if(use_mlp):
    #     x = mlp(
    #         nn=nn,
    #         d_Model=d_Model,
    #         d_mlp=d_mlp,
    #         batch_size=batch_size,
    #         seq_len=seq_len,
    #         init_std=init_std,
    #         x=x
    #     )
    # return nn.add(res_stream,x)