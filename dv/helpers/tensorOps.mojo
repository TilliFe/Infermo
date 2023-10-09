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
fn conv_2d(inout nn: Module, inout x: Tensor, out_channels: Int, kernel_width: Int, kernel_height: Int, stride: Int, padding: Int, use_bias: Bool = True) -> Tensor:
    
    # init
    let batch_size = x.shape[0]
    let in_channels = x.shape[1]
    var kernels = Tensor(shape(out_channels,in_channels,kernel_width,kernel_height))
    kernels.randHe()
    var conv = nn.conv_2d(x,kernels,padding,stride)

    if(use_bias):
        var bias_raw = Tensor(shape(out_channels,1))
        bias_raw.randn(0.001)
        var ones = Tensor(shape(1,conv.shape[2]*conv.shape[3]))
        ones.fill(1.0)
        ones.requires_grad = False
        var bias_ones = nn.mul(bias_raw,ones)
        var bias = nn.reshape(bias_ones,shape(out_channels,conv.shape[2],conv.shape[3]))
        return nn.add(conv,bias)
    else:
        return conv


@always_inline
fn linear(inout nn: Module, inout x: Tensor, num_neurons: Int, addbias : Bool = True, activation: String = 'relu') -> Tensor:
    let x_rows = x.shape[x.num_dims - 2]
    let x_cols = x.shape[x.num_dims - 1]
    var W = Tensor(shape(x_cols,num_neurons))

    W.randHe()
    if(addbias):
        var ones = Tensor(shape(x_rows,1))
        var bias = Tensor(shape(1,num_neurons))
        ones.fill(1)
        ones.requires_grad = False
        bias.fill(0.1)
        var xW = nn.mul(x,W)    
        var ob = nn.mul(ones,bias)
        x = nn.add(xW,ob)
        if(activation == 'relu'):
            x = nn.relu(x)
    else:
        x = nn.mul(x,W)
        if(activation == 'relu'):
            x = nn.relu(x)
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

    var S = Tensor(new_shape)
    for s in range(a.cap // (N*N)):
        for i in range(N):
            S.set_data(s*N*N + i*N+i, scalar)
    S.requires_grad = False

    return nn.mul(a,S)

@always_inline
fn mask(inout nn: Module, inout a: Tensor, l_value: Float32 = 0.0, u_value: Float32 = 1.0, dim: Int = 0) -> Tensor:

    let num_dims = a.num_dims
    var new_shape = DynamicVector[Int]()
    let M = a.shape[a.num_dims-2]
    let N = a.shape[a.num_dims-1]

    for i in range(num_dims-2,num_dims):
        new_shape.push_back(a.shape[i])

    var Mask = Tensor(new_shape)

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
    var W_E = Tensor(shape(d_vocab,d_Model))
    W_E.randn(init_std)
    return nn.mul(x,W_E)


@always_inline
fn unembed(inout nn: Module, d_vocab: Int, d_Model: Int, batch_size: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    var W_U = Tensor(shape(d_Model,d_vocab))
    W_U.randn(init_std)
    return nn.mul(x, W_U)


@always_inline
fn posembed(inout nn: Module, max_ctx: Int, d_Model: Int, batch_size: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    var W_pos = Tensor(shape(max_ctx,d_Model))
    W_pos.randn(init_std)
    return nn.mul(W_pos, x)


@always_inline
fn attention(inout nn: Module, d_Model: Int, num_heads: Int, d_head: Int, n_ctx: Int, batch_size: Int, seq_len: Int, init_std: Float32, inout x: Tensor) -> Tensor:
    
    # init    
    var x_t_unreshaped = nn.transpose(x)
    var x_t = nn.reshape(x_t_unreshaped,shape(batch_size,num_heads,d_Model,seq_len))

    var W_K = Tensor(shape(num_heads, d_head, d_Model))
    var W_Q = Tensor(shape(num_heads, d_head, d_Model))
    var W_V = Tensor(shape(num_heads, d_head, d_Model))
    var W_O = Tensor(shape(num_heads*d_head, d_Model))
    W_K.randn(init_std)
    W_Q.randn(init_std) 
    W_V.randn(init_std)
    W_O.randn(init_std)  

    # attention heads
    var k = nn.mul(W_K,x_t)     # batch_size,num_heads,d_head,seq_len
    var q = nn.mul(W_Q,x_t)     # batch_size,num_heads,d_head,seq_len
    var v = nn.mul(W_V,x_t)     # batch_size,num_heads,d_head,seq_len
    var k_t = nn.transpose(k)   # batch_size,num_heads,seq_len,d_head

    x = nn.mul(k_t,q)           # batch_size,num_heads,seq_len,seq_len
    x = mask(nn, x, 0.0, -10000000.0, 1) 
    x = scale(nn, x, Float32(1.0)/(sqrt(Float32(d_head))))
    x = nn.softmax(x)           # batch_size,num_heads,seq_len,seq_len
    x = nn.mul(v,x)             # batch_size,num_heads,d_head,seq_len

    # concatenate results of all heads
    x = nn.reshape(x,shape(batch_size, num_heads * d_head, seq_len)) 
    x = nn.transpose(x)     	# batch_size,seq_len,num_heads*d_head
    return nn.mul(x,W_O)        # batch_size, seq_len, d_Model


fn mlp(inout nn: Module, d_Model: Int, d_mlp: Int, batch_size: Int, seq_len: Int, init_std: Float32, inout x: Tensor) -> Tensor:

    # init
    var x_t = nn.transpose(x) # shape: batch_size,d_Model,seq_len

    var W_in = Tensor(shape(d_mlp,d_Model))
    W_in.randn(init_std)

    var b_in_flat = Tensor(shape(d_mlp,1))
    var ones_in = Tensor(shape(1,seq_len))
    ones_in.fill(1)
    ones_in.requires_grad = False
    var b_in = nn.mul(b_in_flat,ones_in)

    var W_out = Tensor(shape(d_Model,d_mlp))
    W_out.randn(init_std)

    var b_out_flat = Tensor(shape(d_Model,1))
    var ones_out = Tensor(shape(1,seq_len))
    ones_out.fill(1)
    ones_out.requires_grad = False
    var b_out = nn.mul(b_out_flat,ones_out)

    # architecture
    x = nn.mul(W_in, x_t)     # batch_size,d_mlp,seq_len
    x = nn.add(x,b_in)        # batch_size,d_mlp,seq_len
    x = nn.relu(x)            # batch_size,d_mlp,seq_len
    x = nn.mul(W_out,x)       # batch_size,d_Model,seq_len
    x = nn.add(x,b_out)       # batch_size,d_Model,seq_len
    return nn.transpose(x)    # batch_size,seq_len,d_model = shape(input)


fn transformer_block(inout nn: Module, d_Model: Int, num_heads: Int, d_head: Int, n_ctx: Int, d_mlp: Int, batch_size: Int, seq_len: Int, use_attn: Bool, use_mlp: Bool, init_std: Float32, inout x: Tensor) -> Tensor:
    var res_stream = x
    if(use_attn):
        x = attention(
            nn=nn,
            d_Model=d_Model,
            num_heads=num_heads,
            d_head=d_head,
            n_ctx=seq_len,
            batch_size=batch_size,
            seq_len=seq_len,
            init_std=init_std,
            x=x
        )
    if(use_mlp):
        x = mlp(
            nn=nn,
            d_Model=d_Model,
            d_mlp=d_mlp,
            batch_size=batch_size,
            seq_len=seq_len,
            init_std=init_std,
            x=x
        )
    return nn.add(res_stream,x)