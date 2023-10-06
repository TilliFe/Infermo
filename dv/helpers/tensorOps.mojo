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
    return avgAcc / (Float32(logits.cap) / N)

@always_inline
fn scale(inout nn: Module, inout A: Tensor, scalar: Float32) -> Tensor:
    let num_dims = A.getNum_dims() 
    let N = A.getShape(num_dims-1)
    var new_shape = DynamicVector[Int]()
    for i in range(num_dims-1):
        new_shape.push_back(A.getShape(i))
    new_shape.push_back(N)

    var S = Tensor(new_shape)
    for s in range(A.cap // (N*N)):
        for i in range(N):
            S.setData(s*N*N + i*N+i, scalar)
    S.requiresGradient = False

    return nn.mul(A,S)

@always_inline
fn mask(inout nn: Module, inout A: Tensor, l_value: Float32 = 0.0, u_value: Float32 = 1.0, dim: Int = 0) -> Tensor:

    let num_dims = A.getNum_dims()
    var new_shape = DynamicVector[Int]()
    let M = A.getShape(A.num_dims-2)
    let N = A.getShape(A.num_dims-1)

    for i in range(num_dims):
        new_shape.push_back(A.getShape(i))
    # new_shape[num_dims - 2] = N

    var Mask = Tensor(new_shape)

    for s in range(A.getCap() // (M*N)):
        let offset = s * N * N

        for i in range(M):
            for j in range(N):
                if(j >= i + dim):
                    Mask.setData(offset + i * N + j, u_value)
                else:
                    Mask.setData(offset + i * N + j, l_value)

    Mask.requiresGradient = False

    return nn.add(A,Mask)


from math import sqrt
from random import random_float64, seed
from memory import memset_zero

@always_inline
fn Embed(inout nn: Module, d_vocab: Int, d_model: Int, batch_size: Int, inout x: Tensor) -> Tensor:
    var W_E = Tensor(shape(d_vocab,d_model))
    W_E.initRandomHe()
    # var W_E = nn.reshape(W_E_raw, shape(batch_size,d_vocab,d_model))
    return nn.mul(x,W_E)

@always_inline
fn Unembed(inout nn: Module, d_vocab: Int, d_model: Int, batch_size: Int, inout x: Tensor) -> Tensor:
    var W_U = Tensor(shape(d_model,d_vocab))
    W_U.initRandomHe()
    # var W_U = nn.reshape(W_U_raw,shape(batch_size,d_model,d_vocab))
    return nn.mul(x, W_U)

@always_inline
fn PosEmbed(inout nn: Module, max_ctx: Int, d_model: Int, batch_size: Int, inout x: Tensor) -> Tensor:
    var W_pos_raw = Tensor(shape(max_ctx,d_model))
    W_pos_raw.initRandomHe()
    var W_pos = nn.reshape(W_pos_raw,shape(batch_size,max_ctx,d_model))
    return nn.mul(W_pos, x)


@always_inline
fn Attention(inout nn: Module, d_model: Int, num_heads: Int, d_head: Int, n_ctx: Int, batch_size: Int, seq_len: Int, inout x: Tensor) -> Tensor:
    
    # init    
    var x_t_unreshaped = nn.transpose(x)
    var x_t = nn.reshape(x_t_unreshaped,shape(batch_size,num_heads,d_model,seq_len))

    var W_K = Tensor(shape(num_heads, d_head, d_model))
    var W_Q = Tensor(shape(num_heads, d_head, d_model))
    var W_V = Tensor(shape(num_heads, d_head, d_model))
    var W_O = Tensor(shape(num_heads*d_head, d_model))
    W_K.initRandomHe()
    W_Q.initRandomHe() 
    W_V.initRandomHe()
    W_O.initRandomHe()  

    # attention heads
    var k = nn.mul(W_K,x_t)     # batch_size,num_heads,d_head,seq_len
    var q = nn.mul(W_Q,x_t)     # batch_size,num_heads,d_head,seq_len
    var v = nn.mul(W_V,x_t)     # batch_size,num_heads,d_head,seq_len
    var k_t = nn.transpose(k)   # batch_size,num_heads,seq_len,d_head

    x = nn.mul(k_t,q)           # batch_size,num_heads,seq_len,seq_len
    x = mask(nn, x, 0.0, -1000000.0, 1) 
    x = scale(nn, x, Float32(1.0)/Float32(d_head))
    x = nn.softmax(x)           # batch_size,num_heads,seq_len,seq_len
    x = nn.mul(v,x)             # batch_size,num_heads,d_head,seq_len

    # concatenate results of all heads
    x = nn.reshape(x,shape(batch_size, num_heads * d_head, seq_len)) 
    x = nn.transpose(x)     	# batch_size,seq_len,num_heads*d_head
    
    x = nn.mul(x,W_O)           # batch_size, seq_len, d_model

    return x


fn MLP(inout nn: Module, d_model: Int, d_mlp: Int, batch_size: Int, seq_len: Int, inout x: Tensor) -> Tensor:

    # init
    var x_t = nn.transpose(x) # shape: batch_size,d_model,seq_len

    var W_in = Tensor(shape(d_mlp,d_model))
    W_in.initRandomHe()

    var b_in_flat = Tensor(shape(d_mlp,1))
    b_in_flat.setDataAll(0.0)
    var ones_in = Tensor(shape(1,seq_len))
    ones_in.setDataAll(1)
    ones_in.requiresGradient = False
    var b_in = nn.mul(b_in_flat,ones_in)

    var W_out = Tensor(shape(d_model,d_mlp))
    W_out.initRandomHe()

    var b_out_flat = Tensor(shape(d_model,1))
    b_out_flat.setDataAll(0.0)
    var ones_out = Tensor(shape(1,seq_len))
    ones_out.setDataAll(1)
    ones_out.requiresGradient = False
    var b_out = nn.mul(b_out_flat,ones_out)

    # architecture
    x = nn.mul(W_in, x_t)     # batch_size,d_mlp,seq_len
    x = nn.add(x,b_in)        # batch_size,d_mlp,seq_len
    x = nn.ReLU(x)            # batch_size,d_mlp,seq_len
    x = nn.mul(W_out,x)       # batch_size,d_model,seq_len
    x = nn.add(x,b_out)       # batch_size,d_model,seq_len
    x = nn.transpose(x)       # batch_size,seq_len,d_model = shape(input)

    return x


fn TransformerBlock(inout nn: Module, d_model: Int, num_heads: Int, d_head: Int, n_ctx: Int, d_mlp: Int, batch_size: Int, seq_len: Int, use_attn: Bool, use_mlp: Bool, inout x: Tensor) -> Tensor:
    var res_stream = x
    if(use_attn):
        x = Attention(
            nn=nn,
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            n_ctx=seq_len,
            batch_size=batch_size,
            seq_len=seq_len,
            x=x
        )
    if(use_mlp):
        x = MLP(
            nn=nn,
            d_model=d_model,
            d_mlp=d_mlp,
            batch_size=batch_size,
            seq_len=seq_len,
            x=x
        )
    return nn.add(res_stream,x)