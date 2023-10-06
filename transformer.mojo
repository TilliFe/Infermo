from dv import *
from math import sqrt
from random import random_float64, seed
from memory import memset_zero

@always_inline
fn Embed(inout nn: Module, d_vocab: Int, d_model: Int, batch_size: Int, inout x: Tensor) -> Tensor:
    var W_E_raw = Tensor(shape(d_vocab,d_model))
    W_E_raw.initRandomHe()
    var W_E = nn.reshape(W_E_raw, shape(batch_size,d_vocab,d_model))
    return nn.mul(x,W_E)

@always_inline
fn Unembed(inout nn: Module, d_vocab: Int, d_model: Int, batch_size: Int, inout x: Tensor) -> Tensor:
    var W_U_raw = Tensor(shape(d_model,d_vocab))
    W_U_raw.initRandomHe()
    var W_U = nn.reshape(W_U_raw,shape(batch_size,d_model,d_vocab))
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

# simple algorithmic dataset reversing the sequence
fn dataGenerator(inout inputs: Tensor, inout trueVals: Tensor, batch_size: Int, seq_len: Int, d_vocab: Int, ):

    # random sequence of integers in the range [0,d_vocab-1] - yet not oneHot encoded
    let input_raw = Pointer[Int].alloc(batch_size * seq_len * d_vocab)
    let trueVals_raw = Pointer[Int].alloc(batch_size * seq_len * d_vocab)
    for batch in range(batch_size):
        for seq in range(seq_len):
            seed()
            input_raw.store(batch * seq_len + seq, random_float64(0,Float64(d_vocab)).to_int())
            trueVals_raw.store(batch * seq_len + seq, (input_raw.load(batch * seq_len + seq) + 1 ) % d_vocab)

    # init to zero
    let size = batch_size * seq_len * d_vocab
    let input_data = DTypePointer[DType.float32].alloc(size)
    let trueVals_data = DTypePointer[DType.float32].alloc(size)
    memset_zero(input_data,size)
    memset_zero(trueVals_data,size)

    # create a oneHot encoding of indeces in the input_data - current task: bitshift to the right by one
    for batch in range(batch_size):
        for seq in range(seq_len):
            let input_index = input_raw.load(batch * seq_len + seq)
            input_data.store(batch * seq_len * d_vocab + seq * d_vocab + input_index, Float32(1.0))

            let trueVals_index = trueVals_raw.load(batch * seq_len + seq)
            trueVals_data.store(batch * seq_len * d_vocab + seq * d_vocab + trueVals_index, Float32(1.0))

    # fill input and trueVals Tensors with new data
    inputs.setData(input_data)
    trueVals.setData(trueVals_data)

fn main():

    # init
    var nn = Module()

    # transformer config
    let d_model=32
    let d_vocab=d_model
    let num_heads=4
    let d_head=d_model//num_heads
    let n_ctx=8
    let d_mlp=128
    let batch_size=16
    let seq_len=n_ctx
    let num_layers=1
    let use_attn=True
    let use_mlp=True

    # training config
    let num_epochs = 10000
    let every = 100
    let lr = 0.0001
    let momentum = 0.9

    # init dataset 
    var inputs = Tensor(shape(batch_size,seq_len,d_vocab))
    var trueVals = Tensor(shape(batch_size,seq_len,d_vocab))
    
    # fill inputs and trueVals with a batch of data
    dataGenerator(inputs,trueVals,batch_size,seq_len,d_vocab)

    # architecture of a 1 layer Transformer
    # var x = Embed(nn,d_vocab,d_model,batch_size,inputs)
    # for layer in range(num_layers):
    var x = TransformerBlock(
        nn=nn,
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        n_ctx=n_ctx,
        d_mlp=d_mlp,
        batch_size=batch_size,
        seq_len=seq_len,
        use_attn=use_attn,
        use_mlp=use_mlp,
        x=inputs
    )  
    # x = Unembed(nn,d_vocab,d_model,batch_size,x)  
    x = nn.softmax(x)
    var loss = nn.CE(trueVals,x)

    # training loop
    for epoch in range(1,num_epochs+1):

        # # fill inputs and trueVals with a batch of data - does not work yet
        # dataGenerator(inputs,trueVals,batch_size,seq_len,d_vocab)

        # forward pass through the network
        nn.forward(loss)
        if(epoch % every == 0):
            print("Epoch:",epoch, " Loss:", loss.getData(0))
        
        # compute gradients
        nn.backward(loss)

        # take a optimization step
        nn.optimize('sgd_momentum', lr = lr, momentum = momentum)
    
    # nn.printTensors()