from dv import *
from math import sqrt

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
    # x_t
    var x_t_unreshaped = nn.transpose(x)
    var x_t = nn.reshape(x_t_unreshaped,shape(batch_size,num_heads,d_model,seq_len))

    # W_K
    var W_K_raw = Tensor(shape(num_heads, d_head, d_model))
    W_K_raw.initRandomHe()
    var W_K = nn.reshape(W_K_raw, shape(batch_size, num_heads, d_head, d_model))

    # W_Q
    var W_Q_raw = Tensor(shape(num_heads, d_head, d_model))
    W_Q_raw.initRandomHe()
    var W_Q = nn.reshape(W_Q_raw, shape(batch_size, num_heads, d_head, d_model))    

    # W_V
    var W_V_raw = Tensor(shape(num_heads, d_head, d_model))
    W_V_raw.initRandomHe()
    var W_V = nn.reshape(W_V_raw, shape(batch_size, num_heads, d_head, d_model))   

    # W_O
    var W_O_raw = Tensor(shape(num_heads*d_head, d_model))
    W_O_raw.initRandomHe()
    var W_O = nn.reshape(W_O_raw, shape(batch_size, num_heads*d_head, d_model))   

    # attention heads
    var k = nn.mul(W_K,x_t)     # batch_size,num_heads,d_head,seq_len
    var q = nn.mul(W_Q,x_t)     # batch_size,num_heads,d_head,seq_len
    var v = nn.mul(W_V,x_t)     # batch_size,num_heads,d_head,seq_len
    var k_t = nn.transpose(k)   # batch_size,num_heads,seq_len,d_head

    # architecture
    x = nn.mul(k_t,q)           # batch_size,num_heads,seq_len,seq_len
    x = mask(nn, x, 0.0, -1000000.0, 1) 
    x = scale(nn, x, Float32(1.0)/d_head)
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

    var W_in_raw = Tensor(shape(d_mlp,d_model))
    W_in_raw.initRandomHe()
    var W_in = nn.reshape(W_in_raw, shape(batch_size,d_mlp,d_model))

    var b_in_unreshaped = Tensor(shape(d_mlp))
    b_in_unreshaped.initRandom(-0.01,0.01)
    var b_in_untransposed = nn.reshape(b_in_unreshaped, shape(batch_size,seq_len,d_mlp))
    var b_in = nn.transpose(b_in_untransposed) # batch_size,d_mlp,seq_len

    var W_out_raw = Tensor(shape(d_model,d_mlp))
    W_out_raw.initRandomHe()
    var W_out = nn.reshape(W_out_raw, shape(batch_size,d_model,d_mlp))

    var b_out_unreshaped = Tensor(shape(d_model))
    b_out_unreshaped.initRandom(-0.01,0.01)
    var b_out_untransposed = nn.reshape(b_out_unreshaped, shape(batch_size,seq_len,d_model))
    var b_out = nn.transpose(b_out_untransposed) # batch_size,d_model,seq_len

    # architecture
    x = nn.mul(W_in, x_t)     # batch_size,d_mlp,seq_len
    x = nn.add(x,b_in)        # batch_size,d_mlp,seq_len
    x = nn.ReLU(x)            # batch_size,d_mlp,seq_len
    x = nn.mul(W_out,x)       # batch_size,d_model,seq_len
    x = nn.add(x,b_out)       # batch_size,d_model,seq_len
    x = nn.transpose(x)       # batch_size,seq_len,d_model = shape(input)

    return x


fn TransformerBlock(inout nn: Module, d_model: Int, num_heads: Int, d_head: Int, n_ctx: Int, d_mlp: Int, batch_size: Int, seq_len: Int, inout x: Tensor) -> Tensor:
    x = MLP(
        nn=nn,
        d_model=d_model,
        d_mlp=d_mlp,
        batch_size=batch_size,
        seq_len=seq_len,
        x=x
    )
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
    return x

# simple algorithmic dataset reversing the sequence
fn dataGenerator(inout inputs: Tensor, inout logits: Tensor):
    pass


fn main():

    # init
    var nn = Module()

    # transformer config
    let d_model=128
    let d_vocab=10
    let num_heads=4
    let d_head=16
    let n_ctx=6
    let d_mlp=32
    let batch_size=8
    let seq_len=n_ctx
    let num_layers=1

    # training config
    let num_epochs = 100
    let every = 10
    let lr = 0.001
    let momentum = 0.9

    # init dataset 
    var x = Tensor(shape(batch_size,seq_len,d_vocab))
    var trueVals = Tensor(shape(batch_size,seq_len,d_vocab))
    x.initRandom(0,1)
    trueVals.initRandom(0,1)

    # architecture of a 1 layer Transformer
    x = Embed(nn,d_vocab,d_model,batch_size,x)
    for layer in range(num_layers):
        x = TransformerBlock(
            nn=nn,
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            n_ctx=n_ctx,
            d_mlp=d_mlp,
            batch_size=batch_size,
            seq_len=seq_len,
            x=x
        )  
    x = Unembed(nn,d_vocab,d_model,batch_size,x)  
    x = nn.CE(trueVals,x)

    # training loop
    for epoch in range(1,num_epochs+1):

        # forward pass through the network
        nn.forward(x)
        if(epoch % every == 0):
            print("Epoch:",epoch)
        
        # compute gradients
        nn.backward(x)

        # take a optimization step
        nn.optimize('sgd_momentum', lr = lr, momentum = momentum)
    
        # nn.printTensors()
