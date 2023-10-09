from dv import Tensor,Module,shape,transformer_block,embed,unembed,max,accuracy
from random import random_float64, seed
from memory import memset_zero
from math import sqrt

##################### Transformer Neural Network (trained on an algorithmic (toy) dataset) ########################################

# TOY DaTaSET: add 1 to each number in a sequence.
fn data_generator(inout inputs: Tensor, inout true_vals: Tensor, batch_size: Int, seq_len: Int, d_vocab: Int):

    # random sequence of integers in the range [0,d_vocab-1] - yet not one_hot encoded
    let input_raw = Pointer[Int].alloc(batch_size * seq_len * d_vocab)
    let true_vals_raw = Pointer[Int].alloc(batch_size * seq_len * d_vocab)
    for batch in range(batch_size):
        for seq in range(seq_len):
            seed()
            input_raw.store(batch * seq_len + seq, random_float64(0,Float64(d_vocab)).to_int())
            true_vals_raw.store(batch * seq_len + seq, (input_raw.load(batch * seq_len + seq) + 1 ) % d_vocab)

    # init to zero
    let size = batch_size * seq_len * d_vocab
    let input_data = DTypePointer[DType.float32].alloc(size)
    let true_vals_data = DTypePointer[DType.float32].alloc(size)
    memset_zero(input_data,size)
    memset_zero(true_vals_data,size)

    # create a one_hot encoding of indeces in the input_data - current task: bitshift to the right by one
    for batch in range(batch_size):
        for seq in range(seq_len):
            let input_index = input_raw.load(batch * seq_len + seq)
            input_data.store(batch * seq_len * d_vocab + seq * d_vocab + input_index, Float32(1.0))

            let true_vals_index = true_vals_raw.load(batch * seq_len + seq)
            true_vals_data.store(batch * seq_len * d_vocab + seq * d_vocab + true_vals_index, Float32(1.0))

    # fill input and true_vals tensors with new data
    inputs.set_data(input_data)
    true_vals.set_data(true_vals_data)

fn main():

    # transformer config -> achieves high accuracy but very low confidence (high loss) -> fix?
    let use_attn=True
    let use_mlp=True
    let num_layers=2
    let num_heads=4
    let d_Model=64
    let d_mlp=64
    let d_head=d_Model//num_heads
    let d_vocab=32
    let n_ctx=16
    let seq_len=n_ctx
    let init_std = Float32(0.1)/sqrt(d_Model)   # re: weight initalization - currently works better with smaller standard deviation, used for all weights
    let batch_size=32

    # training config
    let num_epochs = 500
    let every = 25
    let lr = 0.0003
    let momentum = 0.8
    let wd = 0.01               # weight decay for better generalization
    let th = 1.0                # grad clipping threshold

    # init
    var nn = Module()
    var inputs = Tensor(shape(batch_size,seq_len,d_vocab))
    inputs.requires_grad = False
    var true_vals = Tensor(shape(batch_size,seq_len,d_vocab))
    true_vals.requires_grad = False
    
    # architecture of a n layer Transformer
    var x = embed(nn,d_vocab,d_Model,batch_size,init_std,inputs)
    for layer in range(num_layers):
        x = transformer_block(
            nn=nn,
            d_Model=d_Model,
            num_heads=num_heads,
            d_head=d_head,
            n_ctx=n_ctx,
            d_mlp=d_mlp,
            batch_size=batch_size,
            seq_len=seq_len,
            use_attn=use_attn,
            use_mlp=use_mlp,
            init_std=init_std,
            x=x
        )  
    x = unembed(nn,d_vocab,d_Model,batch_size,init_std,x)  
    var logits = nn.softmax(x)
    var loss = nn.cE(true_vals,logits)

    # training loop
    var avg_acc: Float32 = 0.0
    var avgLoss: Float32 = 0.0

    for epoch in range(1,num_epochs+1):

        # feed the network with fresh innput data and the respective labels
        data_generator(inputs,true_vals,batch_size,seq_len,d_vocab)

        # forward pass through the network
        nn.forward(loss)

        # print out stuff, not important to the learning procedure
        let res = max(logits)
        avg_acc += accuracy(res,true_vals)
        avgLoss += loss.data.load(0)
        if(epoch % every == 0):
            print("Epoch:",epoch, ", Loss:", avgLoss/every, ", avg_acc:", avg_acc/every)
            avg_acc = 0.0
            avgLoss = 0.0
            # inputs.print_data()
            # true_vals.print_data()
            # res.print_data()
        
        # compute grads
        nn.backward(loss)

        # take a optimization step
        nn.optimize('sgd_momentum', lr = lr, momentum = momentum, weight_decay=wd, threshold=th)