from dv import *
from random import random_float64, seed
from memory import memset_zero
from math import sqrt

fn dataGenerator(inout inputs: Tensor, inout trueVals: Tensor, batch_size: Int, seq_len: Int, d_vocab: Int, ):

    # ############### TOY DATASET: Add 1 to each number in a sequence. ###############

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

    ##################### TRANSFORMER model (trained on toy dataset) #################

    # transformer config -> achieves high accuracy but very low confidence (high loss) -> fix?
    let use_attn=True
    let use_mlp=True
    let num_layers=2
    let num_heads=4
    let d_model=64
    let d_mlp=64
    let d_head=d_model//num_heads
    let d_vocab=32
    let n_ctx=16
    let seq_len=n_ctx
    let init_std = Float32(0.1)/sqrt(d_model)   # re: weight initalization - currently works better with smaller standard deviation, used for all weights
    let batch_size=32

    # training config
    let num_epochs = 500
    let every = 25
    let lr = 0.0003
    let momentum = 0.8
    let wd = 0.01               # weight decay for better generalization
    let th = 1.0                # gradient clipping threshold

    # init
    var nn = Module()
    var inputs = Tensor(shape(batch_size,seq_len,d_vocab))
    inputs.requiresGradient = False
    var trueVals = Tensor(shape(batch_size,seq_len,d_vocab))
    trueVals.requiresGradient = False
    
    # architecture of a n layer Transformer
    var x = Embed(nn,d_vocab,d_model,batch_size,init_std,inputs)
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
            use_attn=use_attn,
            use_mlp=use_mlp,
            init_std=init_std,
            x=x
        )  
    x = Unembed(nn,d_vocab,d_model,batch_size,init_std,x)  
    var logits = nn.softmax(x)
    var loss = nn.CE(trueVals,logits)

    # training loop
    var avgAcc: Float32 = 0.0
    var avgLoss: Float32 = 0.0

    for epoch in range(1,num_epochs+1):

        # feed the network with fresh innput data and the respective labels
        dataGenerator(inputs,trueVals,batch_size,seq_len,d_vocab)

        # forward pass through the network
        nn.forward(loss)

        # print out stuff, not important to the learning procedure
        let res = max(logits)
        avgAcc += accuracy(res,trueVals)
        avgLoss += loss.getData(0)
        if(epoch % every == 0):
            print("Epoch:",epoch, ", Loss:", avgLoss/every, ", avgAcc:", avgAcc/every)
            avgAcc = 0.0
            avgLoss = 0.0
            # inputs.printData()
            # trueVals.printData()
            # res.printData()
        
        # compute gradients
        nn.backward(loss)

        # take a optimization step
        nn.optimize('sgd_momentum', lr = lr, momentum = momentum, weight_decay=wd, threshold=th)