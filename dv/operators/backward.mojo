from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp, min, max

from ..graph.tensor import Tensor

alias nelts = simdwidthof[DType.float32]()

@always_inline
fn mul_grad(c: Tensor, inout a: Tensor, inout b: Tensor):

    let a_matrix_size = a.shape[a.num_dims-2] * a.shape[a.num_dims-1]
    let b_matrix_size = b.shape[b.num_dims-2] * b.shape[b.num_dims-1]
    let c_matrix_size = c.shape[c.num_dims-2] * c.shape[c.num_dims-1] 

    let M = a.shape[a.num_dims-2]
    let K = b.shape[b.num_dims-2]
    let N = b.shape[b.num_dims-1]

    var offset_a: Int = 0
    var offset_b: Int = 0
    var offset_c: Int = 0

    for s in range(c.cap // c_matrix_size):

        offset_c = s * c_matrix_size

        # consider broadcasting
        if(a.num_dims == b.num_dims):
            offset_a = s * a_matrix_size
            offset_b = s * b_matrix_size
        elif(a.num_dims > b.num_dims):
            offset_a = s * a_matrix_size
        else:
            offset_b = s * b_matrix_size

        if (a.requires_grad):
            @parameter
            fn calc_row_1(m: Int):
                for n in range(N):
                    @parameter
                    fn dot[nelts: Int](k: Int):
                        let index_a = offset_a + m * K + k
                        let index_c = offset_c + m * N + n
                        let index_b = offset_b + k * N + n
                        let val = a.grad.load(index_a) + c.grad.load(index_c) * b.data.load(index_b) 
                        a.grad.store(index_a, val)
                    vectorize[nelts, dot](K)
            parallelize[calc_row_1](M,M)

        if(b.requires_grad):
            @parameter
            fn calc_row_2(k: Int):
                for m in range(M):
                    @parameter
                    fn dot[nelts: Int](n: Int):
                        let index_b = offset_b + k * N + n
                        let index_a = offset_a + m * K + k
                        let index_c = offset_c + m * N + n
                        let val = b.grad.load(index_b) + a.data.load(index_a) * c.grad.load(index_c)  
                        b.grad.store(index_b, val)
                    vectorize[nelts, dot](N)
            parallelize[calc_row_2](K,K)

@always_inline        
fn add_grad(c: Tensor, inout a: Tensor, inout b: Tensor):

    # regular
    if(a.num_dims == b.num_dims):
        if(a.requires_grad):
            @parameter
            fn v_add_gr_1[nelts: Int](i: Int):
                a.grad.simd_store[nelts](
                    i, a.grad.simd_load[nelts](i) + c.grad.simd_load[nelts](i)
                )
            vectorize[nelts, v_add_gr_1](a.cap)
        if(b.requires_grad):
            @parameter
            fn v_add_gr_2[nelts: Int](i: Int):
                b.grad.simd_store[nelts](
                    i, b.grad.simd_load[nelts](i) + c.grad.simd_load[nelts](i)
                )
            vectorize[nelts, v_add_gr_2](b.cap)

    # consider broadcasting
    else:
        var offset_a: Int = 0
        var offset_b: Int = 0
        var offset_c: Int = 0
        var ratio: Int = 0
        var H = 0

        if(a.num_dims > b.num_dims):
            H = b.cap
            ratio = a.cap // b.cap
        else:
            H = a.cap
            ratio = b.cap // a.cap

        for s in range(ratio):
            if(a.num_dims > b.num_dims):
                offset_a = s * H
            else:
                offset_b = s * H

            offset_c = s * H
            if(a.requires_grad):
                @parameter
                fn v_add_a[nelts: Int](i: Int):
                    a.grad.simd_store[nelts](
                        offset_a + i, a.grad.simd_load[nelts](offset_a + i) + c.grad.simd_load[nelts](offset_c + i)
                    )
                vectorize[nelts, v_add_a](H) 

            if(b.requires_grad):
                @parameter
                fn v_add_b[nelts: Int](i: Int):
                    b.grad.simd_store[nelts](
                        offset_b + i, b.grad.simd_load[nelts](offset_b + i) + c.grad.simd_load[nelts](offset_c + i)
                    )
                vectorize[nelts, v_add_b](H) 

    # # Loop over each image in the batch
    # for i in range(a.shape[0]):
    #     # Loop over each filter
    #     for j in range(c.shape[0]):
    #         # Loop over each channel
    #         for x in range(b.shape[2]):
    #             for y in range(b.shape[3]):
    #                 var patch_sum: Float32 = 0.0

    #                 # apply the convolution operation - vectorize?
    #                 for k in range(a.shape[1]):
    #                 # convolve the k-th channel of the i-th image with the k-th channel of the j-th filter
    #                     for dx in range(c.shape[2]):
    #                         for dy in range(c.shape[3]):                        
    #                             # calculate input indices with consideration for padding and stride
    #                             let ix = x * stride - padding + dx
    #                             let iy = y * stride - padding + dy
    #                             # Skip if index is out of bounds (this is 'zero' padding)
    #                             if ix < 0 or iy < 0 or ix >= a.shape[2] or iy >= a.shape[3]:
    #                                 continue
    #                             let a_index = index(j, k, ix, iy, a.shape[1], a.shape[2], a.shape[3])
    #                             let c_grad_index = index(i, k, dx, dy, a.shape[1], c.shape[2], c.shape[3])
    #                             # add to patch sum
    #                             patch_sum += a.data.load(a_index) * c.grad.load(c_grad_index)

    #                 # Store patch sum into c after innermost loops
    #                 let b_grad_index = index(i, j, x, y, c.shape[0], b.shape[2], b.shape[3])
    #                 b.grad.store(b_grad_index, b.grad.load(b_grad_index) + patch_sum)


@always_inline
fn conv_2d_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
  
    let padding = c.other_params.load(0)
    let stride = c.other_params.load(1)

    # Function to calculate the index in the 1D buffer
    fn index(n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int) -> Int:
        return n*(num_channels*height*width) + c*(height*width) + h*width + w

    # ##### compute the gradietn of the Kernel (right tensor) ########################################
    for i in range(a.shape[1]): # in_channels
        for j in range(b.shape[0]): # out_channels
            for x in range(b.shape[2]): # kernel_width
                for y in range(b.shape[3]): # kernel_height
                    var patch_sum: Float32 = 0.0
                    for b in range(a.shape[0]):
                        for dx in range(c.shape[2]):
                            
                            @parameter
                            fn inner_loop[_nelts: Int](dy: Int):                        
                                # calculate input indices with consideration for padding and stride
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                # Skip if index is out of bounds (this is 'zero' padding)
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape[2]
                                    or iy >= a.shape[3]
                                ):
                                    let a_index = index(b, i, ix, iy, a.shape[1], a.shape[2], a.shape[3])
                                    let c_grad_index = index(b, j, dx, dy, c.shape[1], c.shape[2], c.shape[3])
                                    # add to patch sum
                                    patch_sum += (
                                        a.data.simd_load[_nelts](a_index)
                                        * c.grad.simd_load[_nelts](c_grad_index)
                                    ).reduce_add()

                            vectorize[nelts, inner_loop](c.shape[3])
                    let b_grad_index = index(i, j, x, y, b.shape[0], b.shape[2], b.shape[3])
                    b.grad.store(b_grad_index, patch_sum)

    # ##### compute the gradient of the Input (left tensor) ############################################
    @parameter
    fn batch_loop(p: Int):  # batch_size
        for j in range(a.shape[1]): # in_channels
            for i in range(b.shape[0]): # out_channels
                for x in range(a.shape[2]):
                    for y in range(a.shape[3]):
                        var patch_sum : Float32 = 0.0
                        for dx in range(b.shape[2]):

                            @parameter
                            fn dy_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - dx + padding
                                let iy = y * stride - dy + padding
                                # Skip if index is out of bounds (this is 'zero' padding)
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= c.shape[2]
                                    or iy >= c.shape[3]
                                ):
                                    let c_grad_index = index(
                                        p,
                                        i,
                                        ix,
                                        iy,
                                        c.shape[1],
                                        c.shape[2],
                                        c.shape[3],
                                    )
                                    let b_index = index(
                                        i,
                                        j,
                                        b.shape[2] - dx - 1,
                                        b.shape[3] - dy - 1,
                                        b.shape[1],
                                        b.shape[2],
                                        b.shape[3],
                                    )
                                    patch_sum += (
                                        c.grad.simd_load[_nelts](c_grad_index)
                                        * c.data.simd_load[_nelts](b_index)
                                    ).reduce_add()
                            
                            vectorize[nelts, dy_loop](b.shape[3])
                        let a_grad_index = index(p,j,x,y,a.shape[1],a.shape[2],a.shape[3])
                        a.grad.store( a_grad_index, a.grad.load(a_grad_index) + patch_sum)

    parallelize[batch_loop](a.shape[0], a.shape[0])

@always_inline
fn max_pool_2d_grad(b: Tensor, inout a: Tensor):

    let padding = b.other_params.load(0)
    let stride = b.other_params.load(1)
    let kernel_width = b.other_params.load(2)
    let kernel_height = b.other_params.load(3)

    # Function to calculate the index in the 1D buffer
    fn index(n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int) -> Int:
        return n*(num_channels*height*width) + c*(height*width) + h*width + w

    for p in range(a.shape[0]): # batch_size
        for i in range(a.shape[1]): # in_channels
            for x in range(0,a.shape[2]-kernel_width+1 + 2*padding,stride): # width
                for y in range(0,a.shape[3]-kernel_height+1 + 2*padding,stride): # height
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if ix < 0 or iy < 0 or ix >= a.shape[2] or iy >= a.shape[3]:
                                continue
                            let idx = index(p,i,ix,iy,a.shape[1],a.shape[2],a.shape[3])
                            let entry = a.data.load(idx)
                            if(entry > max_val):
                                max_val = entry
                                arg_max = idx
                    let b_grad_idx = index(p,i,(x)//stride,(y)//stride,b.shape[1],b.shape[2],b.shape[3])
                    a.grad.store(arg_max, a.grad.load(arg_max) + b.grad.load(b_grad_idx))

@always_inline
fn relu_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_relu_bw[nelts: Int](i: Int):
        let zeros = SIMD[DType.float32,nelts]()
        a.grad.simd_store[nelts](
            i, (a.data.simd_load[nelts](i) > zeros).cast[DType.float32]() * b.grad.simd_load[nelts](i) + a.grad.simd_load[nelts](i)
        )
    vectorize[nelts, v_relu_bw](a.cap)

@always_inline
fn sum_grad(b: Tensor, inout a: Tensor):
    a.fill_grad(1)

@always_inline
fn softmax_grad(b: Tensor, inout a: Tensor):
    if b.other_params.load(0) == 3001:
        a.set_grad(b.grad)
    else:
        let num_dims = b.num_dims
        let M = b.shape[num_dims - 2]
        let N = b.shape[num_dims - 1]
        for s in range(b.cap // N):
            let offset = s * N

            @parameter
            fn v_softmax_grad_outer[nelts: Int](j: Int):
                var grad: Float32 = 0
                var grad2: Float32 = 0

                @parameter
                fn v_softmax_grad[nelts: Int](i: Int):
                    if i == j:
                        let temp = b.data.simd_load[nelts](offset + j)
                        grad += (
                            b.grad.simd_load[nelts](offset + i)
                            * (temp * (Float32(1.0) - temp))
                        ).reduce_add()
                    else:
                        grad += (
                            b.grad.simd_load[nelts](offset + i)
                            * b.data.simd_load[nelts](offset + i)
                            * b.data.simd_load[nelts](offset + j)
                            * -1
                        ).reduce_add()  # changed to grad +=, because of the *-1

                vectorize[nelts, v_softmax_grad](N)

                a.grad.simd_store[nelts](
                    offset + j, a.grad.simd_load[nelts](offset + j) + grad
                )

            vectorize[nelts, v_softmax_grad_outer](N)

@always_inline
fn mse_grad(c: Tensor, inout a: Tensor, inout b: Tensor):  # a: TrueVals, b: Logits
    let num_dims = a.num_dims
    let M = a.shape[num_dims - 2]
    let N = a.shape[num_dims - 1]

    @parameter
    fn v_mse_grad[nelts: Int](index: Int):
        let grad = Float32(2) * (
            b.data.simd_load[nelts](index) - a.data.simd_load[nelts](index)
        ) / Float32(a.cap)
        if a.requires_grad:
            a.grad.simd_store[nelts](index, a.grad.simd_load[nelts](index) + grad)
        if b.requires_grad:
            b.grad.simd_store[nelts](index, b.grad.simd_load[nelts](index) + grad)

    vectorize[nelts, v_mse_grad](a.cap)

@always_inline
fn ce_grad(c: Tensor, inout a: Tensor, inout b: Tensor): # a: TrueVals, b: Logits
    let num_dims = a.num_dims
    let N = a.shape[num_dims-1]

    if(a.requires_grad):
        if(a.name == "softmax"):
            for index in range(a.cap):
                let grad = (b.data.load(index) - a.data.load(index)) 
                a.grad.store(index,  a.grad.load(index) +  grad /  (Float32(a.cap) / Float32(N)))
            else:
                for index in range(a.cap):
                    let grad_a = - log(b.data.load(index))
                    a.grad.store(index,  a.grad.load(index) +  grad_a / (Float32(a.cap) / Float32(N)))
    if(b.requires_grad):
        if(b.name == "softmax"):
            for index in range(b.cap):
                let grad = (b.data.load(index) - a.data.load(index)) 
                b.grad.store(index,  b.grad.load(index) + grad / (Float32(a.cap) / Float32(N)))
        else:
            for index in range(b.cap):
                let grad_b = - a.data.load(index) / (b.data.load(index))
                b.grad.store(index,  b.grad.load(index) + grad_b / (Float32(a.cap) / Float32(N)))

@always_inline
fn reshape_grad(b: Tensor, inout a: Tensor):
    for s in range(b.cap // a.cap):
        let offset = s * a.cap
        @parameter
        fn v_reshape[nelts: Int](i: Int):
            a.grad.simd_store[nelts](
                i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](offset + i)
            )
        vectorize[nelts, v_reshape](a.cap)


@always_inline
fn transpose_grad(b: Tensor, inout a: Tensor):
    let num_dims = b.num_dims
    let M = b.shape[num_dims - 2]
    let N = b.shape[num_dims - 1]

    for s in range(b.cap // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transpose[nelts: Int](j: Int):
                a.grad.simd_store[nelts](
                    offset + j * M + i,
                    a.grad.simd_load[nelts](offset + j * M + i)
                    + b.grad.simd_load[nelts](offset + i * N + j),
                )

            vectorize[nelts, v_transpose](N)

@always_inline
fn copy_grad(b: Tensor, inout a: Tensor): 
    memcpy(a.grad,b.grad,a.cap)
