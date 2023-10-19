from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import max, min, sqrt, abs, exp2, exp, log2, log, cos, sin, tan, asin, acos, atan, cosh, sinh, tanh
from sys.param_env import env_get_int

from ..graph.tensor import Tensor


alias nelts = simdwidthof[DType.float32]()
alias workers = env_get_int["WORKERS", 0]()


# Non-elementwise operators ####################################################

@always_inline
fn matmul(inout c: Tensor, a: Tensor, b: Tensor):
    var a_matrix_size = a.shape[a.num_dims-2] * a.shape[a.num_dims-1]
    var b_matrix_size = b.shape[b.num_dims-2] * b.shape[b.num_dims-1]
    var c_matrix_size = c.shape[c.num_dims-2] * c.shape[c.num_dims-1]

    let M = c.shape[c.num_dims-2]
    let K = a.shape[a.num_dims-1]
    let N = c.shape[c.num_dims-1] 

    var offset_a: Int = 0
    var offset_b: Int = 0
    var offset_c: Int = 0

    for s in range(c.cap // c_matrix_size):
        
        offset_c = s * c_matrix_size

        # consider broadcasting
        if(a.cap == b.cap):
            offset_a = s * a_matrix_size
            offset_b = s * b_matrix_size
        elif(a.cap > b.cap):
            offset_a = s * a_matrix_size
        else:
            offset_b = s * b_matrix_size

        @parameter
        fn calc_row(m: Int):
            for k in range(K):
                @parameter
                fn dot[nelts: Int](n: Int):
                    c.data.simd_store[nelts](offset_c + m*N+n, c.data.simd_load[nelts](offset_c + m*N+n) + a.data.load(offset_a + m*K+k) * b.data.simd_load[nelts](offset_b + k*N+n))
                vectorize[nelts, dot](N)
        parallelize[calc_row](M, workers if workers > 0 else M) 


@always_inline
fn conv_2d(inout c: Tensor, a: Tensor, b: Tensor):
    
    let padding = c.other_params.load(0)
    let stride = c.other_params.load(1)

    # Function to calculate the index in the 1D buffer
    fn index(n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int) -> Int:
        return n*(num_channels*height*width) + c*(height*width) + h*width + w

    # Loop over each image in the batch
    @parameter
    fn batch_loop(i: Int):
        for j in range(b.shape[0]):
            for x in range(c.shape[2]):
                for y in range(c.shape[3]):
                    var patch_sum: Float32 = 0.0
                    # apply the convolution operation - vectorize?
                    for k in range(a.shape[1]):
                        for dx in range(b.shape[2]):
                            
                            @parameter
                            fn inner_loop[_nelts: Int](dy: Int):                        
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape[2]
                                    or iy >= a.shape[3]
                                ):
                                    let a_index = index(i, k, ix, iy, a.shape[1], a.shape[2], a.shape[3])
                                    let b_index = index(j, k, dx, dy, a.shape[1], b.shape[2], b.shape[3])
                                    patch_sum += (
                                        a.data.simd_load[_nelts](a_index)
                                        * b.data.simd_load[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, inner_loop](b.shape[3])
                    let c_index = index(i, j, x, y, b.shape[0], c.shape[2], c.shape[3])
                    c.data.store(c_index, patch_sum)

    parallelize[batch_loop](a.shape[0], workers if workers > 0 else a.shape[0])


@always_inline
fn max_pool_2d(inout b: Tensor, a: Tensor):
    
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
                    # vectorize ?
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
                    let idx = index(p,i,(x)//stride,(y)//stride,b.shape[1],b.shape[2],b.shape[3])
                    b.data.store(idx,max_val)              


@always_inline
fn sum(inout b: Tensor, a: Tensor):
    var sum: Float32 = 0

    @parameter
    fn v_sum[nelts: Int](i: Int):
        sum += a.data.simd_load[nelts](i).reduce_add()

    b.set_data(0, sum)


@always_inline
fn softmax(inout b: Tensor, a: Tensor):
    # by default take the softmax along the last dimension of the tensor
    let num_dims = a.num_dims
    let N = a.shape[num_dims - 1]

    for s in range(b.cap // N):
        var max_el: Float32 = 0.0

        @parameter
        fn v_max[nelts: Int](i: Int):
            let temp = b.data.simd_load[nelts](s * N + i).reduce_max()
            max_el = max(max_el, temp)

        vectorize[nelts, v_max](N)

        # calculate the exponential of each element and do sum op
        var sum: Float32 = 0.0

        @parameter
        fn v_exp[nelts: Int](i: Int):
            let temp = exp(a.data.simd_load[nelts](s * N + i) - max_el)
            b.data.simd_store[nelts](s * N + i, temp)
            sum += temp.reduce_add()

        vectorize[nelts, v_exp](N)

        # divide each element by the sum
        @parameter
        fn v_div[nelts: Int](i: Int):
            b.data.simd_store[nelts](
                s * N + i, b.data.simd_load[nelts](s * N + i) / sum
            )

        vectorize[nelts, v_div](N)


@always_inline
fn mse(inout c: Tensor, a: Tensor, b: Tensor):
    @parameter
    fn v_mse[nelts: Int](index: Int):
        let error = (
            a.data.simd_load[nelts](index) - b.data.simd_load[nelts](index)
        ) * (a.data.simd_load[nelts](index) - b.data.simd_load[nelts](index))
        c.set_data(0, c.data.load(0) + error.reduce_add())

    vectorize[nelts, v_mse](a.cap)
    c.set_data(0, c.data.load(0) / Float32(a.cap))


@always_inline
fn ce(inout c: Tensor, a: Tensor, b: Tensor):
    let num_dims = a.num_dims
    let N = a.shape[num_dims - 1]
    let epsilon = Float32(1e-8)

    @parameter
    fn v_ce[nelts: Int](index: Int):
        let error = -a.data.simd_load[nelts](index) * log(
            b.data.simd_load[nelts](index) + epsilon
        )
        c.set_data(0, c.data.load(0) + error.reduce_add())

    vectorize[nelts, v_ce](a.cap)
    c.set_data(0, c.data.load(0) / (Float32(a.cap) / Float32(N)))


@always_inline
fn reshape(inout b: Tensor, a: Tensor):
    for s in range(b.cap // a.cap):
        let offset = s * a.cap
        @parameter
        fn v_reshape[nelts: Int](i: Int):
            b.data.simd_store[nelts](
                offset + i, a.data.simd_load[nelts](i)
            )
        vectorize[nelts, v_reshape](a.cap)


@always_inline
fn transpose(inout b: Tensor, a: Tensor):
    # we always tranpose along the last two dimensions of the tensor - vectorize?
    let num_dims = a.num_dims
    let M = a.shape[num_dims - 2]
    let N = a.shape[num_dims - 1]

    for s in range(b.cap // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transpose[nelts: Int](j: Int):
                b.data.simd_store[nelts](
                    offset + j * M + i, a.data.simd_load[nelts](offset + i * N + j)
                )

            vectorize[nelts, v_transpose](N)

@always_inline
fn mean(inout b: Tensor, a: Tensor): 
    let dim_len: Int = b.other_params.load(0)
    
    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]

    var in_dims = DynamicVector[Bool](b.num_dims)
    for d in range(b.num_dims):
        in_dims[d] = False
    for d in range(dim_len):
        in_dims[b.other_params.load(d+1)] = True

    # Iterate over all elements in the tensor
    for i in range(a.cap):
        var indeces = DynamicVector[Int]()
        for dim in range(a.num_dims):
            indeces.push_back((i // a.strides[dim]) % a.shape[dim])
        var output_index = 0
        for dim in range(b.num_dims):
            if not in_dims[dim]:
                output_index += indeces[dim] * b.strides[dim]
            
        b.data.store(output_index, b.data.load(output_index) + a.data.load(i))

    # Divide each element in output tensor by total number of elements in dims
    for i in range(b.cap):
        let value: Float32 = b.data.load(i) / Float32(total_elements_in_dims)
        b.data.store(i, value)


@always_inline
fn variance(inout b: Tensor, a: Tensor): 

    let dim_len: Int = b.other_params.load(0)
    let mean_output = DTypePointer[DType.float32].alloc(b.cap)
    memset_zero(mean_output, b.cap)
    
    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]

    var in_dims = DynamicVector[Bool](b.num_dims)
    for d in range(b.num_dims):
        in_dims[d] = False
    for d in range(dim_len):
        in_dims[b.other_params.load(d+1)] = True

    # Iterate over all elements in the tensor
    for i in range(a.cap):
        var indeces = DynamicVector[Int]()
        for dim in range(a.num_dims):
            indeces.push_back((i // a.strides[dim]) % a.shape[dim])

        var output_index = 0
        for dim in range(b.num_dims):
            if not in_dims[dim]:
                output_index += indeces[dim] * b.strides[dim]
        
        mean_output.store(output_index, mean_output.load(output_index) + a.data.load(i))

    # Divide each element in output tensor by total number of elements in dims
    for i in range(b.cap):
        let value: Float32 = mean_output.load(i) / Float32(total_elements_in_dims)
        mean_output.store(i, value)
    
    # Iterate over all elements in the tensor again to calculate squared differences from the mean
    for i in range(a.cap):
        var indeces = DynamicVector[Int]()
        for dim in range(a.num_dims):
            indeces.push_back((i // a.strides[dim]) % a.shape[dim])

        var output_index = 0
        for dim in range(b.num_dims):
            if not in_dims[dim]:
                output_index += indeces[dim] * b.strides[dim]
        
        let diff = a.data.load(i) - mean_output.load(output_index)
        b.data.store(output_index, b.data.load(output_index) + diff * diff)

    # Divide each element in squared_diff_output tensor by total number of elements in dims to get the variance
    for i in range(b.cap):
        let value: Float32 = b.data.load(i) / Float32(total_elements_in_dims - 1)
        b.data.store(i, value) 



@always_inline
fn std(inout b: Tensor, a: Tensor): 

    let dim_len: Int = b.other_params.load(0)
    let mean_output = DTypePointer[DType.float32].alloc(b.cap)
    memset_zero(mean_output, b.cap)
    
    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]

    var in_dims = DynamicVector[Bool](b.num_dims)
    for d in range(b.num_dims):
        in_dims[d] = False
    for d in range(dim_len):
        in_dims[b.other_params.load(d+1)] = True

    # Iterate over all elements in the tensor
    for i in range(a.cap):
        var indeces = DynamicVector[Int]()
        for dim in range(a.num_dims):
            indeces.push_back((i // a.strides[dim]) % a.shape[dim])

        var output_index = 0
        for dim in range(b.num_dims):
            if not in_dims[dim]:
                output_index += indeces[dim] * b.strides[dim]
        
        mean_output.store(output_index, mean_output.load(output_index) + a.data.load(i))

    # Divide each element in output tensor by total number of elements in dims
    for i in range(b.cap):
        let value: Float32 = mean_output.load(i) / Float32(total_elements_in_dims)
        mean_output.store(i, value)
    
    # Iterate over all elements in the tensor again to calculate squared differences from the mean
    for i in range(a.cap):
        var indeces = DynamicVector[Int]()
        for dim in range(a.num_dims):
            indeces.push_back((i // a.strides[dim]) % a.shape[dim])

        var output_index = 0
        for dim in range(b.num_dims):
            if not in_dims[dim]:
                output_index += indeces[dim] * b.strides[dim]
        
        let diff = a.data.load(i) - mean_output.load(output_index)
        b.data.store(output_index, b.data.load(output_index) + diff * diff)

    # Divide each element in squared_diff_output tensor by total number of elements in dims to get the variance
    for i in range(b.cap):
        let value: Float32 = sqrt(b.data.load(i) / Float32(total_elements_in_dims - 1))
        b.data.store(i, value) 


# elementwise operators ####################################################

# e_add - recursive call for proper broadcasting
fn recursive_mul(inout c: Tensor, borrowed a: Tensor, borrowed b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int, borrowed a_shape: DynamicVector[Int], borrowed b_shape: DynamicVector[Int], borrowed a_strides: DynamicVector[Int], borrowed b_strides: DynamicVector[Int]):

    if(depth == len(a_shape)):
        c.data.store(c_index, a.data.load(a_index) * b.data.load(b_index))
        return

    # if(a_strides[depth]*a_shape[depth] == b_strides[depth]*b_shape[depth] or depth == len(a_shape)):
    #     if(depth == len(a_shape)):
    #         c.data.store(c_index, a.data.load(a_index) * b.data.load(b_index))
    #         return
    #     else:
    #         @parameter
    #         fn v_mul[nelts: Int](i: Int):
    #             c.data.simd_store[nelts](
    #                 c_index*c.strides[depth]*c.shape[depth] + i, a.data.load(a_index*a_strides[depth]*a_shape[depth] + i) * b.data.load(b_index*b_strides[depth]*b_shape[depth] + i)
    #             )
    #         vectorize[nelts, v_mul](c.strides[depth]*c.shape[depth])
    #         return

    if(a_shape[depth] != 1 and b_shape[depth] == 1):
        for s in range(a_shape[depth]):
            recursive_mul(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index,a_shape[depth]*a_index + s, depth+1, a_shape, b_shape, a_strides, b_strides)
    elif(a_shape[depth] == 1 and b_shape[depth] != 1):
        for s in range(b_shape[depth]):
            recursive_mul(c,a,b,a_shape[depth]*a_index, b_shape[depth]*b_index + s, b_shape[depth]*b_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)
    else:
        for s in range(a_shape[depth]):
            recursive_mul(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index + s, a_shape[depth]*a_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)

@always_inline
fn e_mul(inout c: Tensor, a: Tensor, b: Tensor):

    var a_shape = DynamicVector[Int](0)
    var b_shape = DynamicVector[Int](0)
    var a_strides = DynamicVector[Int](0)
    var b_strides = DynamicVector[Int](0)
    if(a.num_dims > b.num_dims): 
        for i in range(a.num_dims - b.num_dims):
            b_shape.push_back(1)
    elif(a.num_dims < b.num_dims): 
        for i in range(b.num_dims - a.num_dims):
            a_shape.push_back(1)
    for i in range(a.num_dims):
        a_shape.push_back(a.shape.load(i))
        b_shape.push_back(b.shape.load(i))
    for i in range(len(a_shape)):
        a_strides.push_back(1)
        b_strides.push_back(1)
    for i in range(len(a_shape)-2,-1,-1):
        a_strides[i] = a_strides[i+1]*a_shape[i+1]
        b_strides[i] = b_strides[i+1]*b_shape[i+1]

    recursive_mul(c,a,b,0,0,0,0,a_shape,b_shape,a_strides,b_strides)

# e_add - recursive call for proper broadcasting
fn recursive_add(inout c: Tensor, borrowed a: Tensor, borrowed b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int, borrowed a_shape: DynamicVector[Int], borrowed b_shape: DynamicVector[Int], borrowed a_strides: DynamicVector[Int], borrowed b_strides: DynamicVector[Int]):
    if(depth == len(a_shape)):
        c.data.store(c_index, a.data.load(a_index) + b.data.load(b_index))
        return

    # if(a_strides[depth]*a_shape[depth] == b_strides[depth]*b_shape[depth] or depth == len(a_shape)):
    #     if(depth == len(a_shape)):
    #         c.data.store(c_index, a.data.load(a_index) + b.data.load(b_index))
    #         return
    #     else:
    #         @parameter
    #         fn v_add[nelts: Int](i: Int):
    #             c.data.simd_store[nelts](
    #                 c_index*c.strides[depth]*c.shape[depth] + i, a.data.load(a_index*a_strides[depth]*a_shape[depth] + i) + b.data.load(b_index*b_strides[depth]*b_shape[depth] + i)
    #             )
    #         vectorize[nelts, v_add](c.strides[depth]*c.shape[depth])
    #         return

    if(a_shape[depth] != 1 and b_shape[depth] == 1):
        for s in range(a_shape[depth]):
            recursive_add(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index,a_shape[depth]*a_index + s, depth+1, a_shape, b_shape, a_strides, b_strides)
    elif(a_shape[depth] == 1 and b_shape[depth] != 1):
        for s in range(b_shape[depth]):
            recursive_add(c,a,b,a_shape[depth]*a_index, b_shape[depth]*b_index + s, b_shape[depth]*b_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)
    else:
        for s in range(a_shape[depth]):
            recursive_add(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index + s, a_shape[depth]*a_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)

@always_inline
fn e_add(inout c: Tensor, a: Tensor, b: Tensor):

    var a_shape = DynamicVector[Int](0)
    var b_shape = DynamicVector[Int](0)
    var a_strides = DynamicVector[Int](0)
    var b_strides = DynamicVector[Int](0)
    if(a.num_dims > b.num_dims): 
        for i in range(a.num_dims - b.num_dims):
            b_shape.push_back(1)
    elif(a.num_dims < b.num_dims): 
        for i in range(b.num_dims - a.num_dims):
            a_shape.push_back(1)
    for i in range(a.num_dims):
        a_shape.push_back(a.shape.load(i))
        b_shape.push_back(b.shape.load(i))
    for i in range(len(a_shape)):
        a_strides.push_back(1)
        b_strides.push_back(1)
    for i in range(len(a_shape)-2,-1,-1):
        a_strides[i] = a_strides[i+1]*a_shape[i+1]
        b_strides[i] = b_strides[i+1]*b_shape[i+1]

    recursive_add(c,a,b,0,0,0,0,a_shape,b_shape,a_strides,b_strides)


# e_sub - recursive call for proper broadcasting
fn recursive_sub(inout c: Tensor, borrowed a: Tensor, borrowed b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int, borrowed a_shape: DynamicVector[Int], borrowed b_shape: DynamicVector[Int], borrowed a_strides: DynamicVector[Int], borrowed b_strides: DynamicVector[Int]):
    
    if(depth == len(a_shape)):
        c.data.store(c_index, a.data.load(a_index) - b.data.load(b_index))
        return
    # if(a_strides[depth]*a_shape[depth] == b_strides[depth]*b_shape[depth] or depth == len(a_shape)):
    #     if(depth == len(a_shape)):
    #         c.data.store(c_index, a.data.load(a_index) - b.data.load(b_index))
    #         return
    #     else:
    #         @parameter
    #         fn v_sub[nelts: Int](i: Int):
    #             c.data.simd_store[nelts](
    #                 c_index*c.strides[depth]*c.shape[depth] + i, a.data.load(a_index*a_strides[depth]*a_shape[depth] + i) - b.data.load(b_index*b_strides[depth]*b_shape[depth] + i)
    #             )
    #         vectorize[nelts, v_sub](c.strides[depth]*c.shape[depth])
    #         return

    if(a_shape[depth] != 1 and b_shape[depth] == 1):
        for s in range(a_shape[depth]):
            recursive_sub(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index,a_shape[depth]*a_index + s, depth+1, a_shape, b_shape, a_strides, b_strides)
    elif(a_shape[depth] == 1 and b_shape[depth] != 1):
        for s in range(b_shape[depth]):
            recursive_sub(c,a,b,a_shape[depth]*a_index, b_shape[depth]*b_index + s, b_shape[depth]*b_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)
    else:
        for s in range(a_shape[depth]):
            recursive_sub(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index + s, a_shape[depth]*a_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)

@always_inline
fn e_sub(inout c: Tensor, a: Tensor, b: Tensor):

    var a_shape = DynamicVector[Int](0)
    var b_shape = DynamicVector[Int](0)
    var a_strides = DynamicVector[Int](0)
    var b_strides = DynamicVector[Int](0)
    if(a.num_dims > b.num_dims): 
        for i in range(a.num_dims - b.num_dims):
            b_shape.push_back(1)
    elif(a.num_dims < b.num_dims): 
        for i in range(b.num_dims - a.num_dims):
            a_shape.push_back(1)
    for i in range(a.num_dims):
        a_shape.push_back(a.shape.load(i))
        b_shape.push_back(b.shape.load(i))
    for i in range(len(a_shape)):
        a_strides.push_back(1)
        b_strides.push_back(1)
    for i in range(len(a_shape)-2,-1,-1):
        a_strides[i] = a_strides[i+1]*a_shape[i+1]
        b_strides[i] = b_strides[i+1]*b_shape[i+1]

    recursive_sub(c,a,b,0,0,0,0,a_shape,b_shape,a_strides,b_strides)

# e_div - recursive call for proper broadcasting
fn recursive_div(inout c: Tensor, borrowed a: Tensor, borrowed b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int, borrowed a_shape: DynamicVector[Int], borrowed b_shape: DynamicVector[Int], borrowed a_strides: DynamicVector[Int], borrowed b_strides: DynamicVector[Int]):

    if(depth == len(a_shape)):
        c.data.store(c_index, a.data.load(a_index) / b.data.load(b_index))
        return

    # if(a_strides[depth]*a_shape[depth] == b_strides[depth]*b_shape[depth] or depth == len(a_shape)):
    #     if(depth == len(a_shape)):
    #         c.data.store(c_index, a.data.load(a_index) / b.data.load(b_index))
    #         return
    #     else:
    #         @parameter
    #         fn v_div[nelts: Int](i: Int):
    #             c.data.simd_store[nelts](
    #                 c_index*c.strides[depth]*c.shape[depth] + i, a.data.load(a_index*a_strides[depth]*a_shape[depth] + i) / b.data.load(b_index*b_strides[depth]*b_shape[depth] + i)
    #             )
    #         vectorize[nelts, v_div](c.strides[depth]*c.shape[depth])
    #         return

    if(a_shape[depth] != 1 and b_shape[depth] == 1):
        for s in range(a_shape[depth]):
            recursive_div(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index,a_shape[depth]*a_index + s, depth+1, a_shape, b_shape, a_strides, b_strides)
    elif(a_shape[depth] == 1 and b_shape[depth] != 1):
        for s in range(b_shape[depth]):
            recursive_div(c,a,b,a_shape[depth]*a_index, b_shape[depth]*b_index + s, b_shape[depth]*b_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)
    else:
        for s in range(a_shape[depth]):
            recursive_div(c,a,b,a_shape[depth]*a_index + s, b_shape[depth]*b_index + s, a_shape[depth]*a_index + s,depth+1, a_shape, b_shape, a_strides, b_strides)

@always_inline
fn e_div(inout c: Tensor, a: Tensor, b: Tensor):

    var a_shape = DynamicVector[Int](0)
    var b_shape = DynamicVector[Int](0)
    var a_strides = DynamicVector[Int](0)
    var b_strides = DynamicVector[Int](0)
    if(a.num_dims > b.num_dims): 
        for i in range(a.num_dims - b.num_dims):
            b_shape.push_back(1)
    elif(a.num_dims < b.num_dims): 
        for i in range(b.num_dims - a.num_dims):
            a_shape.push_back(1)
    for i in range(a.num_dims):
        a_shape.push_back(a.shape.load(i))
        b_shape.push_back(b.shape.load(i))
    for i in range(len(a_shape)):
        a_strides.push_back(1)
        b_strides.push_back(1)
    for i in range(len(a_shape)-2,-1,-1):
        a_strides[i] = a_strides[i+1]*a_shape[i+1]
        b_strides[i] = b_strides[i+1]*b_shape[i+1]

    recursive_div(c,a,b,0,0,0,0,a_shape,b_shape,a_strides,b_strides)


@always_inline
fn e_sqrt(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_sqrt[nelts: Int](i: Int):
        let temp = sqrt(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_sqrt](a.cap)


@always_inline
fn e_abs(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_abs[nelts: Int](i: Int):
        let temp = abs(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_abs](a.cap)

@always_inline
fn e_exp2(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_exp2[nelts: Int](i: Int):
        let temp = exp2(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_exp2](a.cap)


@always_inline
fn e_exp(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_exp[nelts: Int](i: Int):
        let temp = exp(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_exp](a.cap)


@always_inline
fn e_log2(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_log2[nelts: Int](i: Int):
        let temp = log2(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_log2](a.cap)


@always_inline
fn e_log(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_log[nelts: Int](i: Int):
        let temp = log(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_log](a.cap)


@always_inline
fn e_sin(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_sin[nelts: Int](i: Int):
        let temp = sin(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_sin](a.cap)


@always_inline
fn e_cos(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_cos[nelts: Int](i: Int):
        let temp = cos(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_cos](a.cap)


@always_inline
fn e_tan(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_tan[nelts: Int](i: Int):
        let temp = tan(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_tan](a.cap)


@always_inline
fn e_asin(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_asin[nelts: Int](i: Int):
        let temp = asin(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_asin](a.cap)


@always_inline
fn e_acos(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_acos[nelts: Int](i: Int):
        let temp = acos(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_acos](a.cap)


@always_inline
fn e_atan(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_atan[nelts: Int](i: Int):
        let temp = atan(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_atan](a.cap)


@always_inline
fn e_sinh(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_sinh[nelts: Int](i: Int):
        let temp = sinh(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_sinh](a.cap)


@always_inline
fn e_cosh(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_cosh[nelts: Int](i: Int):
        let temp = cosh(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_cosh](a.cap)


@always_inline
fn e_tanh(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_tanh[nelts: Int](i: Int):
        let temp = tanh(a.data.simd_load[nelts](i))
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_tanh](a.cap)


@always_inline
fn e_relu(inout b: Tensor, a: Tensor): 
    @parameter
    fn v_relu[nelts: Int](i: Int):
        let zeros = SIMD[DType.float32,nelts]()
        b.data.simd_store[nelts](
            i, (a.data.simd_load[nelts](i) > zeros).cast[DType.float32]() * a.data.simd_load[nelts](i)
        )
    vectorize[nelts, v_relu](b.cap)


@always_inline
fn e_copy(inout b: Tensor, a: Tensor): 
    memcpy(b.data,a.data,a.cap)
    
