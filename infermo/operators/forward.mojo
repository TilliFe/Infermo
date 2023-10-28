from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import max, min, sqrt, abs, pow, exp2, exp, log2, log, cos, sin, tan, asin, acos, atan, cosh, sinh, tanh
from sys.param_env import env_get_int

from ..graph.tensor import Tensor


alias nelts = simdwidthof[DType.float32]()
alias workers = env_get_int["WORKERS", 0]()


################################################################################################################################

# broadcasting helpers #########################################################################################################

fn shape_a(depth: Int, a: Tensor, b: Tensor) -> Int:
    let diff = max(b.num_dims - a.num_dims,0)
    if(depth < diff):
        return 1
    return a.shape[depth - diff]

fn shape_b(depth: Int, a: Tensor, b: Tensor) -> Int:
    let diff = max(a.num_dims - b.num_dims,0)
    if(depth < diff):
        return 1
    return b.shape[depth - diff]

fn strides_a(depth: Int, a: Tensor, b: Tensor) -> Int:
    let diff = max(b.num_dims - a.num_dims,0)
    if(depth < diff):
        return a.strides[0]
    return a.strides[depth - diff]

fn strides_b(depth: Int, a: Tensor, b: Tensor) -> Int:
    let diff = max(a.num_dims - b.num_dims,0)
    if(depth < diff):
        return b.strides[0]
    return b.strides[depth - diff]


# recursive broadcast
fn recursive_broadcast_fw[kernel: fn(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int ) -> None, base_case: fn(depth: Int, a: Tensor, b: Tensor) -> Bool](
    inout c: Tensor, 
    a: Tensor, 
    b: Tensor, 
    a_index: Int=0, 
    b_index: Int=0, 
    c_index: Int=0, 
    depth: Int=0
):

    # base case - launch kernel
    if(base_case(depth,a,b)):
        kernel(c,a,b,a_index,b_index,c_index,depth)
        return

    # go into depth
    let a_shape = shape_a(depth,a,b)
    let b_shape = shape_b(depth,a,b)
    let c_shape = c.shape[depth]
    if(a_shape != 1 and b_shape == 1):
        for s in range(a_shape):
            recursive_broadcast_fw[kernel,base_case](
                c,a,b,
                a_shape*a_index + s, 
                b_shape*b_index,
                c_shape*c_index + s, 
                depth+1
            )
    elif(a_shape == 1 and b_shape != 1):
        for s in range(b_shape):
            recursive_broadcast_fw[kernel,base_case](
                c,a,b,
                a_shape*a_index, 
                b_shape*b_index + s, 
                c_shape*c_index + s,
                depth+1
            )
    else:
        for s in range(a_shape):
            recursive_broadcast_fw[kernel,base_case](
                c,a,b,
                a_shape*a_index + s, 
                b_shape*b_index + s, 
                c_shape*c_index + s,
                depth+1
            )


# Non-elementwise operators ####################################################################################################

@parameter
fn base_case_matmul_fw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return depth == max(a.num_dims,b.num_dims)-2

@parameter
fn kernel_matmul_fw(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
 
    let offset_a = a_index * a.shape[a.num_dims-2] * a.shape[a.num_dims-1]
    let offset_b = b_index * b.shape[b.num_dims-2] * b.shape[b.num_dims-1]
    let offset_c = c_index * c.shape[c.num_dims-2] * c.shape[c.num_dims-1] 

    let M = a.shape[a.num_dims-2]
    let K = b.shape[b.num_dims-2]
    let N = b.shape[b.num_dims-1]

    @parameter
    fn calc_row_fw(m: Int):
        for k in range(K):
            @parameter
            fn dot_fw[nelts: Int](n: Int):
                c.data.simd_store[nelts](
                    offset_c + m*N+n, 
                    c.data.simd_load[nelts](offset_c + m*N+n) + a.data.load(offset_a + m*K+k) * b.data.simd_load[nelts](offset_b + k*N+n))
            vectorize[nelts, dot_fw](N)
    parallelize[calc_row_fw](M, workers if workers > 0 else M) 

@always_inline
fn matmul(inout c: Tensor, a: Tensor, b: Tensor):
    recursive_broadcast_fw[kernel_matmul_fw, base_case_matmul_fw](c,a,b)


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
fn reduce_unary_operation_iterator[op: fn[_nelts: Int](Int, Int) capturing -> None, op_b: fn(Int) capturing -> None](inout b: Tensor, a: Tensor):
    let dim_len: Int = b.other_params.load(0)
    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]
    
    # Mark which dimensions are we going to reduce
    var in_dims = DynamicVector[Bool](b.num_dims)
    for d in range(b.num_dims):
        in_dims[d] = False
    for d in range(dim_len):
        in_dims[b.other_params.load(d+1)] = True


    let last_dim = b.other_params.load(dim_len)
    let last_dim_size = a.shape[last_dim]
    let last_stride = a.strides[last_dim]

    @parameter
    fn p_op(i: Int):
        var indeces = DynamicVector[Int]()
        var offset_a = 0
        # Calculate offset of a tensor, based on the position of b tensor
        for dim in range(a.num_dims - 1, -1, -1):
            if not in_dims[dim]:
                offset_a += ((i // b.strides[dim]) % b.shape[dim]) * a.strides[dim]
            indeces.push_back(0)

        for j in range(total_elements_in_dims // last_dim_size):
            @parameter
            fn v_op[nelts: Int](k: Int):
                let offset_a_local = offset_a + k * last_stride
                
                op[nelts](i, offset_a_local) # do reduce operation

            vectorize[nelts, v_op](last_dim_size)

            # Increment indeces of a tensor minus the last dimension, because we iterate over the last dimension in the inner loop k (the size loop)
            var count = 0
            for dim in range(a.num_dims - 1, -1, -1):
                if in_dims[dim] and count == 0:
                    count += 1
                    continue
                if not in_dims[dim]:
                    continue
                indeces[dim] += 1
                offset_a += a.strides[dim]
                if indeces[dim] < a.shape[dim]:
                    break
                indeces[dim] = 0
                offset_a -= a.strides[dim] * a.shape[dim]

        op_b(i) # Final operation on the result of the reduction in tensor b

    parallelize[p_op](b.cap, workers if workers > 0 else b.cap)

@always_inline
fn mean(inout b: Tensor, a: Tensor):
    let dim_len: Int = b.other_params.load(0)
    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]

    @parameter
    fn _sum[nelts: Int](idx_b: Int, idx_a: Int):
        b.data.store(idx_b, b.data.load(idx_b) + a.data.simd_load[nelts](idx_a).reduce_add())

    @parameter
    fn _mean(idx_b: Int):
        b.data.store(idx_b, b.data.load(idx_b) / Float32(total_elements_in_dims))

    reduce_unary_operation_iterator[_sum, _mean](b, a)


@always_inline
fn variance(inout b: Tensor, a: Tensor): 
    let dim_len: Int = b.other_params.load(0)
    var b_dims = DynamicVector[Int](0)
    for d in range(b.num_dims):
        b_dims.push_back(b.shape[d])
    var mean_output = Tensor(b_dims)
    mean_output.other_params = b.other_params

    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]
    
    # Calculate the mean of the tensor
    mean(mean_output, a)
    
    @parameter
    fn _diff[nelts: Int](idx_b: Int, idx_a: Int):
        let diff = a.data.simd_load[nelts](idx_a) - mean_output.data.load(idx_b)
        b.data.store(idx_b, b.data.load(idx_b) + (diff ** 2).reduce_add())

    @parameter
    fn _variance(idx_b: Int):
        b.data.store(idx_b, b.data.load(idx_b) / Float32(total_elements_in_dims - 1)) # / (n - 1)

    reduce_unary_operation_iterator[_diff, _variance](b, a)

@always_inline
fn std(inout b: Tensor, a: Tensor): 
    """Population standard deviation."""
    let dim_len: Int = b.other_params.load(0)
    var b_dims = DynamicVector[Int](0)
    for d in range(b.num_dims):
        b_dims.push_back(b.shape[d])
    var mean_output = Tensor(b_dims)
    mean_output.other_params = b.other_params

    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = b.other_params.load(d+1)
        total_elements_in_dims *= a.shape[dim]
    
    # Calculate the mean of the tensor
    mean(mean_output, a)
    
    @parameter
    fn _diff[nelts: Int](idx_b: Int, idx_a: Int):
        let diff = a.data.simd_load[nelts](idx_a) - mean_output.data.load(idx_b)
        b.data.store(idx_b, b.data.load(idx_b) + (diff ** 2).reduce_add())

    @parameter
    fn _std(idx_b: Int):
        b.data.store(idx_b, sqrt(b.data.load(idx_b) / Float32(total_elements_in_dims - 1))) # / (n - 1)

    reduce_unary_operation_iterator[_diff, _std](b, a)


# elementwise operators ####################################################

@parameter
fn base_case_mul_fw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_mul_fw(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:

    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_mul[nelts: Int](i: Int):
        c.data.simd_store[nelts](
            offset_c + i, a.data.simd_load[nelts](offset_a + i) * b.data.simd_load[nelts](offset_b + i)
        )
    vectorize[nelts, v_mul](c_rest)

@always_inline
fn e_mul(inout c: Tensor, a: Tensor, b: Tensor):
    recursive_broadcast_fw[kernel_mul_fw, base_case_mul_fw](c,a,b)


@parameter
fn base_case_add_fw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_add_fw(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_add[nelts: Int](i: Int):
        c.data.simd_store[nelts](
            offset_c + i, a.data.simd_load[nelts](offset_a + i) + b.data.simd_load[nelts](offset_b + i)
        )
    vectorize[nelts, v_add](c_rest)

@always_inline
fn e_add(inout c: Tensor, a: Tensor, b: Tensor):
    recursive_broadcast_fw[kernel_add_fw, base_case_add_fw](c,a,b)


@parameter
fn base_case_sub_fw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_sub_fw(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_sub[nelts: Int](i: Int):
        c.data.simd_store[nelts](
            offset_c + i, a.data.simd_load[nelts](offset_a + i) - b.data.simd_load[nelts](offset_b + i)
        )
    vectorize[nelts, v_sub](c_rest)

@always_inline
fn e_sub(inout c: Tensor, a: Tensor, b: Tensor):
    recursive_broadcast_fw[kernel_sub_fw, base_case_sub_fw](c,a,b)


@parameter
fn base_case_div_fw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_div_fw(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_div[nelts: Int](i: Int):
        c.data.simd_store[nelts](
            offset_c + i, a.data.simd_load[nelts](offset_a + i) / b.data.simd_load[nelts](offset_b + i)
        )
    vectorize[nelts, v_div](c_rest)

@always_inline
fn e_div(inout c: Tensor, a: Tensor, b: Tensor):
    recursive_broadcast_fw[kernel_div_fw, base_case_div_fw](c,a,b)


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


# pow(a,b)
@parameter
fn base_case_pow_fw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_pow_fw(inout c: Tensor, a: Tensor, b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:

    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_pow[nelts: Int](i: Int):
        c.data.simd_store[nelts](
            offset_c + i, pow(a.data.simd_load[nelts](offset_a + i), b.data.simd_load[nelts](offset_b + i))
        )
    vectorize[nelts, v_pow](c_rest)

@always_inline
fn e_pow(inout c: Tensor, a: Tensor, b: Tensor):
    recursive_broadcast_fw[kernel_pow_fw, base_case_pow_fw](c,a,b)


# pow(a,<some_number>)
fn e_pow_all(inout b: Tensor, a: Tensor): 
    let e = b.other_params.load(0)
    @parameter
    fn v_pow_all[nelts: Int](i: Int):
        let temp = pow(a.data.simd_load[nelts](i),e)
        b.data.simd_store[nelts](i, temp)
    vectorize[nelts, v_pow_all](a.cap)


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
    