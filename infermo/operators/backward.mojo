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
fn recursive_broadcast_bw[kernel: fn(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int ) -> None, base_case: fn(depth: Int, a: Tensor, b: Tensor) -> Bool](
    c: Tensor, 
    inout a: Tensor, 
    inout b: Tensor, 
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
            recursive_broadcast_bw[kernel,base_case](
                c,a,b,
                a_shape*a_index + s, 
                b_shape*b_index,
                c_shape*c_index + s, 
                depth+1
            )
    elif(a_shape == 1 and b_shape != 1):
        for s in range(b_shape):
            recursive_broadcast_bw[kernel,base_case](
                c,a,b,
                a_shape*a_index, 
                b_shape*b_index + s, 
                c_shape*c_index + s,
                depth+1
            )
    else:
        for s in range(a_shape):
            recursive_broadcast_bw[kernel,base_case](
                c,a,b,
                a_shape*a_index + s, 
                b_shape*b_index + s, 
                c_shape*c_index + s,
                depth+1
            )


# non-elementwise operators #####################################################################################

@parameter
fn base_case_matmul_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return depth == max(a.num_dims,b.num_dims)-2

@parameter
fn kernel_matmul_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:

    let offset_a = a_index * a.shape[a.num_dims-2] * a.shape[a.num_dims-1]
    let offset_b = b_index * b.shape[b.num_dims-2] * b.shape[b.num_dims-1]
    let offset_c = c_index * c.shape[c.num_dims-2] * c.shape[c.num_dims-1] 

    let M = a.shape[a.num_dims-2]
    let K = b.shape[b.num_dims-2]
    let N = b.shape[b.num_dims-1]
    
    @parameter
    fn calc_row_1(m: Int):
        for n in range(N):
            @parameter
            fn dot_bw_a[nelts: Int](k: Int):
                let val = a.grad.load(offset_a + m * K + k) + c.grad.load(offset_c + m * N + n) * b.data.load(offset_b + k * N + n) 
                a.grad.store(offset_a + m * K + k, val)
            vectorize[nelts, dot_bw_a](K)
    parallelize[calc_row_1](M, workers if workers > 0 else M)

@parameter
fn kernel_matmul_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:

    let offset_a = a_index * a.shape[a.num_dims-2] * a.shape[a.num_dims-1]
    let offset_b = b_index * b.shape[b.num_dims-2] * b.shape[b.num_dims-1]
    let offset_c = c_index * c.shape[c.num_dims-2] * c.shape[c.num_dims-1] 

    let M = a.shape[a.num_dims-2]
    let K = b.shape[b.num_dims-2]
    let N = b.shape[b.num_dims-1]
    
    @parameter
    fn calc_row_2(k: Int):
        for m in range(M):
            @parameter
            fn dot_bw_b[nelts: Int](n: Int):
                let val = b.grad.load(offset_b + k * N + n) + a.data.load(offset_a + m * K + k) * c.grad.load(offset_c + m * N + n)  
                b.grad.store(offset_b + k * N + n, val)
            vectorize[nelts, dot_bw_b](N)
    parallelize[calc_row_2](K, workers if workers > 0 else K)

@always_inline
fn matmul_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_matmul_bw_a, base_case_matmul_bw](c,a,b)
    recursive_broadcast_bw[kernel_matmul_bw_b, base_case_matmul_bw](c,a,b)


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

    parallelize[batch_loop](a.shape[0], workers if workers > 0 else a.shape[0])

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
fn mean_grad(b: Tensor, inout a: Tensor): 
    pass


@always_inline
fn variance_grad(b: Tensor, inout a: Tensor): 
    pass

@always_inline
fn std_grad(b: Tensor, inout a: Tensor): 
    pass


# elementwise operators #####################################################################################################

# binary operators grads ##################################################

# mul grad #####################
@parameter
fn base_case_mul_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_mul_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_mul_grad_a[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            offset_a + i, a.grad.simd_load[nelts](offset_a + i) + b.data.simd_load[nelts](offset_b + i) *  c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_mul_grad_a](c_rest)

@parameter
fn kernel_mul_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_mul_grad_b[nelts: Int](i: Int):
        b.grad.simd_store[nelts](
            offset_b + i, b.grad.simd_load[nelts](offset_b + i) + a.data.simd_load[nelts](offset_a + i) * c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_mul_grad_b](c_rest)

@always_inline
fn e_mul_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_mul_bw_a, base_case_mul_bw](c,a,b)
    recursive_broadcast_bw[kernel_mul_bw_b, base_case_mul_bw](c,a,b)


# add grad #####################
@parameter
fn base_case_add_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_add_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_add_grad_a[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            offset_a + i, a.grad.simd_load[nelts](offset_a + i) + c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_add_grad_a](c_rest)

@parameter
fn kernel_add_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_add_grad_b[nelts: Int](i: Int):
        b.grad.simd_store[nelts](
            offset_b + i, b.grad.simd_load[nelts](offset_b + i) + c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_add_grad_b](c_rest)

@always_inline
fn e_add_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_add_bw_a, base_case_add_bw](c,a,b)
    recursive_broadcast_bw[kernel_add_bw_b, base_case_add_bw](c,a,b)


# sub grad #####################
@parameter
fn base_case_sub_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_sub_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_sub_grad_a[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            offset_a + i, a.grad.simd_load[nelts](offset_a + i) + c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_sub_grad_a](c_rest)

@parameter
fn kernel_sub_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_sub_grad_b[nelts: Int](i: Int):
        b.grad.simd_store[nelts](
            offset_b + i, b.grad.simd_load[nelts](offset_b + i) - c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_sub_grad_b](c_rest)

@always_inline
fn e_sub_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_sub_bw_a, base_case_sub_bw](c,a,b)
    recursive_broadcast_bw[kernel_sub_bw_b, base_case_sub_bw](c,a,b)


# div grad #####################
@parameter
fn base_case_div_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_div_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_div_grad_a[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            offset_a + i, a.grad.simd_load[nelts](offset_a + i) + c.grad.simd_load[nelts](offset_c + i) / b.data.simd_load[nelts](offset_b + i)
        )
    vectorize[nelts, v_div_grad_a](c_rest)

@parameter
fn kernel_div_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:
    
    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_div_grad_b[nelts: Int](i: Int):
        b.grad.simd_store[nelts](
            offset_b + i, b.grad.simd_load[nelts](offset_b + i) - a.data.simd_load[nelts](offset_a + i) * c.grad.simd_load[nelts](offset_c + i)  / pow(b.data.simd_load[nelts](offset_b + i),2)
        )
    vectorize[nelts, v_div_grad_b](c_rest)

@always_inline
fn e_div_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_div_bw_a, base_case_div_bw](c,a,b)
    recursive_broadcast_bw[kernel_div_bw_b, base_case_div_bw](c,a,b)


# other unary operators grad ##############################################################################################################
@always_inline
fn e_sqrt_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_sqrt_grad[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) / (Float32(2.0) * sqrt(a.data.simd_load[nelts](i)))
        )
    vectorize[nelts, v_sqrt_grad](a.cap)


@always_inline
fn e_abs_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_abs_bw[nelts: Int](i: Int):
        let zeros = SIMD[DType.float32,nelts]()
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + (Float32(2.0) * (a.data.simd_load[nelts](i) >= zeros).cast[DType.float32]() - Float32(1.0)) * b.grad.simd_load[nelts](i)
        )
    vectorize[nelts, v_abs_bw](a.cap)


# pow(a,b)
@parameter
fn base_case_pow_bw(depth: Int, a: Tensor, b: Tensor) -> Bool:
    return strides_a(depth,a,b)*shape_a(depth,a,b) == strides_b(depth,a,b)*shape_b(depth,a,b)

@parameter
fn kernel_pow_bw_a(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:

    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_pow_bw_a[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            offset_a + i, a.grad.simd_load[nelts](offset_a + i) + b.data.simd_load[nelts](offset_b + i) * pow(a.data.simd_load[nelts](offset_a + i),b.data.simd_load[nelts](offset_b + i) - Float32(1.0)) * c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_pow_bw_a](c_rest)

@parameter
fn kernel_pow_bw_b(c: Tensor, inout a: Tensor, inout b: Tensor, a_index: Int, b_index: Int, c_index: Int, depth: Int) -> None:

    let offset_a = a_index*shape_a(depth,a,b)*strides_a(depth,a,b)
    let offset_b = b_index*shape_b(depth,a,b)*strides_b(depth,a,b)
    let c_rest = c.shape[depth]*c.strides[depth]
    let offset_c = c_index*c_rest

    @parameter
    fn v_pow_bw_b[nelts: Int](i: Int):
        b.grad.simd_store[nelts](
            offset_b + i, b.grad.simd_load[nelts](offset_b + i) + c.data.simd_load[nelts](offset_c + i) * log(a.data.simd_load[nelts](offset_a + i)) * c.grad.simd_load[nelts](offset_c + i)
        )
    vectorize[nelts, v_pow_bw_b](c_rest)

@always_inline
fn e_pow_grad(c: Tensor, inout a: Tensor, inout b: Tensor):
    recursive_broadcast_bw[kernel_pow_bw_a, base_case_pow_bw](c,a,b)
    recursive_broadcast_bw[kernel_pow_bw_b, base_case_pow_bw](c,a,b)


# pow(a,<some_number>)
fn e_pow_all_grad(b: Tensor, inout a: Tensor): 
    let e = b.other_params.load(0)
    @parameter
    fn v_pow_all_bw[nelts: Int](i: Int):
        let temp = e * pow(a.data.simd_load[nelts](i),e-1) * b.grad.simd_load[nelts](i)
        a.grad.simd_store[nelts](i, temp)
    vectorize[nelts, v_pow_all_bw](a.cap)

@always_inline
fn e_exp2_grad(b: Tensor, inout a: Tensor): 
    @parameter
    fn v_exp2_grad[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) * b.data.simd_load[nelts](i) * log(Float32(2.0))
        )
    vectorize[nelts, v_exp2_grad](a.cap)


@always_inline
fn e_exp_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_exp_grad[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) * b.data.simd_load[nelts](i)
        )
    vectorize[nelts, v_exp_grad](a.cap)


@always_inline
fn e_log2_grad(b: Tensor, inout a: Tensor): 
    @parameter
    fn v_log2_grad[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) / (a.data.simd_load[nelts](i) * log(Float32(2.0)))
        )
    vectorize[nelts, v_log2_grad](a.cap)


@always_inline
fn e_log_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_log_grad[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) / a.data.simd_load[nelts](i)
        )
    vectorize[nelts, v_log_grad](a.cap)


@always_inline
fn e_sin_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_sin_grad[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) * cos(a.data.simd_load[nelts](i))
        )
    vectorize[nelts, v_sin_grad](a.cap)

@always_inline
fn e_cos_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_cos_bw[nelts: Int](i: Int):
       a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) - b.grad.simd_load[nelts](i) * sin(a.data.simd_load[nelts](i))
        )
    vectorize[nelts, v_cos_bw](a.cap)

@always_inline
fn e_tan_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_tan_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) / pow(cos(a.data.simd_load[nelts](i)),2) # 1.0 / sec^2(a)
        )
    vectorize[nelts, v_tan_bw](a.cap)

@always_inline
fn e_asin_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_asin_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) / sqrt(Float32(1.0) - pow(a.data.simd_load[nelts](i),2))
        )
    vectorize[nelts, v_asin_bw](a.cap)


@always_inline
fn e_acos_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_acos_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) - b.grad.simd_load[nelts](i) / sqrt(Float32(1.0) - pow(a.data.simd_load[nelts](i),2))
        )
    vectorize[nelts, v_acos_bw](a.cap)

@always_inline
fn e_atan_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_atan_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) / (Float32(1.0) + pow(a.data.simd_load[nelts](i),2))
        )
    vectorize[nelts, v_atan_bw](a.cap)


@always_inline
fn e_sinh_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_sinh_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) * cosh(a.data.simd_load[nelts](i))
        )
    vectorize[nelts, v_sinh_bw](a.cap)


@always_inline
fn e_cosh_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_cosh_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) * sinh(a.data.simd_load[nelts](i))
        )
    vectorize[nelts, v_cosh_bw](a.cap)


@always_inline
fn e_tanh_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_tanh_bw[nelts: Int](i: Int):
        a.grad.simd_store[nelts](
            i, a.grad.simd_load[nelts](i) + b.grad.simd_load[nelts](i) * (1.0 - pow(b.data.simd_load[nelts](i),2))
        )
    vectorize[nelts, v_tanh_bw](a.cap)


@always_inline
fn e_relu_grad(b: Tensor, inout a: Tensor):
    @parameter
    fn v_relu_bw[nelts: Int](i: Int):
        let zeros = SIMD[DType.float32,nelts]()
        a.grad.simd_store[nelts](
            i, (a.data.simd_load[nelts](i) > zeros).cast[DType.float32]() * b.grad.simd_load[nelts](i) + a.grad.simd_load[nelts](i)
        )
    vectorize[nelts, v_relu_bw](a.cap)


@always_inline
fn e_copy_grad(b: Tensor, inout a: Tensor): 
    memcpy(a.grad,b.grad,a.cap)
