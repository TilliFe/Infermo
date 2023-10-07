from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp, abs

from ..graph.tensor import Tensor

alias nelts = simdwidthof[DType.float32]()

@always_inline
fn mul(inout C: Tensor, A: Tensor, B: Tensor):
    var A_matrix_size = A.shape[A.num_dims-2] * A.shape[A.num_dims-1]
    var B_matrix_size = B.shape[B.num_dims-2] * B.shape[B.num_dims-1]
    var C_matrix_size = C.shape[C.num_dims-2] * C.shape[C.num_dims-1]

    let M = C.shape[C.num_dims-2]
    let K = A.shape[A.num_dims-1]
    let N = C.shape[C.num_dims-1] 

    var offset_A: Int = 0
    var offset_B: Int = 0
    var offset_C: Int = 0

    for s in range(C.getCap() // C_matrix_size):
        
        offset_C = s * C_matrix_size

        # consider broadcasting
        if(A.num_dims == B.num_dims):
            offset_A = s * A_matrix_size
            offset_B = s * B_matrix_size
        elif(A.num_dims > B.num_dims):
            offset_A = s * A_matrix_size
        else:
            offset_B = s * B_matrix_size

        @parameter
        fn calc_row(m: Int):
            for k in range(K):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.data.simd_store[nelts](offset_C + m*N+n, C.data.simd_load[nelts](offset_C + m*N+n) + A.data.load(offset_A + m*K+k) * B.data.simd_load[nelts](offset_B + k*N+n))
                vectorize[nelts, dot](N)
        parallelize[calc_row](M,M)

@always_inline
fn add(inout C: Tensor, A: Tensor, B: Tensor):
    if(A.num_dims == B.num_dims):
        @parameter
        fn v_add_1[nelts: Int](i: Int):
            C.data.simd_store[nelts](
                i, A.data.simd_load[nelts](i) + B.data.simd_load[nelts](i)
            )
        vectorize[nelts, v_add_1](C.getCap())

    elif(A.num_dims > B.num_dims):
        for s in range(A.getCap() // B.getCap()):
            let offset = s * B.getCap()
            @parameter
            fn v_add_2[nelts: Int](i: Int):
                C.data.simd_store[nelts](
                    offset + i, A.data.simd_load[nelts](offset + i) + B.data.simd_load[nelts](i)
                )
            vectorize[nelts, v_add_2](B.getCap())

    else: # (B.num_dims > A.num_dims)
        for s in range(B.getCap() // A.getCap()):
            let offset = s * A.getCap()
            @parameter
            fn v_add_3[nelts: Int](i: Int):
                C.data.simd_store[nelts](
                    offset + i, A.data.simd_load[nelts](i) + B.data.simd_load[nelts](offset + i)
                )
            vectorize[nelts, v_add_3](A.getCap())


@always_inline
fn ReLU(inout B: Tensor, A: Tensor): 
    @parameter
    fn v_relu[nelts: Int](i: Int):
        let zeros = SIMD[DType.float32,nelts]()
        B.data.simd_store[nelts](
            i, (A.data.simd_load[nelts](i) > zeros).cast[DType.float32]() * A.data.simd_load[nelts](i)
        )
    vectorize[nelts, v_relu](B.getCap())

@always_inline
fn sum(inout B: Tensor, A: Tensor):
    var sum: Float32 = 0
    for i in range(A.getCap()):
        sum += A.getData(i)
    B.setData(0,sum)

@always_inline
fn softmax(inout B: Tensor, A: Tensor):
    # #by default take the softmax along the last dimension of the tensor
    let num_dims = A.getNum_dims()
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)
    let num_rows = A.getCap() // N

    for i in range(B.getCap()):
        B.data.store(i,exp(A.data.load(i)))

    for s in range(B.getCap() // (M*N)):
        let offset = s * M * N
        for m in range(M):
            var sum : Float32 = 0
            for n in range(N):
                sum += B.data.load(offset + m*N + n)
            for n in range(N):
                B.data.store(offset + m*N + n, B.data.load(offset + m*N + n) / sum)

    # this does not work yet
    # for s in range(B.cap // N):
    #     @parameter
    #     fn v_exp[nelts: Int](i: Int):
    #         B.data.simd_store[nelts](s*N + i, ((1.0 + A.data.simd_load[nelts](s*N + i) + A.data.simd_load[nelts](s*N + i).__pow__(2) / 2.0 + A.data.simd_load[nelts](s*N + i).__pow__(3) / 6.0 + A.data.simd_load[nelts](s*N + i).__pow__(4) / 24.0 + A.data.simd_load[nelts](s*N + i).__pow__(5) / 120.0  ) ))
    #     vectorize[nelts, v_exp](N)

    #     var row_sum = SIMD[DType.float32,1]()
    #     @parameter
    #     fn v_sum[nelts: Int](i: Int):
    #         row_sum = row_sum + B.data.simd_load[nelts](s*N + i).reduce_add()
    #     vectorize[nelts, v_sum](N)

    #     @parameter
    #     fn v_div[nelts: Int](i: Int):
    #         B.data.simd_store[nelts](s*N + i, B.data.simd_load[nelts](s*N + i) / row_sum)
    #     vectorize[nelts, v_div](N)
        
@always_inline
fn MSE(inout C: Tensor, A: Tensor, B: Tensor):
    for index in range(A.getCap()):
        let error = (A.getData(index) - B.getData(index)) * (A.getData(index) - B.getData(index))
        C.setData(0, C.getData(0) + error)
    C.setData(0, C.getData(0) / Float32(A.getCap()))

@always_inline
fn CE(inout C: Tensor, A: Tensor, B: Tensor):
    let num_dims = A.getNum_dims()
    let N = A.shape[num_dims-1]
    let epsilon = Float32(1e-8)
    for index in range(A.getCap()):
        if(B.getData(index) > Float32(0.0001)):
            let error = -A.getData(index) * log(B.getData(index) + epsilon)
            C.setData(0, C.getData(0) + error)
    C.setData(0, C.getData(0) / (Float32(A.getCap()) / Float32(N)))


@always_inline
fn reshape(inout B: Tensor, A: Tensor):
    for s in range(B.cap // A.cap):
        let offset = s * A.cap
        for i in range(A.cap):
            B.setData(offset + i, A.getData(i))


@always_inline
fn transpose(inout B: Tensor, A: Tensor):
    
    # we always tranpose along the last two dimensions of the tensor

    let num_dims = A.getNum_dims()
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    for s in range(B.getCap() // (M*N)):
        let offset = s * M * N
        for i in range(M):
            for j in range(N):
                B.setData(offset + j * M + i, A.getData(offset + i * N + j))
