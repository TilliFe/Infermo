from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp

from ..graph.tensor import Tensor

alias nelts = simdwidthof[DType.float32]()

@always_inline
fn mul(inout C: Tensor, A: Tensor, B: Tensor, rt: Runtime):
    let num_dims = A.getNum_dims()
    var A_matrix_size = A.shape[num_dims-2] * A.shape[num_dims-1]
    var B_matrix_size = B.shape[num_dims-2] * B.shape[num_dims-1]
    var C_matrix_size = C.shape[num_dims-2] * C.shape[num_dims-1]

    let M = C.shape[num_dims-2]
    let K = A.shape[num_dims-1]
    let N = C.shape[num_dims-1] 

    for s in range(C.getCap() // C_matrix_size):
        let offset_A = s * A_matrix_size
        let offset_B = s * B_matrix_size
        let offset_C = s * C_matrix_size

        @parameter
        fn calc_row(m: Int):
            for k in range(K):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.data.simd_store[nelts](offset_C + m*N+n, C.data.simd_load[nelts](offset_C + m*N+n) + A.data.load(offset_A + m*K+k) * B.data.simd_load[nelts](offset_B + k*N+n))
                vectorize[nelts, dot](N)
        parallelize[calc_row](rt, M)

@always_inline
fn add(inout C: Tensor, A: Tensor, B: Tensor):
    let num_dims = A.getNum_dims()
    let matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)

    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    if(C.getCap() > nelts):
        for i in range(0,C.getCap() - (nelts), nelts):
            C.data.simd_store[nelts](i, A.data.simd_load[nelts](i) + B.data.simd_load[nelts](i))
        for i in range(C.getCap() - nelts, C.getCap()):
            C.data.store(i, A.data.load(i) + B.data.load(i))
    else:
        for i in range(C.getCap()):
            C.data.store(i, A.data.load(i) + B.data.load(i))

@always_inline
fn ReLU(inout B: Tensor, A: Tensor):
    for i in range(A.getCap()):
        let val = A.getData(i)
        if(val < 0):
            B.setData(i,0)
        else:
            B.setData(i,val)

@always_inline
fn sum(inout B: Tensor, A: Tensor):
    var sum: Float32 = 0
    for i in range(A.getCap()):
        sum += A.getData(i)
    B.setData(0,sum)

@always_inline
fn softmax(inout B: Tensor, A: Tensor):
    #by default take the softmax along the last dimension of the tensor
    let num_dims = A.getNum_dims()
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

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


@always_inline
fn MSE(inout C: Tensor, A: Tensor, B: Tensor):
    for index in range(A.getCap()):
        let error = (A.getData(index) - B.getData(index)) * (A.getData(index) - B.getData(index))
        C.setData(0, C.getData(0) + error)
    C.setData(0, C.getData(0) / A.getCap())

@always_inline
fn CE(inout C: Tensor, A: Tensor, B: Tensor):

    let num_dims = A.getNum_dims()
    let M = A.shape[num_dims-2]
    for index in range(A.getCap()):
        let error = -A.getData(index) * log(B.getData(index)) #(A.getData(index) - B.getData(index)) * (A.getData(index) - B.getData(index))
        C.setData(0, C.getData(0) + error)
    C.setData(0, C.getData(0) / M)

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
