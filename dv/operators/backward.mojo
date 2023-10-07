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
fn mul_grad(C: Tensor, inout A: Tensor, inout B: Tensor):

    let A_matrix_size = A.shape[A.num_dims-2] * A.shape[A.num_dims-1]
    let B_matrix_size = B.shape[B.num_dims-2] * B.shape[B.num_dims-1]
    let C_matrix_size = C.shape[C.num_dims-2] * C.shape[C.num_dims-1] 

    let M = A.shape[A.num_dims-2]
    let K = B.shape[B.num_dims-2]
    let N = B.shape[B.num_dims-1]

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

        if (A.getRequiresGradient()):
            @parameter
            fn calc_row_1(m: Int):
                for n in range(N):
                    @parameter
                    fn dot[nelts: Int](k: Int):
                        let index_A = offset_A + m * K + k
                        let index_C = offset_C + m * N + n
                        let index_B = offset_B + k * N + n
                        let a = A.getGradient(index_A) + C.getGradient(index_C) * B.getData(index_B) 
                        A.setGradient(index_A, a)
                    vectorize[nelts, dot](K)
            parallelize[calc_row_1](M,M)

        if(B.requiresGradient):
            @parameter
            fn calc_row_2(k: Int):
                for m in range(M):
                    @parameter
                    fn dot[nelts: Int](n: Int):
                        let index_B = offset_B + k * N + n
                        let index_A = offset_A + m * K + k
                        let index_C = offset_C + m * N + n
                        let b = B.getGradient(index_B) + A.getData(index_A) * C.getGradient(index_C)  
                        B.setGradient(index_B, b)
                    vectorize[nelts, dot](N)
            parallelize[calc_row_2](K,K)

@always_inline        
fn add_grad(C: Tensor, inout A: Tensor, inout B: Tensor):

    # regular
    if(A.num_dims == B.num_dims):
        if(A.requiresGradient):
            @parameter
            fn v_add_gr_1[nelts: Int](i: Int):
                A.gradient.simd_store[nelts](
                    i, A.gradient.simd_load[nelts](i) + C.gradient.simd_load[nelts](i)
                )
            vectorize[nelts, v_add_gr_1](A.getCap())
        if(B.requiresGradient):
            @parameter
            fn v_add_gr_2[nelts: Int](i: Int):
                B.gradient.simd_store[nelts](
                    i, B.gradient.simd_load[nelts](i) + C.gradient.simd_load[nelts](i)
                )
            vectorize[nelts, v_add_gr_2](B.getCap())

    # consider broadcasting
    else:
        var offset_A: Int = 0
        var offset_B: Int = 0
        var offset_C: Int = 0
        var ratio: Int = 0
        var H = 0

        if(A.num_dims > B.num_dims):
            H = B.cap
            ratio = A.cap // B.cap
        else:
            H = A.cap
            ratio = B.cap // A.cap

        for s in range(ratio):
            if(A.num_dims > B.num_dims):
                offset_A = s * H
            else:
                offset_B = s * H

            offset_C = s * H
            if(A.requiresGradient):
                @parameter
                fn v_add_A[nelts: Int](i: Int):
                    A.gradient.simd_store[nelts](
                        offset_A + i, A.gradient.simd_load[nelts](offset_A + i) + C.gradient.simd_load[nelts](offset_C + i)
                    )
                vectorize[nelts, v_add_A](H) 

            if(B.requiresGradient):
                @parameter
                fn v_add_B[nelts: Int](i: Int):
                    B.gradient.simd_store[nelts](
                        offset_B + i, B.gradient.simd_load[nelts](offset_B + i) + C.gradient.simd_load[nelts](offset_C + i)
                    )
                vectorize[nelts, v_add_B](H) 

@always_inline
fn ReLU_grad(B: Tensor, inout A: Tensor):
    @parameter
    fn v_relu_bw[nelts: Int](i: Int):
        let zeros = SIMD[DType.float32,nelts]()
        A.gradient.simd_store[nelts](
            i, (A.data.simd_load[nelts](i) > zeros).cast[DType.float32]() * B.gradient.simd_load[nelts](i) + A.gradient.simd_load[nelts](i)
        )
    vectorize[nelts, v_relu_bw](A.getCap())

@always_inline
fn sum_grad(B: Tensor, inout A: Tensor):
    A.setGradientAll(1)

@always_inline
fn softmax_grad(B: Tensor, inout A: Tensor): 
    if(B.otherParams.load(0) == 3001):
        A.setGradient(B.getGradient())
    else:
        let num_dims = B.getNum_dims()
        let M = B.getShape(num_dims-2)
        let N = B.getShape(num_dims-1)
        for s in range(B.getCap() // N):
            let offset = s * N
            for j in range(N):
                var grad: Float32 = 0
                for i in range(N):
                    if(i == j):
                        grad += B.getGradient(offset + i) * ( B.getData(offset + j) * (Float32(1.0) - B.getData(offset + j)) )
                    else:
                        grad -= B.getGradient(offset + i)  * B.getData(offset + i) * B.getData(offset + j)
                A.setGradient(offset + j,  A.gradient.load(offset + j) + grad)

@always_inline
fn MSE_grad(C: Tensor, inout A: Tensor, inout B: Tensor): # A: TrueVals, B: Logits
    let num_dims = A.getNum_dims()
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    for index in range(A.getCap()):
        let grad = Float32(2) * (B.getData(index) - A.getData(index)) / Float32(A.getCap())
        if(A.requiresGradient):
            A.setGradient(index, A.getGradient(index) + grad) 
        if(B.requiresGradient):
            B.setGradient(index, B.getGradient(index) + grad) 

@always_inline
fn CE_grad(C: Tensor, inout A: Tensor, inout B: Tensor): # A: TrueVals, B: Logits
    let num_dims = A.getNum_dims()
    let N = A.getShape(num_dims-1)

    # for index in range(A.getCap()):
    #     let grad = B.getData(index) - A.getData(index)
    #     A.setGradient(index,  A.getGradient(index) +  grad /  (Float32(A.getCap()) / Float32(N)))))

    # for index in range(B.getCap()):
    #     let grad = B.getData(index) - A.getData(index)
    #     B.setGradient(index,  B.getGradient(index) + grad / (Float32(A.getCap()) / Float32(N)))))

    if(A.requiresGradient):
        if(A.name == "softmax"):
            for index in range(A.getCap()):
                let grad = (B.getData(index) - A.getData(index)) 
                A.setGradient(index,  A.getGradient(index) +  grad /  (Float32(A.getCap()) / Float32(N)))
            else:
                for index in range(A.getCap()):
                    let grad_A = - log(B.getData(index))
                    A.setGradient(index,  A.getGradient(index) +  grad_A / (Float32(A.getCap()) / Float32(N)))
    if(B.requiresGradient):
        if(B.name == "softmax"):
            for index in range(B.getCap()):
                let grad = (B.getData(index) - A.getData(index)) 
                B.setGradient(index,  B.getGradient(index) + grad / (Float32(A.getCap()) / Float32(N)))
        else:
            for index in range(B.getCap()):
                let grad_B = - A.getData(index) / (B.getData(index))
                B.setGradient(index,  B.getGradient(index) + grad_B / (Float32(A.getCap()) / Float32(N)))

@always_inline
fn reshape_grad(B: Tensor, inout A: Tensor):
    for s in range(B.cap // A.cap):
        let offset = s * A.cap
        for i in range(A.cap):
            A.setGradient(i,  A.getGradient(i) + B.getGradient(offset + i))


@always_inline
fn transpose_grad(B: Tensor, inout A: Tensor):
    let num_dims = B.getNum_dims()
    let M = B.getShape(num_dims-2)
    let N = B.getShape(num_dims-1)

    for s in range(B.getCap() // (M*N)):
        let offset = s * M * N
        for i in range(M):
            for j in range(N):
                A.setGradient(offset + j * M + i,  A.getGradient(offset + j * M + i) + B.getGradient(offset + i * N + j))
