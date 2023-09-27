from Tensor import Tensor

@always_inline
fn mul_grad(C: Tensor, inout A: Tensor, inout B: Tensor):

    let num_dims = A.getNum_dims()
    var A_matrix_size = A.shape[num_dims-2] * A.shape[num_dims-1]
    var B_matrix_size = B.shape[num_dims-2] * B.shape[num_dims-1]
    var C_matrix_size = C.shape[num_dims-2] * C.shape[num_dims-1]
    if(num_dims >= 3):
        A_matrix_size = A.skips[num_dims-3]
        B_matrix_size = B.skips[num_dims-3]
        C_matrix_size = C.skips[num_dims-3]   

    let M = A.shape[num_dims-2]
    let K = B.shape[num_dims-2]  
    let N = B.shape[num_dims-1]

    for s in range(A.getCap() // A_matrix_size):
        let offset_C = s * C_matrix_size
        let offset_B = s * B_matrix_size
        let offset_A = s * A_matrix_size 

        if (A.getRequiresGradient()):
            for m in range(M):
                for k in range(K):
                    for n in range(N):
                        let index_A = offset_A + m * K + k
                        let index_C = offset_C + m * N + n
                        let index_B = offset_B + k * N + n
                        let a = A.getGradient(index_A) + C.getGradient(index_C) * B.getData(index_B) 
                        A.setGradient(index_A, a) 

            for i in range(A_matrix_size):
                A.setGradient(offset_A + i, A.getGradient(offset_A + i) / N)

        if (B.getRequiresGradient()): 
            for k in range(K):
                for n in range(N): 
                    for m in range(M): 
                        let index_B = offset_B + k * N + n
                        let index_A = offset_A + m * K + k
                        let index_C = offset_C + m * N + n
                        let b = B.getGradient(index_B) + A.getData(index_A) * C.getGradient(index_C)  
                        B.setGradient(index_B, b) 
            
            for i in range(B_matrix_size):
                B.setGradient(offset_B + i, B.getGradient(offset_B + i) / N) 

@always_inline
fn mul_grad_last(C: Tensor, inout A: Tensor, inout B: Tensor):

    let num_dims = A.getNum_dims()
    var A_matrix_size = A.shape[num_dims-2] * A.shape[num_dims-1]
    var B_matrix_size = B.shape[num_dims-2] * B.shape[num_dims-1]
    var C_matrix_size = C.shape[num_dims-2] * C.shape[num_dims-1]
    if(num_dims >= 3):
        A_matrix_size = A.skips[num_dims-3]
        B_matrix_size = B.skips[num_dims-3]
        C_matrix_size = C.skips[num_dims-3]   

    let M = A.shape[num_dims-2]
    let K = B.shape[num_dims-2]  
    let N = B.shape[num_dims-1]

    for s in range(A.getCap() // A_matrix_size):
        let offset_C = s * C_matrix_size
        let offset_B = s * B_matrix_size
        let offset_A = s * A_matrix_size 

        if (A.getRequiresGradient()):
            for m in range(M):
                for k in range(K):
                    for n in range(N):
                        let index_A = offset_A + m * K + k
                        let index_B = offset_B + k * N + n
                        let a = A.getGradient(index_A) + B.getData(index_B) 
                        A.setGradient(index_A, a) 

            for i in range(A_matrix_size):
                A.setGradient(offset_A + i, A.getGradient(offset_A + i) / N)

        if (B.getRequiresGradient()): 
            for k in range(K):
                for n in range(N): 
                    for m in range(M): 
                        let index_B = offset_B + k * N + n
                        let index_A = offset_A + m * K + k
                        let b = B.getGradient(index_B) + A.getData(index_A)
                        B.setGradient(index_B, b) 
            
            for i in range(B_matrix_size):
                B.setGradient(offset_B + i, B.getGradient(offset_B + i) / N) 

fn add_grad(C: Tensor, inout A: Tensor, inout B: Tensor):
    A.setGradient(C.getGradient())
    B.setGradient(C.getGradient())

fn ReLU_grad(B: Tensor, inout A: Tensor):
    A.setGradient(B.getGradient())
    for i in range(A.getCap()):
        let val = A.getData(i)
        if val < 0:
            A.setGradient(i, 0)

fn MSE_grad(C: Tensor, inout A: Tensor, inout B: Tensor): # A: TrueVals, B: Logits
    let num_dims = A.getNum_dims()
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)
    var matrix_size = M*N
    if(num_dims >= 3):
        matrix_size = A.getSkips(num_dims-3)

    for index in range(A.getCap()):
        let grad = Float32(2) * (B.getData(index) - A.getData(index)) / N
        A.setGradient(index, grad) 
        B.setGradient(index, grad) 

fn reshape_grad(B: Tensor, inout A: Tensor):
    A.setGradient(B.getGradient())