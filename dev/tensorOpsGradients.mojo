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

        for i in range(M):
            for j in range(K):
                for l in range(N):
                    let index_A = offset_A + i * K + j 
                    let index_B = offset_B + l * N + l 
                    let index_C = offset_C + i * N + l  
                    let a = A.getGradient(index_A) + C.getGradient(index_C) * B.getData(index_B) 
                    A.setGradient(index_A, a / N )  
        
        for i in range(K):
            for j in range(N):
                for l in range(M):
                    let index_A = offset_A + l * K + i 
                    let index_B = offset_B + i * N + j 
                    let index_C = offset_C + l * N + j 
                    let b = B.getGradient(index_B) + A.getData(index_A) * C.getGradient(index_C)  
                    B.setGradient(index_B, b / N) 


fn add_grad(C: Tensor, inout A: Tensor, inout B: Tensor):
    A.setGradient(C.getGradient())
    B.setGradient(C.getGradient())

fn ReLU_grad(B: Tensor, inout A: Tensor):
    A.setGradient(B.getGradient())
    for i in range(A.getCap()):
        let val = A.getData(i)
        if val < 0:
            A.setGradient(i, 0)

fn MSE_grad(C: Tensor, inout A: Tensor, inout B: Tensor):
    let num_dims = A.getNum_dims()
    var matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)
    if(num_dims >= 3):
        matrix_size = A.getSkips(num_dims-3)

    for index in range(A.getCap()):
        let grad = Float32(2) * (A.getData(index) - B.getData(index)) #/ matrix_size
        A.setGradient(index, grad) 
        B.setGradient(index, grad) 

fn reshape_grad(B: Tensor, inout A: Tensor):
    A.setGradient(B.getGradient())