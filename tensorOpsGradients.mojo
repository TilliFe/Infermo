from node import Node

@always_inline
fn mul_grad(C: Node, inout A: Node, inout B: Node):

    let num_dims = A.getNum_dims()
    var A_matrix_size = A.shape[num_dims-2] * A.shape[num_dims-1]
    var B_matrix_size = B.shape[num_dims-2] * B.shape[num_dims-1]
    var C_matrix_size = C.shape[num_dims-2] * C.shape[num_dims-1]
    if(num_dims >= 3):
        A_matrix_size = A.skips[num_dims-3]
        B_matrix_size = B.skips[num_dims-3]
        C_matrix_size = C.skips[num_dims-3]   

    let M = A.shape[num_dims-2]
    let K = B.shape[num_dims-2]  # Changed from num_dims-1 because B is now transposed
    let N = A.shape[num_dims-1]  # Changed from C.shape because we're now computing A

    for s in range(A.getCap() // A_matrix_size):
        let offset_C = s * C_matrix_size
        let offset_B = s * B_matrix_size
        let offset_A = s * A_matrix_size  # Changed from C_matrix_size because we're now computing A

        for i in range(M):
            for j in range(N):
                for l in range(K):
                    let index_C = offset_C + i * K + l  # Changed from i * N + j because we're accessing rows of C
                    let index_B = offset_B + l * N + j  # Remains same because we're still accessing rows of B (B_transpose columns)
                    let index_A = offset_A + i * N + j  # Changed from i * K + l because we're computing A
                    let a = A.getGradient(index_A) + C.getGradient(index_C) * B.getData(index_B)  # Changed from c = ... because we're computing A
                    A.setGradient(index_A, a)  # Changed from C.setData because we're computing A

        for i in range(M):
            for j in range(N):
                for l in range(K):
                    let index_A = offset_A + l * N + i  # Changed from i * K + l because we're accessing columns of A (A_transpose rows)
                    let index_C = offset_C + l * N + j  # Changed from i * K + l because we're accessing rows of C
                    let index_B = offset_B + i * N + j  # Changed from i * K + l because we're computing B
                    let b = B.getGradient(index_B) + A.getData(index_A) * C.getGradient(index_C)  # Changed from c = ... because we're computing B
                    B.setGradient(index_B, b)  # Changed from C.setData because we're computing B


fn add_grad(C: Node, inout A: Node, inout B: Node):
    A.setGradient(C.getGradient())
    B.setGradient(C.getGradient())

fn ReLU_grad(B: Node, inout A: Node):
    B.setGradient(0,1)
    B.setGradient(4,1)
    B.setGradient(5,2)
    A.setGradient(B.getGradient())
    for i in range(A.getCap()):
        let val = A.getData(i)
        if val < 0:
            A.setGradient(i, 0)