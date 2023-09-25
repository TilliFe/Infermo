from Tensor import Tensor

@always_inline
fn mul(inout C: Tensor, A: Tensor, B: Tensor):
    let num_dims = A.getNum_dims()
    var A_matrix_size = A.shape[num_dims-2] * A.shape[num_dims-1]
    var B_matrix_size = B.shape[num_dims-2] * B.shape[num_dims-1]
    var C_matrix_size = C.shape[num_dims-2] * C.shape[num_dims-1]
    if(num_dims >= 3):
        A_matrix_size = A.skips[num_dims-3]
        B_matrix_size = B.skips[num_dims-3]
        C_matrix_size = C.skips[num_dims-3]

    let M = C.shape[num_dims-2]
    let K = A.shape[num_dims-1]
    let N = C.shape[num_dims-1] 

    for s in range(C.getCap() // C_matrix_size):
        let offset_A = s * A_matrix_size
        let offset_B = s * B_matrix_size
        let offset_C = s * C_matrix_size

        for i in range(M):
            for j in range(N):
                for l in range(K):
                    let index_A = offset_A + i * K + l
                    let index_B = offset_B + l * N + j
                    let index_C = offset_C + i * N + j
                    let c = C.getData(index_C) + A.getData(index_A) * B.getData(index_B)
                    C.setData(index_C, c)

fn add(inout C: Tensor, A: Tensor, B: Tensor):
    let num_dims = A.getNum_dims()
    var matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)
    if(num_dims >= 3):
        matrix_size = A.getSkips(num_dims-3)

    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    for s in range(C.getCap() // matrix_size):
        let offset = s * matrix_size
        for i in range(M):
            for j in range(N):
                let index = offset + i * N + j
                C.setData(index, A.getData(index) + B.getData(index))

fn ReLU(inout B: Tensor, A: Tensor):
    for i in range(A.getCap()):
        let val = A.getData(i)
        if(val < 0):
            B.setData(i,0)
        else:
            B.setData(i,val)

fn MSE(inout C: Tensor, A: Tensor, B: Tensor):
    let num_dims = A.getNum_dims()
    var matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)
    if(num_dims >= 3):
        matrix_size = A.getSkips(num_dims-3)

    for index in range(A.getCap()):
        let error = (A.getData(index) - B.getData(index)) * (A.getData(index) - B.getData(index))
        C.setData(0, C.getData(0) + error)
    C.setData(0, C.getData(0) / matrix_size)

fn reshape(inout B: Tensor, A: Tensor):
    return