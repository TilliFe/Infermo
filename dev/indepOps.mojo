from Tensor import Tensor

@always_inline
fn oneHot(inout A: Tensor) -> Tensor:

    let num_dims = A.getNum_dims()
    var new_shape = DynamicVector[Int]()

    for i in range(num_dims-1):
        new_shape.push_back(A.getShape(i))
    new_shape.push_back(1)

    let B = Tensor(new_shape)

    let A_matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    for s in range(A.getCap() // A_matrix_size):
        let A_offset = s * A_matrix_size
        let B_offset = s * M
        for m in range(M):
            var max: Float32 = 0
            var argMax: Float32 = 0
            var idx: Float32 = 0
            for n in range(N):
                let index = A_offset + m*N+n
                if(A.getData(index) > max):
                    max = A.getData(index)
                    argMax = idx
                idx += 1
            B.setData(B_offset + m, argMax)
    return B

# compute similarity accuracy between tweo tensors along the last dimension
fn accuracy(logits: Tensor, trueVals: Tensor) -> Float32:
    var avgAcc: Float32 = 0
    let N = trueVals.getShape(trueVals.num_dims-1)
    for i in range(trueVals.getCap()):
        if(trueVals.getData(i) == logits.getData(i)):
            avgAcc += 1
    return avgAcc / (trueVals.cap - N)


