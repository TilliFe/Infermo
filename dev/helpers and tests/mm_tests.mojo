from benchmark import Benchmark
from sys.intrinsics import strided_load
from utils.list import VariadicList
from math import div_ceil, min
from memory import memset_zero
from memory.unsafe import DTypePointer
from random import rand, random_float64
from sys.info import simdwidthof
from runtime.llcl import Runtime


struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        # rand(self.data, rows*cols)
        memset_zero(self.data, rows*cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)

    @always_inline
    fn print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print_no_newline(self.data.load(i), ",")
            put_new_line()
        put_new_line()
        

    @always_inline
    fn fill(self, val: Float32):
        for i in range(self.rows * self.cols):
            self.data.store(i, val)



# compute C = A * B [MxN] = [MxK] [KxN]
fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows): # M
        for k in range(A.cols): # K
            for n in range(C.cols): # N
                C[m, n] += A[m, k] * B[k, n]


#compute  A = C * B_trans [MxK] = [MxN] * [NxK]
fn matmulCB_naive(A: Matrix, C: Matrix, B: Matrix):
    for m in range(C.rows): # M
        for n in range(C.cols): # N
            for k in range(A.cols): # K
                A[m, k] += C[m, n] * B[k, n]

#compute  B = A_trans * C [KxN] = [KxM] * [M,N]
fn matmulAC_naive(B: Matrix, A: Matrix, C: Matrix):
    for m in range(C.rows): # M
        for k in range(A.cols): # K
            for n in range(C.cols): # N
                B[k, n] += A[m, k] * C[m, n]


###################################################


fn test1():
    print("\ntest1")
    let M = 1
    let K = 2
    let N = 3

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    A.fill(3)
    matmulAC_naive(A,C,B)
    A.print()
    C.print()
    B.print()

fn test2():
    print("\ntest2")
    let M = 2
    let K = 3
    let N = 1

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    A.fill(3)
    matmulAC_naive(A,C,B)
    A.print()
    C.print()
    B.print()

fn test3():
    print("\ntest3")
    let M = 1
    let K = 3
    let N = 2

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    A.fill(3)
    matmulAC_naive(A,C,B)
    A.print()
    C.print()
    B.print()

fn test4():
    print("\ntest4")
    let M = 2
    let K = 1
    let N = 3

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    A.fill(3)
    matmulAC_naive(A,C,B)
    A.print()
    C.print()
    B.print()

##################################################


fn test5():
    print("\ntest5")
    let M = 1
    let K = 2
    let N = 3

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    B.fill(3)
    matmulCB_naive(A,C,B)
    B.print()
    C.print()
    A.print()

fn test6():
    print("\ntest6")
    let M = 2
    let K = 1
    let N = 3

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    B.fill(3)
    matmulCB_naive(A,C,B)
    B.print()
    C.print()
    A.print()

fn test7():
    print("\ntest7")
    let M = 1
    let K = 3
    let N = 2

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    B.fill(3)
    matmulCB_naive(A,C,B)
    B.print()
    C.print()
    A.print()

fn test8():
    print("\ntest8")
    let M = 1
    let K = 2
    let N = 3

    let A = Matrix(M,K)
    let B = Matrix(K,N)
    let C = Matrix(M,N)

    C.fill(2)
    B.fill(3)
    matmulCB_naive(A,C,B)
    B.print()
    C.print()
    A.print()

fn main():
    # test1()
    # test2()
    # test3()
    # test4()

    test5()
    test6()
    test7()
    test8()