from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize, sum
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp

from dv import *

alias nelts = simdwidthof[DType.float32]()


fn main():
    var nn = Module()

    var a = Tensor(shape(4))
    a.setDataAll(2)

    var b = Tensor(shape(3,4))
    b.setDataAll(3)

    var c = nn.add(a,b)
    var d = nn.sum(c)

    nn.forward(d)
    nn.backward(d)

    nn.printTensors()

    # a.printData()
    # b.printData()
    # c.printData()


    # nn.forward(c)
    # a.printData()
    # b.printData()
    # c.printData()

    # var a_r = Tensor(shape(3,2))
    # a_r.setDataAll(2)
    # a_r.setData(0,0.01)
    # var a_t = nn.reshape(a_r,shape(1,3,2))
    # var a = nn.transpose(a_t)

    # var b = Tensor(shape(1,3,4))
    # b.setDataAll(3)
    # b.setData(1,0.01)

    # var c = nn.mul(a,b)

    # var d = nn.softmax(c)

    # var e = Tensor(shape(1,2,4))
    # e.setDataAll(0)
    # e.setData(2,1.0)
    # e.setData(7,1.0)

    # var f = nn.CE(e,d)

    # nn.forward(f)
    # nn.backward(f)

    # nn.printTensors()







# # fn main():
# #     print(nelts)
# #     let vec_size = nelts * 2
# #     let buffer = DTypePointer[DType.float32].alloc(vec_size)
# #     for i in range(0,vec_size):
# #         buffer.store(i,Float32(i))

# #     var row_sum = SIMD[DType.float32,1]()
# #     @parameter
# #     fn v_softmax_row[nelts: Int](i: Int):
# #         let sum = buffer.simd_load[nelts](i).reduce_add()
# #         print(sum)
# #         row_sum = row_sum + sum
# #     vectorize[nelts, v_softmax_row](vec_size)

# #     print(row_sum)

# fn main():
#     let size = 10
#     let A = DTypePointer[DType.float32].alloc(size)
#     let B = DTypePointer[DType.float32].alloc(size)
#     memset_zero(A,size)
#     memset_zero(B,size)
#     for i in range(size):
#         A.store(i, Float32(i) * 0.1)

#     @parameter
#     fn v_exp[nelts: Int](i: Int):
#         B.simd_store[nelts](i, ((1.0 + A.simd_load[nelts](i) + A.simd_load[nelts](i).__pow__(2) / 2.0 + A.simd_load[nelts](i).__pow__(3) / 6.0 + A.simd_load[nelts](i).__pow__(4) / 24.0  )))# + A.simd_load[nelts](i).__pow__(5) / 120.0 + A.simd_load[nelts](i).__pow__(6) / 720.0 + A.simd_load[nelts](i).__pow__(7) / 5040.0 + A.simd_load[nelts](i).__pow__(8) / 40320.0 + A.simd_load[nelts](i).__pow__(9) / 362880.0  ))
#     vectorize[nelts, v_exp](size)

#     var row_sum = SIMD[DType.float32,1]()
#     @parameter
#     fn v_softmax_row[nelts: Int](i: Int):
#         row_sum = row_sum + B.simd_load[nelts](i).reduce_add()
#     vectorize[nelts, v_softmax_row](size)

#     @parameter
#     fn v_div[nelts: Int](i: Int):
#         print(row_sum)
#         B.simd_store[nelts](i, B.simd_load[nelts](i) / row_sum )
#     vectorize[nelts, v_div](size)

#     for i in range(size):
#         print_no_newline(A.load(i),",")
#     put_new_line()
#     put_new_line()
#     var sum = SIMD[DType.float32,1]()
#     for i in range(size):
#         sum = sum + B.load(i)
#         print_no_newline(B.load(i),",")
#     put_new_line()
#     print(sum)