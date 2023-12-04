##################################################################################################################
# CPU COMPUTE KERNELS: unary + binary + reductions (forward + backward)
##################################################################################################################

# Imports
from memory import memset_zero, memcpy
from random import rand, random_si64, seed, randint
from math import (
    max,
    min,
    sqrt,
    abs,
    pow,
    exp2,
    exp,
    log2,
    log,
    cos,
    sin,
    tan,
    asin,
    acos,
    atan,
    cosh,
    sinh,
    tanh,
)
from sys.param_env import env_get_int
from algorithm import vectorize, parallelize
from autograd.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)

from autograd import Node

alias nelts = simdwidthof[DType.float32]()
alias workers = env_get_int["WORKERS", 0]()


###############################################################################################
# BINARY
#   - addition, multiplication, subtraction, division
#   - matrix multiplication
###############################################################################################


# Addition ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
@parameter
fn base_case_add_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_add_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) + b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_add](c_rest)


# @always_inline
fn fw_add(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_add_fw, base_case_add_fw](c, a, b)


# backward
@parameter
fn base_case_add_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_add_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_add_grad_a](c_rest)


@parameter
fn kernel_add_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i) + c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_add_grad_b](c_rest)


# @always_inline
fn bw_add(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_add_bw_a, base_case_add_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_add_bw_b, base_case_add_bw](c, a, b)


# mul ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
@parameter
fn base_case_mul_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_mul_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) * b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_mul](c_rest)


# @always_inline
fn fw_mul(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_mul_fw, base_case_mul_fw](c, a, b)


# backward
@parameter
fn base_case_mul_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_mul_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + b.load_data[nelts](offset_b + i) * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_mul_grad_a](c_rest)


@parameter
fn kernel_mul_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            + a.load_data[nelts](offset_a + i) * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_mul_grad_b](c_rest)


# @always_inline
fn bw_mul(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_mul_bw_a, base_case_mul_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_mul_bw_b, base_case_mul_bw](c, a, b)


# subtraction ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
@parameter
fn base_case_sub_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_sub_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) - b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_sub](c_rest)


# @always_inline
fn fw_sub(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_sub_fw, base_case_sub_fw](c, a, b)


# backward
@parameter
fn base_case_sub_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_sub_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_sub_grad_a](c_rest)


@parameter
fn kernel_sub_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i) - c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_sub_grad_b](c_rest)


# @always_inline
fn bw_sub(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_sub_bw_a, base_case_sub_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_sub_bw_b, base_case_sub_bw](c, a, b)


# division ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
@parameter
fn base_case_div_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_div_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) / b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_div](c_rest)


# @always_inline
fn fw_div(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_div_fw, base_case_div_fw](c, a, b)


# backward
@parameter
fn base_case_div_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_div_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + c.load_grad[nelts](offset_c + i) / b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_div_grad_a](c_rest)


@parameter
fn kernel_div_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            - a.load_data[nelts](offset_a + i)
            * c.load_grad[nelts](offset_c + i)
            / pow(b.load_data[nelts](offset_b + i), 2),
        )

    vectorize[nelts, v_div_grad_b](c_rest)


# @always_inline
fn bw_div(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_div_bw_a, base_case_div_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_div_bw_b, base_case_div_bw](c, a, b)


# Matrix Multplication ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# forward
@parameter
fn base_case_matmul_fw(depth: Int, a: Node, b: Node) -> Bool:
    return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2


@parameter
fn kernel_matmul_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * a.shape_ptr.load().load(
        a.num_dims_ptr.load() - 2
    ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
    let offset_b = b_index * b.shape_ptr.load().load(
        b.num_dims_ptr.load() - 2
    ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
    let offset_c = c_index * c.shape_ptr.load().load(
        c.num_dims_ptr.load() - 2
    ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

    let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
    let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
    let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

    @parameter
    fn calc_row_fw(m: Int):
        for k in range(K):

            @parameter
            fn dot_fw[_nelts: Int](n: Int):
                c.store_data[_nelts](
                    offset_c + m * N + n,
                    c.load_data[_nelts](offset_c + m * N + n)
                    + a.load_data(offset_a + m * K + k)
                    * b.load_data[_nelts](offset_b + k * N + n),
                )

            vectorize[nelts, dot_fw](N)

    parallelize[calc_row_fw](M, workers if workers > 0 else M)


# @always_inline
fn fw_mmul(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_matmul_fw, base_case_matmul_fw](c, a, b)


# backward
@parameter
fn base_case_matmul_bw(depth: Int, a: Node, b: Node) -> Bool:
    return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2


@parameter
fn kernel_matmul_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * a.shape_ptr.load().load(
        a.num_dims_ptr.load() - 2
    ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
    let offset_b = b_index * b.shape_ptr.load().load(
        b.num_dims_ptr.load() - 2
    ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
    let offset_c = c_index * c.shape_ptr.load().load(
        c.num_dims_ptr.load() - 2
    ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

    let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
    let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
    let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

    @parameter
    fn calc_row_1(m: Int):
        for n in range(N):

            @parameter
            fn dot_bw_a[nelts: Int](k: Int):
                let val = a.load_grad(offset_a + m * K + k) + c.load_grad(
                    offset_c + m * N + n
                ) * b.load_data(offset_b + k * N + n)
                a.store_grad(offset_a + m * K + k, val)

            vectorize[1, dot_bw_a](K)

    parallelize[calc_row_1](M, workers if workers > 0 else M)


@parameter
fn kernel_matmul_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * a.shape_ptr.load().load(
        a.num_dims_ptr.load() - 2
    ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
    let offset_b = b_index * b.shape_ptr.load().load(
        b.num_dims_ptr.load() - 2
    ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
    let offset_c = c_index * c.shape_ptr.load().load(
        c.num_dims_ptr.load() - 2
    ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

    let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
    let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
    let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

    @parameter
    fn calc_row_2(k: Int):
        for m in range(M):

            @parameter
            fn dot_bw_b[nelts: Int](n: Int):
                let val = b.load_grad(offset_b + k * N + n) + a.load_data(
                    offset_a + m * K + k
                ) * c.load_grad(offset_c + m * N + n)
                b.store_grad(offset_b + k * N + n, val)

            vectorize[1, dot_bw_b](N)

    parallelize[calc_row_2](K, workers if workers > 0 else K)


# @always_inline
fn bw_mmul(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_matmul_bw_a, base_case_matmul_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_matmul_bw_b, base_case_matmul_bw](c, a, b)


###############################################################################################
# UNARY
#   - trigonometric functions
#   - activations
###############################################################################################


# sqrt ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_sqrt(node: Node, parent1: Node):
    @parameter
    fn v_sqrt[_nelts: Int](i: Int):
        node.store_data[_nelts](i, sqrt(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_sqrt](node.load_cap())


# backward
fn bw_sqrt(node: Node, parent1: Node):
    @parameter
    fn v_sqrt_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            / (Float32(2.0) * sqrt(parent1.load_data[_nelts](i))),
        )

    vectorize[nelts, v_sqrt_bw](node.load_cap())


# abs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_abs(node: Node, parent1: Node):
    @parameter
    fn v_abs[_nelts: Int](i: Int):
        let zeros = SIMD[DType.float32, _nelts]()
        node.store_data[_nelts](
            i,
            (parent1.load_data[_nelts](i) >= zeros).cast[DType.float32]()
            * parent1.load_data[_nelts](i)
            + (parent1.load_data[_nelts](i) < zeros).cast[DType.float32]()
            * (-parent1.load_data[_nelts](i)),
        )

    vectorize[nelts, v_abs](node.load_cap())


# backward
fn bw_abs(node: Node, parent1: Node):
    @parameter
    fn v_abs_bw[_nelts: Int](i: Int):
        let zeros = SIMD[DType.float32, _nelts]()
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + (
                Float32(2.0)
                * (parent1.load_data[_nelts](i) >= zeros).cast[DType.float32]()
                - Float32(1.0)
            )
            * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_abs_bw](node.load_cap())


# exp2 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_exp2(node: Node, parent1: Node):
    @parameter
    fn v_exp2[_nelts: Int](i: Int):
        node.store_data[_nelts](i, exp2(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_exp2](node.load_cap())


# backward
fn bw_exp2(node: Node, parent1: Node):
    @parameter
    fn v_exp2_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * node.load_data[_nelts](i) * log(Float32(2.0)),
        )

    vectorize[nelts, v_exp2_bw](node.load_cap())


# exp ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_exp(node: Node, parent1: Node):
    @parameter
    fn v_exp[_nelts: Int](i: Int):
        node.store_data[_nelts](i, exp(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_exp](node.load_cap())


# backward
fn bw_exp(node: Node, parent1: Node):
    @parameter
    fn v_exp_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * node.load_data[_nelts](i),
        )

    vectorize[nelts, v_exp_bw](node.load_cap())


# log2 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_log2(node: Node, parent1: Node):
    @parameter
    fn v_log2[_nelts: Int](i: Int):
        node.store_data[_nelts](i, log2(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_log2](node.load_cap())


# backward
fn bw_log2(node: Node, parent1: Node):
    @parameter
    fn v_log2_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            / (parent1.load_data[_nelts](i) * log(Float32(2.0))),
        )

    vectorize[nelts, v_log2_bw](node.load_cap())


# log ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_log(node: Node, parent1: Node):
    @parameter
    fn v_log[_nelts: Int](i: Int):
        node.store_data[_nelts](i, log(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_log](node.load_cap())


# backward
fn bw_log(node: Node, parent1: Node):
    @parameter
    fn v_log_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) / parent1.load_data[_nelts](i),
        )

    vectorize[nelts, v_log_bw](node.load_cap())


# pow ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
# base case
@parameter
fn base_case_pow_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


# kernel
@parameter
fn kernel_pow_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            pow(a.load_data[nelts](offset_a + i), b.load_data[nelts](offset_b + i)),
        )

    vectorize[nelts, v_pow](c_rest)


# @always_inline
fn fw_pow(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_pow_fw, base_case_pow_fw](c, a, b)


# backward
# base case
@parameter
fn base_case_pow_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


# kernel
@parameter
fn kernel_pow_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow_bw_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + b.load_data[nelts](offset_b + i)
            * pow(
                a.load_data[nelts](offset_a + i),
                b.load_data[nelts](offset_b + i) - Float32(1.0),
            )
            * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_pow_bw_a](c_rest)


@parameter
fn kernel_pow_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow_bw_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            + c.load_data[nelts](offset_c + i)
            * log(a.load_data[nelts](offset_a + i))
            * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_pow_bw_b](c_rest)


# @always_inline
fn bw_pow(c: Node, a: Node, b: Node):
    recursive_broadcast_bw[kernel_pow_bw_a, base_case_pow_bw](c, a, b)
    recursive_broadcast_bw[kernel_pow_bw_b, base_case_pow_bw](c, a, b)


# sin ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_sin(node: Node, parent1: Node):
    @parameter
    fn v_sin[_nelts: Int](i: Int):
        node.store_data[_nelts](i, sin(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_sin](node.load_cap())


# backward
fn bw_sin(node: Node, parent1: Node):
    @parameter
    fn v_sin_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + cos(parent1.load_data[_nelts](i)) * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_sin_bw](node.load_cap())


# cos ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_cos(node: Node, parent1: Node):
    @parameter
    fn v_cos[_nelts: Int](i: Int):
        node.store_data[_nelts](i, cos(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_cos](node.load_cap())


# backward
fn bw_cos(node: Node, parent1: Node):
    @parameter
    fn v_cos_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            - sin(parent1.load_data[_nelts](i)) * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_cos_bw](node.load_cap())


# tan ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_tan(node: Node, parent1: Node):
    @parameter
    fn v_tan[_nelts: Int](i: Int):
        node.store_data[_nelts](i, tan(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_tan](node.load_cap())


# backward
fn bw_tan(node: Node, parent1: Node):
    @parameter
    fn v_tan_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) / pow(cos(parent1.load_data[_nelts](i)), 2),
        )

    vectorize[nelts, v_tan_bw](node.load_cap())


# asin ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_asin(node: Node, parent1: Node):
    @parameter
    fn v_asin[_nelts: Int](i: Int):
        node.store_data[_nelts](i, asin(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_asin](node.load_cap())


# backward
fn bw_asin(node: Node, parent1: Node):
    @parameter
    fn v_asin_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            / sqrt(Float32(1.0) - pow(parent1.load_data[_nelts](i), 2)),
        )

    vectorize[nelts, v_asin_bw](node.load_cap())


# acos ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_acos(node: Node, parent1: Node):
    @parameter
    fn v_acos[_nelts: Int](i: Int):
        node.store_data[_nelts](i, acos(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_acos](node.load_cap())


# backward
fn bw_acos(node: Node, parent1: Node):
    @parameter
    fn v_acos_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            - node.load_grad[_nelts](i)
            / sqrt(Float32(1.0) - pow(parent1.load_data[_nelts](i), 2)),
        )

    vectorize[nelts, v_acos_bw](node.load_cap())


# atan ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_atan(node: Node, parent1: Node):
    @parameter
    fn v_atan[_nelts: Int](i: Int):
        node.store_data[_nelts](i, atan(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_atan](node.load_cap())


# backward
fn bw_atan(node: Node, parent1: Node):
    @parameter
    fn v_atan_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            / (Float32(1.0) + pow(parent1.load_data[_nelts](i), 2)),
        )

    vectorize[nelts, v_atan_bw](node.load_cap())


# sinh ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_sinh(node: Node, parent1: Node):
    @parameter
    fn v_sinh[_nelts: Int](i: Int):
        node.store_data[_nelts](i, sinh(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_sinh](node.load_cap())


# backward
fn bw_sinh(node: Node, parent1: Node):
    @parameter
    fn v_sinh_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * cosh(parent1.load_data[_nelts](i)),
        )

    vectorize[nelts, v_sinh_bw](node.load_cap())


# cosh ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_cosh(node: Node, parent1: Node):
    @parameter
    fn v_cosh[_nelts: Int](i: Int):
        node.store_data[_nelts](i, cosh(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_cosh](node.load_cap())


# backward
fn bw_cosh(node: Node, parent1: Node):
    @parameter
    fn v_cosh_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * sinh(parent1.load_data[_nelts](i)),
        )

    vectorize[nelts, v_cosh_bw](node.load_cap())


# tanh ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_tanh(node: Node, parent1: Node):
    @parameter
    fn v_tanh[_nelts: Int](i: Int):
        node.store_data[_nelts](i, tanh(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_tanh](node.load_cap())


# backward
fn bw_tanh(node: Node, parent1: Node):
    @parameter
    fn v_tanh_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            * (Float32(1.0) - pow(node.load_data[_nelts](i), 2)),
        )

    vectorize[nelts, v_tanh_bw](node.load_cap())


# relu ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_relu(node: Node, parent1: Node):
    @parameter
    fn v_relu[_nelts: Int](i: Int):
        let zeros = SIMD[DType.float32, _nelts]()
        node.store_data[_nelts](
            i,
            abs(
                (parent1.load_data[_nelts](i) > zeros).cast[DType.float32]()
                * parent1.load_data[_nelts](i)
            ),
        )

    vectorize[nelts, v_relu](node.load_cap())


# backward
fn bw_relu(node: Node, parent1: Node):
    @parameter
    fn v_relu_bw[_nelts: Int](i: Int):
        let zeros = SIMD[DType.float32, _nelts]()
        parent1.store_grad[_nelts](
            i,
            abs((parent1.load_data[_nelts](i) > zeros).cast[DType.float32]())
            * node.load_grad[_nelts](i)
            + parent1.load_grad[_nelts](i),
        )

    vectorize[nelts, v_relu_bw](node.load_cap())


# copy ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
fn fw_copy(node: Node, parent1: Node):
    pass


# backward
fn bw_copy(node: Node, parent1: Node):
    pass


###############################################################################################
# REDUCTIONS
#   - sum
#   - softmax, mse, ce, ...
#   - transpose, reshape
###############################################################################################


# sum ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
# @always_inline
fn fw_sum(node: Node, parent1: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_sum[nelts: Int](i: Int):
        sum += parent1.load_data[nelts](i).reduce_add()

    vectorize[nelts, v_sum](parent1.load_cap())
    node.store_data(0, sum)


# backward
# @always_inline
fn bw_sum(node: Node, parent1: Node):
    @parameter
    fn v_sum_bw[nelts: Int](i: Int):
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + node.load_grad(0))

    vectorize[nelts, v_sum_bw](parent1.load_cap())


# softmax ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
# @always_inline
fn fw_softmax(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    let epsilon = Float32(1e-8)
    for s in range(node.cap_ptr.load() // N):
        let offset = s * N
        var max_el = Float32(0.0)

        @parameter
        fn v_max[nelts: Int](i: Int):
            let temp = parent1.load_data[nelts](offset + i).reduce_max()
            max_el = max(max_el, temp)

        vectorize[nelts, v_max](N)
        var sum = Float32(0.0)

        @parameter
        fn v_exp[nelts: Int](i: Int):
            let temp = exp(parent1.load_data[nelts](offset + i) - max_el)
            node.store_data[nelts](offset + i, temp)
            sum += temp.reduce_add()

        vectorize[nelts, v_exp](N)

        @parameter
        fn v_div[nelts: Int](i: Int):
            node.store_data[nelts](offset + i, node.load_data[nelts](offset + i) / sum)

        vectorize[nelts, v_div](N)


# backward
# @always_inline
fn bw_softmax(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    let epsilon = Float32(1e-8)
    for s in range(node.cap_ptr.load() // N):
        let offset = s * N

        @parameter
        fn v_softmax_bw_outer[nelts: Int](j: Int):
            var grad = Float32(0.0)
            var grad2 = Float32(0.0)

            @parameter
            fn v_softmax_bw[nelts: Int](i: Int):
                if i == j:
                    let temp = node.load_data[nelts](offset + j)
                    grad += (
                        node.load_grad[nelts](offset + i)
                        * (temp * (Float32(1.0) - temp))
                    ).reduce_add()
                else:
                    grad += (
                        node.load_grad[nelts](offset + i)
                        * node.load_data[nelts](offset + i)
                        * node.load_data[nelts](offset + j)
                        * -1
                    ).reduce_add()

            vectorize[nelts, v_softmax_bw](N)
            parent1.store_grad[nelts](
                offset + j, parent1.load_grad[nelts](offset + j) + grad
            )

        vectorize[nelts, v_softmax_bw_outer](N)


# mse ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
# @always_inline
fn fw_mse(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_mse[nelts: Int](i: Int):
        let error = (parent1.load_data[nelts](i) - parent2.load_data[nelts](i)) * (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        )
        sum += error.reduce_add()

    vectorize[nelts, v_mse](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


# backward
# @always_inline
fn bw_mse(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_mse_bw[nelts: Int](i: Int):
        let grad = -Float32(2.0) * (
            parent2.load_data[nelts](i) - parent1.load_data[nelts](i)
        ) / Float32(parent1.load_cap())
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_mse_bw](parent1.load_cap())


# ce ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
# @always_inline
fn fw_ce(c: Node, a: Node, b: Node):
    let num_dims = a.shape_ptr.load().len.load()
    let N = a.shape_ptr.load().load(num_dims - 1)
    let epsilon = Float32(1e-8)

    @parameter
    fn v_ce[nelts: Int](index: Int):
        let error = -a.load_data[nelts](index) * log(
            b.load_data[nelts](index) + epsilon
        )
        c.store_data(0, c.load_data(0) + error.reduce_add())

    vectorize[nelts, v_ce](a.cap_ptr.load())
    c.store_data(0, c.load_data(0) / (Float32(a.cap_ptr.load()) / Float32(N)))


# backward
# @always_inline
fn bw_ce(c: Node, a: Node, b: Node):
    let num_dims = a.num_dims_ptr.load()
    let N = a.shape_ptr.load().load(num_dims - 1)

    for index in range(a.cap_ptr.load()):
        let grad_a = -log(b.load_data(index))
        a.store_grad(
            index,
            a.load_grad(index) + grad_a / (Float32(a.cap_ptr.load()) / Float32(N)),
        )
    for index in range(b.cap_ptr.load()):
        let grad_b = -a.load_data(index) / (b.load_data(index))
        b.store_grad(
            index,
            b.load_grad(index) + grad_b / (Float32(a.cap_ptr.load()) / Float32(N)),
        )


# reshape ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# forward
# @always_inline
fn fw_reshape(node: Node, parent1: Node):
    for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
        let offset = s * parent1.cap_ptr.load()

        @parameter
        fn v_reshape[nelts: Int](i: Int):
            node.store_data[nelts](i, parent1.load_data[nelts](i))

        vectorize[nelts, v_reshape](parent1.cap_ptr.load())


# backward
# @always_inline
fn bw_reshape(node: Node, parent1: Node):
    for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
        let offset = s * parent1.cap_ptr.load()

        @parameter
        fn v_reshape[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i, parent1.load_grad[nelts](i) + node.load_grad[nelts](i)
            )

        vectorize[nelts, v_reshape](parent1.cap_ptr.load())


# # transpose ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # forward
# @always_inline
fn fw_transpose(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let M = parent1.shape_ptr.load().load(num_dims - 2)
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transpose[nelts: Int](j: Int):
                node.store_data[nelts](
                    offset + j * M + i, parent1.load_data[nelts](offset + i * N + j)
                )

            vectorize[nelts, v_transpose](N)


# backward
# @always_inline
fn bw_transpose(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let M = parent1.shape_ptr.load().load(num_dims - 2)
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transpose_bw[nelts: Int](j: Int):
                parent1.store_grad[nelts](
                    offset + j * M + i,
                    parent1.load_grad[nelts](offset + j * M + i)
                    + node.load_grad[nelts](offset + i * N + j),
                )

            vectorize[nelts, v_transpose_bw](N)


# Function to calculate the index in the 1D buffer
fn index(
    n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
) -> Int:
    return n * (num_channels * height * width) + c * (height * width) + h * width + w


@always_inline
fn conv_2d(c: Node, a: Node, b: Node):
    let padding = c.other_params_ptr.load().load(0)
    let stride = c.other_params_ptr.load().load(1)

    # Loop over each image in the batch
    @parameter
    fn batch_loop(i: Int):
        for j in range(b.shape_ptr.load().load(0)):
            for x in range(c.shape_ptr.load().load(2)):
                for y in range(c.shape_ptr.load().load(3)):
                    var patch_sum: Float32 = 0.0
                    # apply the convolution operation - vectorize?
                    for k in range(a.shape_ptr.load().load(1)):
                        for dx in range(b.shape_ptr.load().load(2)):

                            @parameter
                            fn inner_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    let a_index = index(
                                        i,
                                        k,
                                        ix,
                                        iy,
                                        a.shape_ptr.load().load(1),
                                        a.shape_ptr.load().load(2),
                                        a.shape_ptr.load().load(3),
                                    )
                                    let b_index = index(
                                        j,
                                        k,
                                        dx,
                                        dy,
                                        a.shape_ptr.load().load(1),
                                        b.shape_ptr.load().load(2),
                                        b.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        a.load_data[_nelts](a_index)
                                        * b.load_data[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, inner_loop](b.shape_ptr.load().load(3))
                    let c_index = index(
                        i,
                        j,
                        x,
                        y,
                        b.shape_ptr.load().load(0),
                        c.shape_ptr.load().load(2),
                        c.shape_ptr.load().load(3),
                    )
                    c.store_data(c_index, patch_sum)

    parallelize[batch_loop](
        a.shape_ptr.load().load(0),
        workers if workers > 0 else a.shape_ptr.load().load(0),
    )


# @always_inline
# fn conv_2d_grad(c: Tensor, inout a: Tensor, inout b: Tensor):

#     let padding = c.other_params.load(0)
#     let stride = c.other_params.load(1)

#     # Function to calculate the index in the 1D buffer
#     fn index(n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int) -> Int:
#         return n*(num_channels*height*width) + c*(height*width) + h*width + w

#     # ##### compute the gradietn of the Kernel (right tensor) ########################################
#     for i in range(a.shape[1]): # in_channels
#         for j in range(b.shape[0]): # out_channels
#             for x in range(b.shape[2]): # kernel_width
#                 for y in range(b.shape[3]): # kernel_height
#                     var patch_sum: Float32 = 0.0
#                     for b in range(a.shape[0]):
#                         for dx in range(c.shape[2]):

#                             @parameter
#                             fn inner_loop[_nelts: Int](dy: Int):
#                                 # calculate input indices with consideration for padding and stride
#                                 let ix = x * stride - padding + dx
#                                 let iy = y * stride - padding + dy
#                                 # Skip if index is out of bounds (this is 'zero' padding)
#                                 if not (
#                                     ix < 0
#                                     or iy < 0
#                                     or ix >= a.shape[2]
#                                     or iy >= a.shape[3]
#                                 ):
#                                     let a_index = index(b, i, ix, iy, a.shape[1], a.shape[2], a.shape[3])
#                                     let c_grad_index = index(b, j, dx, dy, c.shape[1], c.shape[2], c.shape[3])
#                                     # add to patch sum
#                                     patch_sum += (
#                                         a.data.simd_load[_nelts](a_index)
#                                         * c.grad.simd_load[_nelts](c_grad_index)
#                                     ).reduce_add()

#                             vectorize[nelts, inner_loop](c.shape[3])
#                     let b_grad_index = index(i, j, x, y, b.shape[0], b.shape[2], b.shape[3])
#                     b.grad.store(b_grad_index, patch_sum)

#     # ##### compute the gradient of the Input (left tensor) ############################################
#     @parameter
#     fn batch_loop(p: Int):  # batch_size
#         for j in range(a.shape[1]): # in_channels
#             for i in range(b.shape[0]): # out_channels
#                 for x in range(a.shape[2]):
#                     for y in range(a.shape[3]):
#                         var patch_sum : Float32 = 0.0
#                         for dx in range(b.shape[2]):

#                             @parameter
#                             fn dy_loop[_nelts: Int](dy: Int):
#                                 let ix = x * stride - dx + padding
#                                 let iy = y * stride - dy + padding
#                                 # Skip if index is out of bounds (this is 'zero' padding)
#                                 if not (
#                                     ix < 0
#                                     or iy < 0
#                                     or ix >= c.shape[2]
#                                     or iy >= c.shape[3]
#                                 ):
#                                     let c_grad_index = index(
#                                         p,
#                                         i,
#                                         ix,
#                                         iy,
#                                         c.shape[1],
#                                         c.shape[2],
#                                         c.shape[3],
#                                     )
#                                     let b_index = index(
#                                         i,
#                                         j,
#                                         b.shape[2] - dx - 1,
#                                         b.shape[3] - dy - 1,
#                                         b.shape[1],
#                                         b.shape[2],
#                                         b.shape[3],
#                                     )
#                                     patch_sum += (
#                                         c.grad.simd_load[_nelts](c_grad_index)
#                                         * c.data.simd_load[_nelts](b_index)
#                                     ).reduce_add()

#                             vectorize[nelts, dy_loop](b.shape[3])
#                         let a_grad_index = index(p,j,x,y,a.shape[1],a.shape[2],a.shape[3])
#                         a.grad.store( a_grad_index, a.grad.load(a_grad_index) + patch_sum)

#     parallelize[batch_loop](a.shape[0], workers if workers > 0 else a.shape[0])


# updated ibackward implementation
fn bw_conv_2d(c: Node, a: Node, b: Node):
    let padding = c.other_params_ptr.load().load(0)
    let stride = c.other_params_ptr.load().load(1)

    #  compute the gradient of the Kernel (right tensor) ########################################
    for i in range(a.shape_ptr.load().load(1)):
        for j in range(b.shape_ptr.load().load(0)):
            for x in range(b.shape_ptr.load().load(2)):
                for y in range(b.shape_ptr.load().load(3)):
                    var patch_sum: Float32 = 0.0
                    for b in range(a.shape_ptr.load().load(0)):
                        for dx in range(c.shape_ptr.load().load(2)):
                            for dy in range(c.shape_ptr.load().load(3)):
                                # calculate input indices with consideration for padding and stride
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                # Skip if index is out of bounds (this is 'zero' padding)
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    let a_index = index(
                                        b,
                                        i,
                                        ix,
                                        iy,
                                        a.shape_ptr.load().load(1),
                                        a.shape_ptr.load().load(2),
                                        a.shape_ptr.load().load(3),
                                    )
                                    let c_grad_index = index(
                                        b,
                                        j,
                                        dx,
                                        dy,
                                        c.shape_ptr.load().load(1),
                                        c.shape_ptr.load().load(2),
                                        c.shape_ptr.load().load(3),
                                    )
                                    # add to patch sum
                                    patch_sum += (
                                        a.load_data(a_index) * c.load_grad(c_grad_index)
                                    ).reduce_add()
                    let b_grad_index = index(
                        i,
                        j,
                        x,
                        y,
                        b.shape_ptr.load().load(0),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    b.store_grad(b_grad_index, patch_sum)

    #  compute the gradient of the Input (left tensor) ############################################
    @parameter
    fn batch_loop(p: Int):  # batch_size
        for j in range(a.shape_ptr.load().load(1)):  # in_channels
            for i in range(b.shape_ptr.load().load(0)):  # out_channels
                for x in range(a.shape_ptr.load().load(2)):
                    for y in range(a.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        for dx in range(b.shape_ptr.load().load(2)):

                            @parameter
                            fn dy_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - dx + padding
                                let iy = y * stride - dy + padding
                                # Skip if index is out of bounds (this is 'zero' padding)
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= c.shape_ptr.load().load(2)
                                    or iy >= c.shape_ptr.load().load(3)
                                ):
                                    let c_grad_index = index(
                                        p,
                                        i,
                                        ix,
                                        iy,
                                        c.shape_ptr.load().load(1),
                                        c.shape_ptr.load().load(2),
                                        c.shape_ptr.load().load(3),
                                    )
                                    let b_index = index(
                                        i,
                                        j,
                                        b.shape_ptr.load().load(2) - dx - 1,
                                        b.shape_ptr.load().load(3) - dy - 1,
                                        b.shape_ptr.load().load(1),
                                        b.shape_ptr.load().load(2),
                                        b.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        c.load_grad[_nelts](c_grad_index)
                                        * c.load_data[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, dy_loop](b.shape_ptr.load().load(3))
                        let a_grad_index = index(
                            p,
                            j,
                            x,
                            y,
                            a.shape_ptr.load().load(1),
                            a.shape_ptr.load().load(2),
                            a.shape_ptr.load().load(3),
                        )
                        a.store_grad(
                            a_grad_index, a.load_grad(a_grad_index) + patch_sum
                        )

    parallelize[batch_loop](
        a.shape_ptr.load().load(0),
        workers if workers > 0 else a.shape_ptr.load().load(0),
    )


fn max_pool_2d(b: Node, a: Node):
    let padding = b.other_params_ptr.load().load(0)
    let stride = b.other_params_ptr.load().load(1)
    let kernel_width = b.other_params_ptr.load().load(2)
    let kernel_height = b.other_params_ptr.load().load(3)

    for p in range(a.shape_ptr.load().load(0)):  # batch_size
        for i in range(a.shape_ptr.load().load(1)):  # in_channels
            for x in range(
                0, a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding, stride
            ):  # width
                for y in range(
                    0,
                    a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                    stride,
                ):  # height
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    # vectorize ?
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= a.shape_ptr.load().load(2)
                                or iy >= a.shape_ptr.load().load(3)
                            ):
                                continue
                            let idx = index(
                                p,
                                i,
                                ix,
                                iy,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            let entry = a.load_data(idx)
                            if entry > max_val:
                                max_val = entry
                                arg_max = idx
                    let idx = index(
                        p,
                        i,
                        (x) // stride,
                        (y) // stride,
                        b.shape_ptr.load().load(1),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    b.store_data(idx, max_val)


# @always_inline
# fn max_pool_2d_grad(b: Tensor, inout a: Tensor):

#     let padding = b.other_params.load(0)
#     let stride = b.other_params.load(1)
#     let kernel_width = b.other_params.load(2)
#     let kernel_height = b.other_params.load(3)

#     # Function to calculate the index in the 1D buffer
#     fn index(n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int) -> Int:
#         return n*(num_channels*height*width) + c*(height*width) + h*width + w

#     for p in range(a.shape[0]): # batch_size
#         for i in range(a.shape[1]): # in_channels
#             for x in range(0,a.shape[2]-kernel_width+1 + 2*padding,stride): # width
#                 for y in range(0,a.shape[3]-kernel_height+1 + 2*padding,stride): # height
#                     var arg_max: Int = 0
#                     var max_val: Float32 = -1000000.0
#                     for dx in range(kernel_width):
#                         for dy in range(kernel_height):
#                             let ix = x - padding + dx
#                             let iy = y - padding + dy
#                             if ix < 0 or iy < 0 or ix >= a.shape[2] or iy >= a.shape[3]:
#                                 continue
#                             let idx = index(p,i,ix,iy,a.shape[1],a.shape[2],a.shape[3])
#                             let entry = a.data.load(idx)
#                             if(entry > max_val):
#                                 max_val = entry
#                                 arg_max = idx
#                     let b_grad_idx = index(p,i,(x)//stride,(y)//stride,b.shape[1],b.shape[2],b.shape[3])
#                     a.grad.store(arg_max, a.grad.load(arg_max) + b.grad.load(b_grad_idx))


# updated version
fn bw_max_pool_2d(b: Node, a: Node):
    let padding = b.other_params_ptr.load().load(0)
    let stride = b.other_params_ptr.load().load(1)
    let kernel_width = b.other_params_ptr.load().load(2)
    let kernel_height = b.other_params_ptr.load().load(3)

    for p in range(a.shape_ptr.load().load(0)):  # batch_size
        for i in range(a.shape_ptr.load().load(1)):  # in_channels
            for x in range(
                0, a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding, stride
            ):  # width
                for y in range(
                    0,
                    a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                    stride,
                ):  # height
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= a.shape_ptr.load().load(2)
                                or iy >= a.shape_ptr.load().load(3)
                            ):
                                continue
                            let idx = index(
                                p,
                                i,
                                ix,
                                iy,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            let entry = a.load_data(idx)
                            if entry > max_val:
                                max_val = entry
                                arg_max = idx
                    let b_grad_idx = index(
                        p,
                        i,
                        (x) // stride,
                        (y) // stride,
                        b.shape_ptr.load().load(1),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    a.store_grad(
                        arg_max, a.load_grad(arg_max) + b.load_grad(b_grad_idx)
                    )
