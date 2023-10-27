from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from time import now
from algorithm import vectorize, parallelize, unroll
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp, min, max, abs

from ..graph.tensor import Tensor
from ..operators.forward import (
    matmul,
    conv_2d,
    max_pool_2d,
    sum,
    softmax,
    mse,
    ce,
    reshape,
    transpose,
    mean,
    variance,
    std,
    e_mul,
    e_add,
    e_sub,
    e_div,
    e_sqrt,
    e_abs,
    e_pow,
    e_pow_all,
    e_exp2,
    e_exp,
    e_log2,
    e_log,
    e_sin,
    e_cos,
    e_tan,
    e_asin,
    e_acos,
    e_atan,
    e_sinh,
    e_cosh,
    e_tanh,
    e_relu,
    e_copy,
)
from ..operators.backward import (
    matmul_grad,
    conv_2d_grad,
    max_pool_2d_grad,
    sum_grad,
    softmax_grad,
    mse_grad,
    ce_grad,
    reshape_grad,
    transpose_grad,
    mean_grad,
    variance_grad,
    std_grad,
    e_mul_grad,
    e_add_grad,
    e_sub_grad,
    e_div_grad,
    e_sqrt_grad,
    e_abs_grad,
    e_pow_grad,
    e_pow_all_grad,
    e_exp2_grad,
    e_exp_grad,
    e_log2_grad,
    e_log_grad,
    e_sin_grad,
    e_cos_grad,
    e_tan_grad,
    e_asin_grad,
    e_acos_grad,
    e_atan_grad,
    e_sinh_grad,
    e_cosh_grad,
    e_tanh_grad,
    e_relu_grad,
    e_copy_grad,
)
from ..helpers.shape import shape, Vec

alias nelts = simdwidthof[DType.float32]()


struct Module:
    var nodes: DynamicVector[Tensor]
    var num_static_nodes: Int
    var num_dynamic_nodes: Int
    var backward_tape: DynamicVector[Int]
    var backward_tape_nodes_dynamic: DynamicVector[Bool]
    var num_passes: Int
    var forward_durations: DynamicVector[Int]
    var backward_durations: DynamicVector[Int]
    var operation_labels: DynamicVector[Pointer[StringLiteral]]

    fn __init__(inout self):
        self.nodes = DynamicVector[Tensor]()
        self.num_static_nodes = 0
        self.num_dynamic_nodes = 0
        self.backward_tape = DynamicVector[Int]()
        self.backward_tape_nodes_dynamic = DynamicVector[Bool]()
        self.num_passes = 0
        self.forward_durations = DynamicVector[Int](34)
        self.backward_durations = DynamicVector[Int](34)
        self.operation_labels = DynamicVector[Pointer[StringLiteral]](34)
        var operation_labels = [
            "matmul:        ",
            "conv2d:        ",
            "max_pool_2d:   ",
            "sum:           ",
            "softmax:       ",
            "mse:           ",
            "ce:            ",
            "reshape:       ",
            "transpose:     ",
            "mean:          ",
            "variance:      ",
            "std            ",
            "mul:           ",
            "add:           ",
            "sub:           ",
            "div:           ",
            "sqrt:          ",
            "abs:           ",
            "pow            ",
            "pow            ",
            "exp2:          ",
            "exp:           ",
            "log2:          ",
            "log:           ",
            "sin:           ",
            "cos:           ",
            "tan:           ",
            "asin:          ",
            "acos:          ",
            "atan:          ",
            "sinh:          ",
            "cosh:          ",
            "tanh:          ",
            "relu:          ",
            "copy:          ",
        ]

        @parameter
        fn loop[idx: Int]():
            self.forward_durations[idx] = 0
            self.backward_durations[idx] = 0
            let ptr = Pointer[StringLiteral].alloc(1)
            ptr.store(0, operation_labels.get[idx, StringLiteral]())
            self.operation_labels[idx] = ptr

        unroll[34, loop]()

    # some basic methods ################################################################
    @always_inline
    fn add_leaf_node(inout self, inout a: Tensor):
        a.set_id(self.num_static_nodes)
        a.is_dynamic = False
        self.nodes.push_back(a)
        self.num_static_nodes += 1

    @always_inline
    fn add_dynamic_node(inout self, inout a: Tensor):
        a.set_id(self.num_static_nodes + self.num_dynamic_nodes)
        a.is_dynamic = True
        self.nodes.push_back(a)
        self.num_dynamic_nodes += 1

    @always_inline
    fn print_backward_tape(self):
        print_no_newline("[ ")
        let len = len(self.backward_tape)
        for i in range(len):
            print_no_newline(self.backward_tape[i])
            if i < len - 1:
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn tensor(
        inout self, shape: DynamicVector[Int], requires_grad: Bool = True
    ) -> Tensor:
        var new_tensor = Tensor(shape, requires_grad)
        self.add_leaf_node(new_tensor)
        return new_tensor

    # compute nodes initialization ##########################################################

    # elementwise operators #################################################################

    @always_inline
    fn matmul(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        # check dimensions
        if (
            a.shape[a.num_dims - 1] != b.shape[b.num_dims - 2]
            or min(a.num_dims, b.num_dims) < 2
        ):
            print(
                "Error (at mul): For Matrix Multiplication, Matrices need to in the"
                " following shape: c[mxn] = a[mxk] * b[kxn]"
            )

        let new_num_dims = max(a.num_dims, b.num_dims)
        var new_shape = DynamicVector[Int]()
        let diff = a.num_dims - b.num_dims
        for i in range(new_num_dims - 2):
            if diff > 0 and i < abs(diff):
                new_shape.push_back(a.shape[i])
            elif diff < 0 and i < abs(diff):
                new_shape.push_back(b.shape[i])
            else:
                new_shape.push_back(max(a.shape[i], b.shape[i]))
        new_shape.push_back(a.shape[a.num_dims - 2])
        new_shape.push_back(b.shape[b.num_dims - 1])

        var c = Tensor(new_shape)

        c.set_name("matmul")
        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        matmul(c, a, b)

        return c

    @always_inline
    fn conv_2d(
        inout self, inout a: Tensor, inout b: Tensor, padding: Int, stride: Int
    ) -> Tensor:  # a: input, b: kernels
        # assumption: a (batch of input images) is of shape (batch_size, channels, width, height)
        #             b (set of kernels) is of shape (num_filters, channels, a, b)

        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims

        let batch_size = a.shape[0]
        let in_channels = a.shape[1]
        let width = a.shape[2]
        let height = a.shape[3]

        let out_channels = b.shape[0]
        if in_channels != b.shape[1]:
            print(
                "Error (at conv_2d): number of channels must be equal in the input and"
                " the kernels"
            )
        let kernel_width = b.shape[2]
        let kernel_height = b.shape[3]

        # init result Tensor
        let new_shape = shape(
            batch_size,
            out_channels,
            (width - kernel_width + 2 * padding) // stride + 1,
            (height - kernel_height + 2 * padding) // stride + 1,
        )
        var c = Tensor(new_shape)

        c.other_params.store(0, padding)
        c.other_params.store(1, stride)

        c.set_name("conv_2d")

        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        conv_2d(c, a, b)

        return c

    @always_inline
    fn max_pool_2d(
        inout self,
        inout a: Tensor,
        kernel_width: Int,
        kernel_height: Int,
        stride: Int,
        padding: Int,
    ) -> Tensor:
        let new_shape = shape(
            a.shape[0],
            a.shape[1],
            (2 * padding + a.shape[2] - (kernel_width - 1) - 1) // stride + 1,
            (2 * padding + a.shape[3] - (kernel_height - 1) - 1) // stride + 1,
        )

        var b = Tensor(new_shape)

        b.other_params.store(0, padding)
        b.other_params.store(1, stride)
        b.other_params.store(2, kernel_width)
        b.other_params.store(3, kernel_height)

        b.set_name("max_pool_2d")
        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        max_pool_2d(b, a)

        return b

    @always_inline
    fn sum(inout self, inout a: Tensor) -> Tensor:
        var b = Tensor(shape(1, 1))

        b.set_name("sum")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        sum(b, a)

        return b

    @always_inline
    fn softmax(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("softmax")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        softmax(b, a)

        return b

    @always_inline
    fn mse(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        # check dimensions
        if a.num_dims != b.num_dims:
            print("Error (at mse): number of dimensions are not equal")
        let num_dims = a.num_dims
        if (
            a.shape[num_dims - 2] != b.shape[num_dims - 2]
            or a.shape[num_dims - 1] != b.shape[num_dims - 1]
        ):
            print(
                "Error (at mse): For mse computation, Matrices need to in the following"
                " shape: c[mxn] = (a[mxn] - b[mxn])^2"
            )

        # init result Tensor
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(a.shape[i])
        var c = Tensor(shape(1))

        c.set_name("mse")

        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        mse(c, a, b)

        return c

    @always_inline
    fn ce(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        # check dimensions
        if a.num_dims != b.num_dims:
            print("Error (at ce): number of dimensions are not equal")
        let num_dims = a.num_dims
        if (
            a.shape[num_dims - 2] != b.shape[num_dims - 2]
            or a.shape[num_dims - 1] != b.shape[num_dims - 1]
        ):
            print(
                "Error (at ce): For ce computation, Matrices need to in the following"
                " shape: c[mxn] = op(a[mxn],b[mxn])"
            )

        # init result Tensor
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(a.shape[i])
        var c = Tensor(shape(1))

        c.set_name("ce")
        if a.name == "softmax":
            a.other_params.store(
                0, 3001
            )  # 3001 means that the child is ce node -> simplifies grad computation
        if b.name == "softmax":
            b.other_params.store(0, 3001)

        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        ce(c, a, b)

        return c

    @always_inline
    fn reshape(
        inout self, inout a: Tensor, newShape: DynamicVector[Int]
    ) -> Tensor:  # also braodcastv
        let num_dims = len(newShape)
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(newShape[i])

        var b = Tensor(new_shape)

        if b.cap % a.cap != 0:
            print(
                "Error (at reshape): b.cap % a.cap == 0 and b.cap // a.cap >= 1 is not"
                " fulfilled"
            )

        b.set_name("reshape")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        reshape(b, a)

        return b

    @always_inline
    fn transpose(inout self, inout a: Tensor) -> Tensor:
        let num_dims = a.num_dims
        if num_dims < 2:
            print(
                "Error (at transpose): a transposed Tensor need to heave at least two"
                " dimenions!"
            )

        var new_shape = DynamicVector[Int]()
        for i in range(num_dims - 2):
            new_shape.push_back(a.shape[i])
        new_shape.push_back(a.shape[num_dims - 1])
        new_shape.push_back(a.shape[num_dims - 2])

        var b = Tensor(new_shape)

        b.set_name("transpose")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        transpose(b, a)

        return b

    @always_inline
    fn mean(inout self, inout a: Tensor, dim: DynamicVector[Int]) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        for i in range(len(dim)):
            new_shape[dim[i]] = 1

        var b = Tensor(new_shape)

        # pass the dim list to the result tensor as otherParams
        b.other_params.store(0, len(dim))  # store the num of dims first
        for i in range(len(dim)):
            b.other_params.store(i + 1, dim[i])

        b.set_name("mean")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        mean(b, a)

        return b

    @always_inline
    fn variance(inout self, inout a: Tensor, dim: DynamicVector[Int]) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        for i in range(len(dim)):
            new_shape[dim[i]] = 1

        var b = Tensor(new_shape)

        # pass the dim list to the result tensor as otherParams
        b.other_params.store(0, len(dim))  # store the num of dims first
        for i in range(len(dim)):
            b.other_params.store(i + 1, dim[i])

        b.set_name("variance")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        variance(b, a)

        return b

    @always_inline
    fn std(inout self, inout a: Tensor, dim: DynamicVector[Int]) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        for i in range(len(dim)):
            new_shape[dim[i]] = 1

        var b = Tensor(new_shape)

        # pass the dim list to the result tensor as otherParams
        b.other_params.store(0, len(dim))  # store the num of dims first
        for i in range(len(dim)):
            b.other_params.store(i + 1, dim[i])

        b.set_name("std")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        std(b, a)

        return b

    # elementwise operators ###########################################################

    @always_inline
    fn mul(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        let new_num_dims = max(a.num_dims, b.num_dims)
        var new_shape = DynamicVector[Int]()
        let diff = a.num_dims - b.num_dims
        for i in range(new_num_dims):
            if diff > 0 and i < abs(diff):
                new_shape.push_back(a.shape[i])
            elif diff < 0 and i < abs(diff):
                new_shape.push_back(b.shape[i])
            else:
                new_shape.push_back(max(a.shape[i], b.shape[i]))

        var c = Tensor(new_shape)

        c.set_name("mul")
        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        e_mul(c, a, b)

        return c

    @always_inline
    fn add(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        let new_num_dims = max(a.num_dims, b.num_dims)
        var new_shape = DynamicVector[Int]()
        let diff = a.num_dims - b.num_dims
        for i in range(new_num_dims):
            if diff > 0 and i < abs(diff):
                new_shape.push_back(a.shape[i])
            elif diff < 0 and i < abs(diff):
                new_shape.push_back(b.shape[i])
            else:
                new_shape.push_back(max(a.shape[i], b.shape[i]))

        var c = Tensor(new_shape)

        c.set_name("add")
        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        e_add(c, a, b)

        return c

    @always_inline
    fn sub(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        let new_num_dims = max(a.num_dims, b.num_dims)
        var new_shape = DynamicVector[Int]()
        let diff = a.num_dims - b.num_dims
        for i in range(new_num_dims):
            if diff > 0 and i < abs(diff):
                print(1)
                new_shape.push_back(a.shape[i])
            elif diff < 0 and i < abs(diff):
                print(2)
                new_shape.push_back(b.shape[i])
            else:
                print(3)
                new_shape.push_back(max(a.shape[i], b.shape[i]))

        var c = Tensor(new_shape)

        c.set_name("sub")
        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        e_sub(c, a, b)

        return c

    @always_inline
    fn div(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        let new_num_dims = max(a.num_dims, b.num_dims)
        var new_shape = DynamicVector[Int]()
        let diff = a.num_dims - b.num_dims
        for i in range(new_num_dims):
            if diff > 0 and i < abs(diff):
                new_shape.push_back(a.shape[i])
            elif diff < 0 and i < abs(diff):
                new_shape.push_back(b.shape[i])
            else:
                new_shape.push_back(max(a.shape[i], b.shape[i]))

        var c = Tensor(new_shape)

        c.set_name("div")
        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        e_div(c, a, b)

        return c

    @always_inline
    fn sqrt(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("sqrt")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_sqrt(b, a)

        return b

    @always_inline
    fn abs(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("abs")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_abs(b, a)

        return b

    @always_inline
    fn pow(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:
        let new_num_dims = max(a.num_dims, b.num_dims)
        var new_shape = DynamicVector[Int]()
        let diff = a.num_dims - b.num_dims
        for i in range(new_num_dims):
            if diff > 0 and i < abs(diff):
                new_shape.push_back(a.shape[i])
            elif diff < 0 and i < abs(diff):
                new_shape.push_back(b.shape[i])
            else:
                new_shape.push_back(max(a.shape[i], b.shape[i]))

        var c = Tensor(new_shape)

        c.set_name("e_pow")
        c.num_parents = 2
        c.set_parent(0, a)
        c.set_parent(1, b)

        self.add_dynamic_node(c)

        e_pow(c, a, b)

        return c

    @always_inline
    fn pow(inout self, inout a: Tensor, e: Int) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.other_params.store(0, e)
        b.set_name("e_pow_all")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_pow_all(b, a)

        return b

    @always_inline
    fn exp2(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("exp2")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_exp2(b, a)

        return b

    @always_inline
    fn exp(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("exp")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_exp(b, a)

        return b

    @always_inline
    fn log2(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("log2")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_log2(b, a)

        return b

    @always_inline
    fn log(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("log")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_log(b, a)

        return b

    @always_inline
    fn sin(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("sin")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_sin(b, a)

        return b

    @always_inline
    fn cos(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("cos")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_cos(b, a)

        return b

    @always_inline
    fn tan(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("tan")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_tan(b, a)

        return b

    @always_inline
    fn asin(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("asin")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_asin(b, a)

        return b

    @always_inline
    fn acos(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("acos")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_acos(b, a)

        return b

    @always_inline
    fn atan(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("atan")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_atan(b, a)

        return b

    @always_inline
    fn sinh(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("sinh")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_sinh(b, a)

        return b

    @always_inline
    fn cosh(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("cosh")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_cosh(b, a)

        return b

    @always_inline
    fn tanh(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("tanh")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_tanh(b, a)

        return b

    @always_inline
    fn relu(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("relu")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_relu(b, a)

        return b

    @always_inline
    fn copy(inout self, inout a: Tensor) -> Tensor:
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name("copy")

        b.num_parents = 1
        b.set_parent(0, a)

        self.add_dynamic_node(b)

        e_copy(b, a)

        return b

    fn clear_cache(inout self):
        for i in range(self.num_dynamic_nodes):
            let curr = self.nodes.pop_back()
            curr.shape.free()
            curr.strides.free()
            curr.parents_dynamic.free()
            curr.data.free()
            curr.grad.free()
            curr.velocity.free()
            curr.other_params.free()
            curr.parents.free()
        self.backward_tape.clear()
        self.backward_tape_nodes_dynamic.clear()
        self.num_dynamic_nodes = 0

    @always_inline
    fn zero_grads(inout self):
        for i in range(self.num_static_nodes):
            self.nodes[i].fill_grad(0.0)

    @always_inline
    fn get_parent(inout self, curr_tensor: Tensor, index: Int) -> Tensor:
        return self.nodes[curr_tensor.parents.load(index)]

    fn backward_order(inout self, Tensor: Tensor):
        self.backward_tape = DynamicVector[Int](0)
        self.backward_tape.push_back(Tensor.id)
        var it = 0
        while it < len(self.backward_tape):
            let currId = self.backward_tape[it]
            let curr = self.nodes[currId]
            for i in range(curr.num_parents):
                let parId = curr.get_parent(i)
                let par = self.nodes[parId]
                if par.requires_grad:
                    self.backward_tape.push_back(parId)
            it += 1

    @always_inline
    fn backward(inout self, inout lastTensor: Tensor):
        lastTensor.fill_grad(1)  # the last tensor needs to have a grad of 1, to be able to do the operations backward

        # if lastTensor.cap != 1:
        #     print("Error: Gradient can be implicitly created only for scalar outputs")
        #     return
        self.backward_order(lastTensor)

        for i in range(len(self.backward_tape)):
            let curr = self.nodes[self.backward_tape[i]]

            # Non-elementwise operators
            if curr.name == "matmul":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                matmul_grad(curr, par1, par2)
                self.backward_durations[0] += now() - start
            if curr.name == "conv_2d":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                conv_2d_grad(curr, par1, par2)
                self.backward_durations[1] += now() - start
            if curr.name == "max_pool_2d":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                max_pool_2d_grad(curr, par1)
                self.backward_durations[2] += now() - start
            if curr.name == "sum":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                sum_grad(curr, par1)
                self.backward_durations[3] += now() - start
            if curr.name == "softmax":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                softmax_grad(curr, par1)
                self.backward_durations[4] += now() - start
            if curr.name == "mse":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                mse_grad(curr, par1, par2)
                self.backward_durations[5] += now() - start
            if curr.name == "ce":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                ce_grad(curr, par1, par2)
                self.backward_durations[6] += now() - start
            if curr.name == "reshape":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                reshape_grad(curr, par1)
                self.backward_durations[7] += now() - start
            if curr.name == "transpose":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                transpose_grad(curr, par1)
                self.backward_durations[8] += now() - start
            if curr.name == "mean":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                mean_grad(curr, par1)
                self.backward_durations[9] += now() - start
            if curr.name == "variance":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                variance_grad(curr, par1)
                self.backward_durations[10] += now() - start
            if curr.name == "std":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                std_grad(curr, par1)
                self.backward_durations[11] += now() - start

            # elementwise operators #########################
            if curr.name == "mul":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                e_mul_grad(curr, par1, par2)
                self.backward_durations[12] += now() - start
            if curr.name == "add":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                e_add_grad(curr, par1, par2)
                self.backward_durations[13] += now() - start
            if curr.name == "sub":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                e_sub_grad(curr, par1, par2)
                self.backward_durations[14] += now() - start
            if curr.name == "div":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                e_div_grad(curr, par1, par2)
                self.backward_durations[15] += now() - start
            if curr.name == "sqrt":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_sqrt_grad(curr, par1)
                self.backward_durations[16] += now() - start
            if curr.name == "abs":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_abs_grad(curr, par1)
                self.backward_durations[17] += now() - start
            if curr.name == "e_pow":
                var par1 = self.get_parent(curr, 0)
                var par2 = self.get_parent(curr, 1)
                let start = now()
                e_pow_grad(curr, par1, par2)
                self.forward_durations[18] += now() - start
            if curr.name == "e_pow_all":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_pow_all_grad(curr, par1)
                self.forward_durations[19] += now() - start
            if curr.name == "exp2":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_exp2_grad(curr, par1)
                self.backward_durations[20] += now() - start
            if curr.name == "exp":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_exp_grad(curr, par1)
                self.backward_durations[21] += now() - start
            if curr.name == "log2":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_log2_grad(curr, par1)
                self.backward_durations[22] += now() - start
            if curr.name == "log":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_log_grad(curr, par1)
                self.backward_durations[23] += now() - start
            if curr.name == "sin":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_sin_grad(curr, par1)
                self.backward_durations[24] += now() - start
            if curr.name == "cos":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_cos_grad(curr, par1)
                self.backward_durations[25] += now() - start
            if curr.name == "tan":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_tan_grad(curr, par1)
                self.backward_durations[26] += now() - start
            if curr.name == "asin":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_asin_grad(curr, par1)
                self.backward_durations[27] += now() - start
            if curr.name == "acos":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_acos_grad(curr, par1)
                self.backward_durations[28] += now() - start
            if curr.name == "atan":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_atan_grad(curr, par1)
                self.backward_durations[29] += now() - start
            if curr.name == "sinh":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_sinh_grad(curr, par1)
                self.backward_durations[30] += now() - start
            if curr.name == "cosh":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_cosh_grad(curr, par1)
                self.backward_durations[31] += now() - start
            if curr.name == "tanh":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_tanh_grad(curr, par1)
                self.backward_durations[32] += now() - start
            if curr.name == "relu":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_relu_grad(curr, par1)
                self.backward_durations[33] += now() - start
            if curr.name == "copy":
                var par1 = self.get_parent(curr, 0)
                let start = now()
                e_copy_grad(curr, par1)
                self.backward_durations[34] += now() - start

        self.num_passes += 1

    fn optimize(
        inout self,
        optType: String,
        lr: Float32 = 0.001,
        momentum: Float32 = 0.9,
        weight_decay: Float32 = 0.001,
        threshold: Float32 = Float32(100.0),
    ):
        if optType == "sgd":
            for i in range(self.num_static_nodes):
                var curr_node = self.nodes[i]
                if curr_node.requires_grad:
                    for index in range(curr_node.cap):
                        curr_node.set_data(
                            index,
                            (1 - lr * weight_decay) * curr_node.data.load(index)
                            - lr
                            * min(
                                threshold, max(-threshold, curr_node.grad.load(index))
                            ),
                        )

                    @parameter
                    fn v_update_data_sgd[nelts: Int](index: Int):
                        curr_node.data.simd_store[nelts](
                            index,
                            (1 - lr * weight_decay)
                            * curr_node.data.simd_load[nelts](index)
                            - lr * curr_node.grad.simd_load[nelts](index),
                        )

                    vectorize[nelts, v_update_data_sgd](curr_node.cap)

        if optType == "sgd_momentum":
            for i in range(self.num_static_nodes):
                var curr_node = self.nodes[i]
                if curr_node.requires_grad:

                    @parameter
                    fn v_set_velocity[nelts: Int](index: Int):
                        curr_node.velocity.simd_store[nelts](
                            index,
                            momentum * curr_node.velocity.simd_load[nelts](index)
                            + lr * curr_node.grad.simd_load[nelts](index),
                        )

                    vectorize[nelts, v_set_velocity](curr_node.cap)

                    @parameter
                    fn v_update_data_sgdPlus[nelts: Int](index: Int):
                        curr_node.data.simd_store[nelts](
                            index,
                            (1 - lr * weight_decay)
                            * curr_node.data.simd_load[nelts](index)
                            - curr_node.velocity.simd_load[nelts](index),
                        )

                    vectorize[nelts, v_update_data_sgdPlus](curr_node.cap)

    @always_inline
    fn print_nodes(self):
        print("Printing all nodes of the current computational Graph .....\n")
        for i in range(self.num_static_nodes):
            let n = self.nodes[i]
            print(
                "Tensor ID: ",
                n.id,
                ", Name: ",
                n.name,
                ", rquiresGrad: ",
                n.requires_grad,
                ", cap = ",
                n.cap,
            )
            n.print_data()
            n.print_grad()
        print("End of Printing all nodes of the current computational Graph.")

    @always_inline
    fn print_forward_durations(self):
        var summed_duration: Int = 0
        for i in range(34):
            if self.forward_durations[i] > 0:
                summed_duration += self.forward_durations[i]

        let average_ms_divisor = Float32(1_000_000 * self.num_passes)
        print("\nAverage time spent in each operation per forward pass:")

        @parameter
        fn loop[idx: Int]():
            if self.forward_durations[idx] > 0:
                print(
                    self.operation_labels[idx].load(0),
                    Float32(self.forward_durations[idx]) / average_ms_divisor,
                    "ms",
                    " (",
                    100.0 * Float32(self.forward_durations[idx]) / summed_duration,
                    "% )",
                )

        unroll[34, loop]()

    @always_inline
    fn print_backward_durations(self):
        var summed_duration: Int = 0
        for i in range(34):
            if self.backward_durations[i] > 0:
                summed_duration += self.backward_durations[i]

        let average_ms_divisor = Float32(1_000_000 * self.num_passes)
        print("\nAverage time spent in each operation per backward pass:")

        @parameter
        fn loop[idx: Int]():
            if self.backward_durations[idx] > 0:
                let label: String = self.operation_labels[idx].load(0)
                print(
                    self.operation_labels[idx].load(0),
                    Float32(self.backward_durations[idx]) / average_ms_divisor,
                    "ms",
                    " (",
                    100.0 * Float32(self.backward_durations[idx]) / summed_duration,
                    "% )",
                )

        unroll[34, loop]()
