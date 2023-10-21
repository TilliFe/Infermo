from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from time import now
from algorithm import vectorize, parallelize, unroll
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp, min, max

from ..graph.tensor import Tensor
from ..operators.forward import matmul, conv_2d, max_pool_2d, sum, softmax, mse, ce, reshape, transpose, mean, variance, std, e_mul, e_add, e_sub, e_div, e_sqrt, e_abs, e_pow, e_pow_all, e_exp2, e_exp, e_log2, e_log, e_sin, e_cos, e_tan, e_asin, e_acos, e_atan, e_sinh, e_cosh, e_tanh, e_relu, e_copy
from ..operators.backward import matmul_grad, conv_2d_grad, max_pool_2d_grad, sum_grad, softmax_grad, mse_grad, ce_grad, reshape_grad, transpose_grad, mean_grad, variance_grad, std_grad, e_mul_grad, e_add_grad, e_sub_grad, e_div_grad, e_sqrt_grad, e_abs_grad, e_pow_grad, e_pow_all_grad, e_exp2_grad, e_exp_grad, e_log2_grad, e_log_grad, e_sin_grad, e_cos_grad, e_tan_grad, e_asin_grad, e_acos_grad, e_atan_grad, e_sinh_grad, e_cosh_grad, e_tanh_grad, e_relu_grad, e_copy_grad 
from ..helpers.shape import shape, Vec

alias nelts = simdwidthof[DType.float32]()


struct Module:
    var tensors: DynamicVector[Tensor]
    var counter: Int
    var forward_tape: DynamicVector[Int]
    var backward_tape: DynamicVector[Int]
    var num_passes: Int
    var forward_durations: DynamicVector[Int]
    var backward_durations: DynamicVector[Int]
    var operation_labels: DynamicVector[Pointer[StringLiteral]]

    fn __init__(inout self):
        self.tensors = DynamicVector[Tensor](0)
        self.counter = 0
        self.forward_tape = DynamicVector[Int]()
        self.backward_tape = DynamicVector[Int]()
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
    fn add_to_graph(inout self, inout a: Tensor):
        a.set_id(self.counter)
        a.in_tensors = True
        self.counter += 1
        self.tensors.push_back(a)

    @always_inline
    fn print_forward_tape(self):
        print_no_newline("[ ")
        let len = len(self.forward_tape)
        for i in range(len):
            print_no_newline(self.forward_tape[i])
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")
    
    @always_inline
    fn print_backward_tape(self):
        print_no_newline("[ ")
        let len = len(self.backward_tape)
        for i in range(len):
            print_no_newline(self.backward_tape[i])
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")




    # compute nodes initialization ##########################################################
    
    # elementwise operators #################################################################
    
    @always_inline
    fn matmul(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        # # check dimensions
        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims
        if(a.shape[a_num_dims-1] != b.shape[b_num_dims-2]):
            print("Error (at mul): For Matrix Multiplication, Matrices need to in the following shape: c[mxn] = a[mxk] * b[kxn]")

        # init result Tensor 
        var new_shape = DynamicVector[Int](0)
        
        # regular
        if(a_num_dims == b_num_dims):
            for i in range(b_num_dims-2):
                if(a.shape[i] == 1 and b.shape[i] != 1):
                    new_shape.push_back(b.shape[i])
                elif(a.shape[i] != 1 and b.shape[i] == 1):
                    new_shape.push_back(a.shape[i])
                else:
                    new_shape.push_back(a.shape[i])
            new_shape.push_back(a.shape[a_num_dims-2])
            new_shape.push_back(b.shape[b_num_dims-1])

        # broadcast a
        elif(b_num_dims > a_num_dims):
            for i in range(b_num_dims-2):
                new_shape.push_back(b.shape[i])
            new_shape.push_back(a.shape[a_num_dims-2])
            new_shape.push_back(b.shape[b_num_dims-1])

        # broadcast b 
        elif(a_num_dims > b_num_dims):
            for i in range(a_num_dims-1):
                new_shape.push_back(a.shape[i])
            new_shape.push_back(b.shape[b_num_dims-1])        

        var c = Tensor(new_shape)

        c.set_name('matmul')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 


    @always_inline
    fn conv_2d(inout self, inout a: Tensor, inout b: Tensor, padding: Int, stride: Int) -> Tensor: # a: input, b: kernels

        # assumption: a (batch of input images) is of shape (batch_size, channels, width, height)
        #             b (set of kernels) is of shape (num_filters, channels, a, b)

        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims

        let batch_size = a.shape[0]
        let in_channels = a.shape[1]
        let width = a.shape[2]
        let height = a.shape[3]

        let out_channels = b.shape[0]
        if(in_channels != b.shape[1]):
            print("Error (at conv_2d): number of channels must be equal in the input and the kernels")
        let kernel_width = b.shape[2]
        let kernel_height = b.shape[3]

        # init result Tensor 
        let new_shape = shape(batch_size,out_channels, (width - kernel_width + 2*padding) // stride + 1, (height - kernel_height + 2*padding) // stride + 1) 
        var c = Tensor(new_shape)

        c.other_params.store(0, padding)
        c.other_params.store(1, stride)

        c.set_name('conv_2d')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 


    @always_inline
    fn max_pool_2d(inout self, inout a: Tensor, kernel_width: Int, kernel_height: Int, stride: Int, padding: Int) -> Tensor: 
        let new_shape = shape(a.shape[0],a.shape[1],(2*padding + a.shape[2] - (kernel_width - 1) - 1)//stride + 1, (2*padding + a.shape[3] - (kernel_height - 1) - 1)//stride + 1)

        var b = Tensor(new_shape)

        b.other_params.store(0,padding)
        b.other_params.store(1,stride)
        b.other_params.store(2,kernel_width)
        b.other_params.store(3,kernel_height)

        b.set_name('max_pool_2d')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b


    @always_inline
    fn sum(inout self, inout a: Tensor) -> Tensor: 

        var b = Tensor(shape(1,1))

        b.set_name('sum')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b


    @always_inline
    fn softmax(inout self, inout a: Tensor) -> Tensor: 

        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('softmax')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b


    @always_inline
    fn mse(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        # check dimensions
        if(a.num_dims != b.num_dims):
            print("Error (at mse): number of dimensions are not equal")
        let num_dims = a.num_dims
        if(a.shape[num_dims-2] != b.shape[num_dims-2] or a.shape[num_dims-1] != b.shape[num_dims-1]):
            print("Error (at mse): For mse computation, Matrices need to in the following shape: c[mxn] = (a[mxn] - b[mxn])^2")

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(a.shape[i])
        var c = Tensor(shape(1))

        c.set_name('mse')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 

    @always_inline
    fn ce(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        # check dimensions
        if(a.num_dims != b.num_dims):
            print("Error (at ce): number of dimensions are not equal")
        let num_dims = a.num_dims
        if(a.shape[num_dims-2] != b.shape[num_dims-2] or a.shape[num_dims-1] != b.shape[num_dims-1]):
            print("Error (at ce): For ce computation, Matrices need to in the following shape: c[mxn] = op(a[mxn],b[mxn])")

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(a.shape[i])
        var c = Tensor(shape(1))

        c.set_name('ce')
        if(a.name == "softmax"):
            a.other_params.store(0,3001) # 3001 means that the child is ce node -> simplifies grad computation
        if(b.name == "softmax"):
            b.other_params.store(0,3001)

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 

    @always_inline
    fn reshape(inout self, inout a: Tensor, newShape: DynamicVector[Int]) -> Tensor: # also braodcastv
        let num_dims = len(newShape)
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(newShape[i])

        var b = Tensor(new_shape)

        if(b.cap % a.cap != 0):
            print("Error (at reshape): b.cap % a.cap == 0 and b.cap // a.cap >= 1 is not fulfilled")

        b.set_name('reshape')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn transpose(inout self, inout a: Tensor) -> Tensor: 
        let num_dims = a.num_dims
        if(num_dims < 2):
            print("Error (at transpose): a transposed Tensor need to heave at least two dimenions!")

        var new_shape = DynamicVector[Int]()
        for i in range(num_dims - 2):
            new_shape.push_back(a.shape[i])
        new_shape.push_back(a.shape[num_dims-1])
        new_shape.push_back(a.shape[num_dims-2])

        var b = Tensor(new_shape)

        b.set_name('transpose')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

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
        b.other_params.store(0,len(dim)) # store the num of dims first
        for i in range(len(dim)):
            b.other_params.store(i+1,dim[i])

        b.set_name('mean')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

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
        b.other_params.store(0,len(dim)) # store the num of dims first
        for i in range(len(dim)):
            b.other_params.store(i+1,dim[i])

        b.set_name('variance')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

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
        b.other_params.store(0,len(dim)) # store the num of dims first
        for i in range(len(dim)):
            b.other_params.store(i+1,dim[i])

        b.set_name('std')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    
    # elementwise operators ###########################################################

    @always_inline
    fn mul(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        var new_shape_reversed = DynamicVector[Int]()

        # Calculate the maximum number of dimensions
        let max_dims = max(a.num_dims, b.num_dims)

        for i in range(max_dims-1, -1, -1):
            # Calculate the corresponding index for a and b
            let idx_a = i - (max_dims - a.num_dims)
            let idx_b = i - (max_dims - b.num_dims)

            # Check if the index is valid for both tensors
            if (idx_a >= 0 and idx_b >= 0):
                if (a.shape[idx_a] == 1 and b.shape[idx_b] != 1) :
                    new_shape_reversed.push_back(b.shape[idx_b])
                elif (a.shape[idx_a] != 1 and b.shape[idx_b] == 1):
                    new_shape_reversed.push_back(a.shape[idx_a])
                elif (a.shape[idx_a] == b.shape[idx_b]) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                else :
                    print("Error (at mul): shapes are not compatible for broadcasting")
                
            elif (idx_a >= 0) :  # The index is only valid for a
                new_shape_reversed.push_back(a.shape[idx_a])
            else :  # The index is only valid for b
                new_shape_reversed.push_back(b.shape[idx_b])

        # Reverse the shape to get the correct order
        for i in range(len(new_shape_reversed)):
            new_shape.push_back(new_shape_reversed[len(new_shape_reversed)-1-i])

        var c = Tensor(new_shape)

        c.set_name('mul')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 


    @always_inline
    fn add(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        var new_shape_reversed = DynamicVector[Int]()

        # Calculate the maximum number of dimensions
        let max_dims = max(a.num_dims, b.num_dims)

        for i in range(max_dims-1, -1, -1):
            # Calculate the corresponding index for a and b
            let idx_a = i - (max_dims - a.num_dims)
            let idx_b = i - (max_dims - b.num_dims)

            # Check if the index is valid for both tensors
            if (idx_a >= 0 and idx_b >= 0):
                if (a.shape[idx_a] == 1 and b.shape[idx_b] != 1) :
                    new_shape_reversed.push_back(b.shape[idx_b])
                elif (a.shape[idx_a] != 1 and b.shape[idx_b] == 1) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                elif (a.shape[idx_a] == b.shape[idx_b]) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                else :
                    print("Error (at mul): shapes are not compatible for broadcasting")
                
            elif (idx_a >= 0) :  # The index is only valid for a
                new_shape_reversed.push_back(a.shape[idx_a])
            else :  # The index is only valid for b
                new_shape_reversed.push_back(b.shape[idx_b])

        # Reverse the shape to get the correct order
        for i in range(len(new_shape_reversed)):
            new_shape.push_back(new_shape_reversed[len(new_shape_reversed)-1-i])

        var c = Tensor(new_shape)

        c.set_name('add')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c    

    @always_inline
    fn sub(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        var new_shape_reversed = DynamicVector[Int]()

        # Calculate the maximum number of dimensions
        let max_dims = max(a.num_dims, b.num_dims)

        for i in range(max_dims-1, -1, -1):
            # Calculate the corresponding index for a and b
            let idx_a = i - (max_dims - a.num_dims)
            let idx_b = i - (max_dims - b.num_dims)

            # Check if the index is valid for both tensors
            if (idx_a >= 0 and idx_b >= 0):
                if (a.shape[idx_a] == 1 and b.shape[idx_b] != 1) :
                    new_shape_reversed.push_back(b.shape[idx_b])
                elif (a.shape[idx_a] != 1 and b.shape[idx_b] == 1) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                elif (a.shape[idx_a] == b.shape[idx_b]) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                else :
                    print("Error (at mul): shapes are not compatible for broadcasting")
                
            elif (idx_a >= 0) :  # The index is only valid for a
                new_shape_reversed.push_back(a.shape[idx_a])
            else :  # The index is only valid for b
                new_shape_reversed.push_back(b.shape[idx_b])

        # Reverse the shape to get the correct order
        for i in range(len(new_shape_reversed)):
            new_shape.push_back(new_shape_reversed[len(new_shape_reversed)-1-i])

        var c = Tensor(new_shape)

        c.set_name('sub')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 

    @always_inline
    fn div(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        var new_shape_reversed = DynamicVector[Int]()

        # Calculate the maximum number of dimensions
        let max_dims = max(a.num_dims, b.num_dims)

        for i in range(max_dims-1, -1, -1):
            # Calculate the corresponding index for a and b
            let idx_a = i - (max_dims - a.num_dims)
            let idx_b = i - (max_dims - b.num_dims)

            # Check if the index is valid for both tensors
            if (idx_a >= 0 and idx_b >= 0):
                if (a.shape[idx_a] == 1 and b.shape[idx_b] != 1) :
                    new_shape_reversed.push_back(b.shape[idx_b])
                elif (a.shape[idx_a] != 1 and b.shape[idx_b] == 1) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                elif (a.shape[idx_a] == b.shape[idx_b]) :
                    new_shape_reversed.push_back(a.shape[idx_a])
                else :
                    print("Error (at mul): shapes are not compatible for broadcasting")
                
            elif (idx_a >= 0) :  # The index is only valid for a
                new_shape_reversed.push_back(a.shape[idx_a])
            else :  # The index is only valid for b
                new_shape_reversed.push_back(b.shape[idx_b])

        # Reverse the shape to get the correct order
        for i in range(len(new_shape_reversed)):
            new_shape.push_back(new_shape_reversed[len(new_shape_reversed)-1-i])

        var c = Tensor(new_shape)

        c.set_name('div')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 


    @always_inline
    fn sqrt(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('sqrt')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn abs(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('abs')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b
        
    @always_inline
    fn pow(inout self, inout a: Tensor, inout b: Tensor) -> Tensor:

        # # check dimensions
        let a_num_dims = a.num_dims
        let b_num_dims = b.num_dims
        
        # init result Tensor 
        var new_shape = DynamicVector[Int](0)
        
        # regular
        if(a_num_dims == b_num_dims):
            for i in range(b_num_dims-2):
                if(a.shape[i] == 1 and b.shape[i] != 1):
                    new_shape.push_back(b.shape[i])
                elif(a.shape[i] != 1 and b.shape[i] == 1):
                    new_shape.push_back(a.shape[i])
                else:
                    new_shape.push_back(a.shape[i])
            new_shape.push_back(a.shape[a_num_dims-2])
            new_shape.push_back(b.shape[b_num_dims-1])

        # broadcast a
        elif(b_num_dims > a_num_dims):
            for i in range(b_num_dims-2):
                new_shape.push_back(b.shape[i])
            new_shape.push_back(a.shape[a_num_dims-2])
            new_shape.push_back(b.shape[b_num_dims-1])

        # broadcast b 
        elif(a_num_dims > b_num_dims):
            for i in range(a_num_dims-1):
                new_shape.push_back(a.shape[i])
            new_shape.push_back(b.shape[b_num_dims-1])        

        var c = Tensor(new_shape)

        c.set_name('e_pow')

        if(not a.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            c.add_parent(a.id)

        if(not b.in_tensors):
            c.add_parent(self.counter)
            self.add_to_graph(b)
        else:
            c.add_parent(b.id)
        self.add_to_graph(c)

        return c 


    @always_inline
    fn pow(inout self, inout a: Tensor, e: Int) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.other_params.store(0,e)
        b.set_name('e_pow_all')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn exp2(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('exp2')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn exp(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('exp')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn log2(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('log2')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn log(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('log')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn sin(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('sin')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn cos(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('cos')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn tan(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('tan')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn asin(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('asin')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn acos(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('acos')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn atan(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('atan')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b


    @always_inline
    fn sinh(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('sinh')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn cosh(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('cosh')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    @always_inline
    fn tanh(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('tanh')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b

    
    @always_inline
    fn relu(inout self, inout a: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('relu')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b


    @always_inline
    fn copy(inout self, inout a: Tensor) -> Tensor: 

        var new_shape = DynamicVector[Int]()
        for i in range(a.num_dims):
            new_shape.push_back(a.shape[i])

        var b = Tensor(new_shape)

        b.set_name('copy')

        if(not a.in_tensors):
            b.add_parent(self.counter)
            self.add_to_graph(a)
        else:
            b.add_parent(a.id)
        self.add_to_graph(b)

        return b




    # Graph tranversal: forward and backward ###############################################

    fn top_order(inout self, inout Tensor: Tensor):  
        if not Tensor.visited:
            for i in range(Tensor.num_parents):
                let nextTensorId = Tensor.get_parent(i)
                var nextTensor = self.tensors[nextTensorId]
                self.top_order(nextTensor)
            self.forward_tape.push_back(Tensor.id)
            Tensor.visited = True

    @always_inline
    fn forward(inout self, inout computingTensor: Tensor):
        for i in range(self.counter):
            self.tensors[i].set_visited(False)
            if(self.tensors[i].name != 'none'):
                self.tensors[i].fill(0)
        self.forward_tape = DynamicVector[Int]()
        self.top_order(computingTensor)
        self.num_passes += 1

        for i in range(self.counter):
            var curr = self.tensors[i]

            # Non-elementwise operators
            if(curr.name == 'matmul'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                matmul(curr,par1,par2)
                self.forward_durations[0] += (now() - start)
            if(curr.name == 'conv_2d'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                conv_2d(curr,par1,par2)
                self.forward_durations[1] += (now() - start)
            if(curr.name == 'max_pool_2d'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                max_pool_2d(curr,par1) 
                self.forward_durations[2] += (now() - start)
            if(curr.name == 'sum'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                sum(curr,par1)
                self.forward_durations[3] += (now() - start)
            if(curr.name == 'softmax'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                softmax(curr,par1)
                self.forward_durations[4] += (now() - start)
            if(curr.name == 'mse'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                mse(curr,par1,par2) 
                self.forward_durations[5] += (now() - start)
            if(curr.name == 'ce'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                ce(curr,par1,par2) 
                self.forward_durations[6] += (now() - start)
            if(curr.name == 'reshape'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                reshape(curr,par1)
                self.forward_durations[7] += (now() - start)
            if(curr.name == 'transpose'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                transpose(curr,par1)
                self.forward_durations[8] += (now() - start)
            if(curr.name == 'mean'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                mean(curr,par1)
                self.forward_durations[9] += (now() - start)
            if(curr.name == 'variance'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                variance(curr,par1)
                self.forward_durations[10] += (now() - start)
            if(curr.name == 'std'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                std(curr,par1)
                self.forward_durations[11] += (now() - start)

            # elementwise operators
            if(curr.name == 'mul'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_mul(curr,par1,par2)
                self.forward_durations[12] += (now() - start)
            if(curr.name == 'add'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_add(curr,par1,par2)
                self.forward_durations[13] += (now() - start)
            if(curr.name == 'sub'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_sub(curr,par1,par2)
                self.forward_durations[14] += (now() - start)
            if(curr.name == 'div'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_div(curr,par1,par2)
                self.forward_durations[15] += (now() - start)
            if(curr.name == 'sqrt'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_sqrt(curr,par1)
                self.forward_durations[16] += (now() - start)
            if(curr.name == 'abs'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_abs(curr,par1)
                self.forward_durations[17] += (now() - start)
            if(curr.name == 'e_pow'):
                let par1 = self.tensors[curr.get_parent(0)]
                let par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_pow(curr,par1,par2) 
                self.forward_durations[18] += (now() - start)
            if(curr.name == 'e_pow_all'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_pow_all(curr,par1) 
                self.forward_durations[19] += (now() - start)
            if(curr.name == 'exp2'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_exp2(curr,par1)
                self.forward_durations[20] += (now() - start)
            if(curr.name == 'exp'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_exp(curr,par1)
                self.forward_durations[21] += (now() - start)
            if(curr.name == 'log2'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_log2(curr,par1)
                self.forward_durations[22] += (now() - start)
            if(curr.name == 'log'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_log(curr,par1)
                self.forward_durations[23] += (now() - start)
            if(curr.name == 'sin'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_sin(curr,par1)
                self.forward_durations[24] += (now() - start)
            if(curr.name == 'cos'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_cos(curr,par1)
                self.forward_durations[25] += (now() - start)
            if(curr.name == 'tan'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_tan(curr,par1)
                self.forward_durations[26] += (now() - start)
            if(curr.name == 'asin'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_asin(curr,par1)
                self.forward_durations[27] += (now() - start)
            if(curr.name == 'acos'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_acos(curr,par1)
                self.forward_durations[28] += (now() - start)
            if(curr.name == 'atan'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_atan(curr,par1)
                self.forward_durations[29] += (now() - start)
            if(curr.name == 'sinh'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_sinh(curr,par1)
                self.forward_durations[30] += (now() - start)
            if(curr.name == 'cosh'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_cosh(curr,par1)
                self.forward_durations[31] += (now() - start)
            if(curr.name == 'tanh'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_tanh(curr,par1)
                self.forward_durations[32] += (now() - start)
            if(curr.name == 'relu'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_relu(curr,par1) 
                self.forward_durations[33] += (now() - start)
            if(curr.name == 'copy'):
                let par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_copy(curr,par1)
                self.forward_durations[34] += (now() - start)

    fn backward_order(inout self, Tensor: Tensor):
        self.backward_tape = DynamicVector[Int](0)
        self.backward_tape.push_back(Tensor.id)
        var it = 0
        while(it < len(self.backward_tape)):
            let currId = self.backward_tape[it]
            let curr = self.tensors[currId]
            for i in range(curr.num_parents):
                let parId = curr.get_parent(i)
                let par = self.tensors[parId]
                if(par.requires_grad):
                    self.backward_tape.push_back(parId)
            it += 1

    @always_inline
    fn backward(inout self, inout lastTensor: Tensor):
        if(lastTensor.cap != 1):
            print("Error: Gradient can be implicitly created only for scalar outputs")
            return
        self.backward_order(lastTensor)
        for i in range(self.counter):
            if(self.tensors[i].requires_grad):
                self.tensors[i].fill_grad(0)

        for i in range(len(self.backward_tape)):
            let currId = self.backward_tape[i]
            let curr = self.tensors[currId]

            # Non-elementwise operators
            if(curr.name == 'matmul'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                matmul_grad(curr,par1,par2)
                self.backward_durations[0] += (now() - start)
            if(curr.name == 'conv_2d'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                conv_2d_grad(curr,par1,par2)
                self.backward_durations[1] += (now() - start)
            if(curr.name == 'max_pool_2d'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                max_pool_2d_grad(curr,par1)
                self.backward_durations[2] += (now() - start)
            if(curr.name == 'sum'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                sum_grad(curr,par1)
                self.backward_durations[3] += (now() - start)
            if(curr.name == 'softmax'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                softmax_grad(curr,par1)
                self.backward_durations[4] += (now() - start)
            if(curr.name == 'mse'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                mse_grad(curr,par1,par2)
                self.backward_durations[5] += (now() - start)
            if(curr.name == 'ce'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                ce_grad(curr,par1,par2)
                self.backward_durations[6] += (now() - start)
            if(curr.name == 'reshape'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                reshape_grad(curr,par1)
                self.backward_durations[7] += (now() - start)
            if(curr.name == 'transpose'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                transpose_grad(curr,par1)
                self.backward_durations[8] += (now() - start)
            if(curr.name == 'mean'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                mean_grad(curr,par1)
                self.backward_durations[9] += (now() - start)
            if(curr.name == 'variance'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                variance_grad(curr,par1)
                self.backward_durations[10] += (now() - start)
            if(curr.name == 'std'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                std_grad(curr,par1)
                self.backward_durations[11] += (now() - start)

            # elementwise operators #########################
            if(curr.name == 'mul'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_mul_grad(curr,par1,par2)
                self.backward_durations[12] += (now() - start)
            if(curr.name == 'add'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_add_grad(curr,par1,par2)
                self.backward_durations[13] += (now() - start)
            if(curr.name == 'sub'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_sub_grad(curr,par1,par2)
                self.backward_durations[14] += (now() - start)
            if(curr.name == 'div'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_div_grad(curr,par1,par2)
                self.backward_durations[15] += (now() - start)
            if(curr.name == 'sqrt'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_sqrt_grad(curr,par1)
                self.backward_durations[16] += (now() - start)
            if(curr.name == 'abs'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_abs_grad(curr,par1)
                self.backward_durations[17] += (now() - start)
            if(curr.name == 'e_pow'):
                var par1 = self.tensors[curr.get_parent(0)]
                var par2 = self.tensors[curr.get_parent(1)]
                let start = now()
                e_pow_grad(curr,par1,par2) 
                self.forward_durations[18] += (now() - start)
            if(curr.name == 'e_pow_all'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_pow_all_grad(curr,par1) 
                self.forward_durations[19] += (now() - start)
            if(curr.name == 'exp2'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_exp2_grad(curr,par1)
                self.backward_durations[20] += (now() - start)
            if(curr.name == 'exp'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_exp_grad(curr,par1)
                self.backward_durations[21] += (now() - start)
            if(curr.name == 'log2'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_log2_grad(curr,par1)
                self.backward_durations[22] += (now() - start)
            if(curr.name == 'log'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_log_grad(curr,par1)
                self.backward_durations[23] += (now() - start)
            if(curr.name == 'sin'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_sin_grad(curr,par1)
                self.backward_durations[24] += (now() - start)
            if(curr.name == 'cos'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_cos_grad(curr,par1)
                self.backward_durations[25] += (now() - start)
            if(curr.name == 'tan'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_tan_grad(curr,par1)
                self.backward_durations[26] += (now() - start)
            if(curr.name == 'asin'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_asin_grad(curr,par1)
                self.backward_durations[27] += (now() - start)
            if(curr.name == 'acos'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_acos_grad(curr,par1)
                self.backward_durations[28] += (now() - start)
            if(curr.name == 'atan'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_atan_grad(curr,par1)
                self.backward_durations[29] += (now() - start)
            if(curr.name == 'sinh'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_sinh_grad(curr,par1)
                self.backward_durations[30] += (now() - start)
            if(curr.name == 'cosh'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_cosh_grad(curr,par1)
                self.backward_durations[31] += (now() - start)
            if(curr.name == 'tanh'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_tanh_grad(curr,par1)
                self.backward_durations[32] += (now() - start)
            if(curr.name == 'relu'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_relu_grad(curr,par1)
                self.backward_durations[33] += (now() - start)
            if(curr.name == 'copy'):
                var par1 = self.tensors[curr.get_parent(0)]
                let start = now()
                e_copy_grad(curr,par1)
                self.backward_durations[34] += (now() - start)


    fn optimize(inout self, optType: String, lr: Float32 = 0.001, momentum: Float32 = 0.9, weight_decay: Float32 = 0.001, threshold: Float32 = Float32(100.0)):
        
        if(optType == "sgd"):
            for i in range(len(self.backward_tape)):
                let id = self.tensors[self.backward_tape[i]].id
                for index in range(self.tensors[id].cap):
                    self.tensors[id].set_data(index, (1 - lr * weight_decay) * self.tensors[id].data.load(index) - lr * min(threshold,max(-threshold,self.tensors[id].grad.load(index))))
                @parameter
                fn v_update_data_sgd[nelts: Int](index: Int):
                    self.tensors[id].data.simd_store[nelts](
                        index, (1 - lr * weight_decay) * self.tensors[id].data.simd_load[nelts](index) - lr * self.tensors[id].grad.simd_load[nelts](index)
                    )
                vectorize[nelts, v_update_data_sgd](self.tensors[id].cap)
        
        if(optType == "sgd_momentum"):
            for i in range(len(self.backward_tape)):
                let id = self.tensors[self.backward_tape[i]].id

                @parameter
                fn v_set_velocity[nelts: Int](index: Int):
                    self.tensors[id].velocity.simd_store[nelts](
                        index, momentum * self.tensors[id].velocity.simd_load[nelts](index) + lr * self.tensors[id].grad.simd_load[nelts](index)
                    )
                vectorize[nelts, v_set_velocity](self.tensors[id].cap)

                @parameter
                fn v_update_data_sgdPlus[nelts: Int](index: Int):
                    self.tensors[id].data.simd_store[nelts](
                        index, (1 - lr * weight_decay) * self.tensors[id].data.simd_load[nelts](index) - self.tensors[id].velocity.simd_load[nelts](index)
                    )
                vectorize[nelts, v_update_data_sgdPlus](self.tensors[id].cap)


    @always_inline
    fn print_graph(self): 
        print("Printing all tensors of the computational Graph .....\n")
        for i in range(self.counter):
            let n = self.tensors[i]
            print("Tensor ID: ", n.id, ", Name: ", n.name, ", rquiresGrad: ", n.requires_grad, ", cap = ", n.cap)
            n.print_data()
            n.print_grad()
        print("End of Printing all tensors of the computational Graph.")


    @always_inline
    fn print_forward_durations(self):
        var summed_duration: Int = 0
        for i in range(34):
            if(self.forward_durations[i] > 0):
                summed_duration += self.forward_durations[i]

        let average_ms_divisor = Float32(1_000_000 * self.num_passes)
        print("\nAverage time spent in each operation per forward pass:")
        @parameter
        fn loop[idx: Int]():
            if self.forward_durations[idx] > 0:
                print(self.operation_labels[idx].load(0), Float32(self.forward_durations[idx]) / average_ms_divisor, "ms", " (", 100.0 * Float32(self.forward_durations[idx])/summed_duration,"% )")
        unroll[34, loop]()
        

    @always_inline
    fn print_backward_durations(self):
        var summed_duration: Int = 0
        for i in range(34):
            if(self.backward_durations[i] > 0):
                summed_duration += self.backward_durations[i]

        let average_ms_divisor = Float32(1_000_000 * self.num_passes)
        print("\nAverage time spent in each operation per backward pass:")
        @parameter
        fn loop[idx: Int]():
            if self.backward_durations[idx] > 0:
                let label: String = self.operation_labels[idx].load(0)
                print(self.operation_labels[idx].load(0), Float32(self.backward_durations[idx]) / average_ms_divisor, "ms", " (", 100.0 * Float32(self.backward_durations[idx])/summed_duration,"% )")
        unroll[34, loop]()
        

