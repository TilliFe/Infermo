
from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed, randint
from math import sin, cos, log, sqrt, exp

from ..helpers.shape import shape, Vec

@register_passable("trivial")
struct Tensor:
    var id: Int
    var num_dims: Int
    var cap: Int
    var shape: Pointer[Int]
    var strides: Pointer[Int]
    var data: DTypePointer[DType.float32]
    var grad: DTypePointer[DType.float32] 
    var velocity: DTypePointer[DType.float32] 
    var parents: Pointer[Int]
    var num_parents: Int
    var name: StringRef
    var in_tensors: Bool
    var visited: Bool
    var requires_grad: Bool
    var other_params: Pointer[Int]

    fn __init__(_shape: DynamicVector[Int]) -> Self:
        let _num_dims = len(_shape)
        var _cap = _shape[0]
        for i in range(1,_num_dims):
            _cap *= _shape[i]

        let shape = Pointer[Int].alloc(_num_dims)
        for i in range(_num_dims):
            shape.store(i,_shape[i])

        let strides = Pointer[Int].alloc(_num_dims)
        memset_zero(strides,_num_dims)
        strides.store(_num_dims-1,1)
        for i in range(_num_dims-1):
            strides.store(_num_dims - i - 2, strides.load(_num_dims - i - 1) * _shape[_num_dims - i - 1])

        let data = DTypePointer[DType.float32].alloc(_cap)
        memset_zero(data, _cap)

        let grad = DTypePointer[DType.float32].alloc(_cap)
        memset_zero(grad, _cap)
    	
        let velocity = DTypePointer[DType.float32].alloc(_cap)
        memset_zero(velocity, _cap)

        let parents = Pointer[Int].alloc(64)
        memset_zero(parents, 64)
        let num_parents = 0 

        let name = StringRef('none')

        let other_params = Pointer[Int].alloc(64)
        memset_zero(other_params, 64)

        return Tensor{
            id: 1,
            name: name,
            num_dims: _num_dims,
            cap: _cap,
            shape: shape,
            strides: strides,
            data: data,
            grad: grad,
            velocity: velocity,
            parents: parents,
            num_parents: num_parents,
            in_tensors: False,
            visited: False,
            requires_grad: True,
            other_params: other_params
        }

    @always_inline
    fn getId(self) -> Int:
        return self.id

    @always_inline
    fn set_id(inout self: Self, newId: Int):
        self.id = newId

    @always_inline
    fn set_name(inout self, newName: StringRef):
        self.name = newName

    @always_inline
    fn print_shape(self):
        print_no_newline("[ ")
        let len = self.num_dims
        for i in range(len):
            print_no_newline(self.shape.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn print_strides(self):
        print_no_newline("[ ")
        let len = self.num_dims
        for i in range(len):
            print_no_newline(self.strides.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn set_visited(inout self: Self, val: Bool):
        self.visited = val

    @always_inline
    fn fill(self, val: Float32):
        if(val == 0):
            memset_zero(self.data,self.cap)
        else:
            for i in range(self.cap):
                self.data.store(i,val)

    @always_inline
    fn set_data(self, val: DTypePointer[DType.float32]):
        memcpy(self.data, val, self.cap)

    @always_inline
    fn set_data(self, index: Int, val: Float32):
        self.data.store(index, val)

    fn randu(self, min: Float32, max: Float32):
        seed()
        rand(self.data, self.cap)
        for i in range(self.cap):
            self.set_data(i, self.data.load(i) * (max - min) + min)

    fn randHe(self):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap) 
        let u2 = DTypePointer[DType.float32].alloc(self.cap) 
        rand(u1, self.cap)
        rand(u2, self.cap)
        for i in range(self.cap):
            let z = sqrt(-Float32(2.0) * log(u1.load(i))) * cos(Float32(2.0) * pi * u2.load(i))
            let sigma = sqrt( Float32(2.0) / self.shape[self.num_dims-1]) 
            self.set_data(i, z * sigma)

    fn randn(self, std: Float32 = Float32(1.0), mu: Float32 = Float32(0.0)):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap) 
        let u2 = DTypePointer[DType.float32].alloc(self.cap) 
        rand(u1, self.cap)
        rand(u2, self.cap)
        for i in range(self.cap):
            let z = sqrt(-Float32(2.0) * log(u1.load(i))) * cos(Float32(2.0) * pi * u2.load(i))
            self.set_data(i, z * std + mu) 

    @always_inline
    fn set_data(self, pos: DynamicVector[Int], val: Float32):
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.strides[j] * pos[j]

        self.data.store(index, val)

    @always_inline
    fn set_data(self, _pos: Vec, val: Float32):
        let pos = _pos.get()
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.strides[j] * pos[j]

        self.data.store(index, val)

    @always_inline
    fn print_data(self):
        let num_dims = self.num_dims
        let row: Int = self.shape[num_dims-2]
        let cols: Int = self.shape[num_dims-1]
        let col_strides: Int = (self.strides[0] * self.shape[0]) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_strides):
            if(col_strides > 10 and i > 4 and i < col_strides - 5):
                if(i == 5):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-1):
                    if(cols * i % self.strides[d] == 0):
                        print_no_newline("[ ")
                        indent += 1
                    else:
                        print_no_newline("  ")

                for j in range(cols):
                    if(cols > 10 and j >= 3 and j < cols-3):
                        if(j == 3):
                            print_no_newline("... , ")
                        continue
                    else:
                        let idx = cols * i + j
                        print_no_newline(self.data.load(idx))
                        if(j != cols-1):
                            print_no_newline(', ')

                for d in range(num_dims-2,-1,-1):
                    if(cols * (i + 1) % self.strides[d] == 0):
                        print_no_newline(" ]")

                if(i < col_strides-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.shape[i])
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], Data>\n\n")  

    @always_inline
    fn fill_grad(self, val: Float32):
        if(val == 0):
            memset_zero(self.grad,self.cap)
        else:
            for i in range(self.cap):
                self.grad.store(i,val)

    @always_inline
    fn set_grad(self, val: DTypePointer[DType.float32]):
        memcpy(self.grad, val, self.cap)

    @always_inline
    fn set_grad(self, index: Int, val: Float32):
        self.grad.store(index, val)

    @always_inline
    fn print_grad(self):
        let num_dims = self.num_dims
        let cols: Int = self.shape[num_dims-1]
        let col_strides: Int = (self.strides[0] * self.shape[0]) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_strides):
            if(col_strides > 10 and i > 4 and i < col_strides - 5):
                if(i == 5):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-1):
                    if(cols * i % self.strides[d] == 0):
                        print_no_newline("[ ")
                        indent += 1
                    else:
                        print_no_newline("  ")

                for j in range(cols):
                    if(cols > 10 and j >= 3 and j < cols-3):
                        if(j == 3):
                            print_no_newline("... , ")
                        continue
                    else:
                        let idx = cols * i + j
                        print_no_newline(self.grad.load(idx))
                        if(j != cols-1):
                            print_no_newline(', ')

                for d in range(num_dims-2,-1,-1):
                    if(cols * (i + 1) % self.strides[d] == 0):
                        print_no_newline(" ]")

                if(i < col_strides-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.shape[i])
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], Gradient>\n\n")  

    @always_inline
    fn fill_velocity(self, val: Float32):
        if(val == 0):
            memset_zero(self.velocity,self.cap)
        else:
            for i in range(self.cap):
                self.velocity.store(i,val)

    @always_inline
    fn set_velocity(self, val: DTypePointer[DType.float32]):
        memcpy(self.velocity, val, self.cap)

    @always_inline
    fn set_velocity(self, index: Int, val: Float32):
        self.velocity.store(index, val)

    @always_inline
    fn set_velocity(self, pos: DynamicVector[Int], val: Float32):
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.strides[j] * pos[j]

        self.velocity.store(index, val)

    @always_inline
    fn set_velocity(self, _pos: Vec, val: Float32):
        let pos = _pos.get()
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.strides[j] * pos[j]

        self.velocity.store(index, val)

    @always_inline
    fn print_velocity(self):
        let num_dims = self.num_dims
        let row: Int = self.shape[num_dims-2]
        let cols: Int = self.shape[num_dims-1]
        let col_strides: Int = (self.strides[0] * self.shape[0]) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_strides):
            if(col_strides > 10 and i > 4 and i < col_strides - 5):
                if(i == 5):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-1):
                    if(cols * i % self.strides[d] == 0):
                        print_no_newline("[ ")
                        indent += 1
                    else:
                        print_no_newline("  ")

                for j in range(cols):
                    if(cols > 10 and j >= 3 and j < cols-3):
                        if(j == 3):
                            print_no_newline("... , ")
                        continue
                    else:
                        let idx = cols * i + j
                        print_no_newline(self.velocity.load(idx))
                        if(j != cols-1):
                            print_no_newline(', ')

                for d in range(num_dims-2,-1,-1):
                    if(cols * (i + 1) % self.strides[d] == 0):
                        print_no_newline(" ]")

                if(i < col_strides-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.shape[i])
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], velocity>\n\n")              

    @always_inline
    fn add_parent(inout self, parentId: Int):
        let index = self.num_parents
        self.parents.store(index, parentId)
        self.num_parents += 1

    @always_inline
    fn get_parent(self, index: Int) -> Int:
        return self.parents.load(index)

    @always_inline
    fn print_parents(self):
        print_no_newline("[ ")
        let len = self.num_parents
        for i in range(len):
            print_no_newline(self.parents.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    
    @always_inline
    fn print_arg_max(self):
        let num_dims = self.num_dims
        let row: Int = self.shape[num_dims-2]
        let cols: Int = self.shape[num_dims-1]
        let col_strides: Int = (self.strides[0] * self.shape[0]) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_strides):
            if(col_strides > 10 and i > 4 and i < col_strides - 5):
                if(i == 5):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-2):
                    if(cols * i % self.strides[d] == 0):
                        print_no_newline("[ ")
                        indent += 1
                    else:
                        print_no_newline("  ")

                var max : Float32 = 0
                var max_counter: Float32 = 0
                var max_idx: Float32 = 0
                for j in range(cols):
                    let idx = cols * i + j
                    max_counter += Float32(1)
                    if(self.data.load(idx) > max):
                        max = self.data.load(idx)
                        max_idx = max_counter                   
                print_no_newline(max_idx)
                for d in range(num_dims-2,0,-1):
                    if(cols * (i + 1) % self.strides[d] == 0):
                        print_no_newline(" ]")

                if(i < col_strides-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.shape[i])
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], Data>\n\n")  