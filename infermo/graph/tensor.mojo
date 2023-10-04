
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
    var skips: Pointer[Int]
    var data: DTypePointer[DType.float32]
    var gradient: DTypePointer[DType.float32] 
    var velocity: DTypePointer[DType.float32] 
    var parents: Pointer[Int]
    var num_parents: Int
    var name: StringRef
    var inTensors: Bool
    var visited: Bool
    var requiresGradient: Bool

    fn __init__(_shape: DynamicVector[Int]) -> Self:
        let _num_dims = len(_shape)
        var _cap = _shape[0]
        for i in range(1,_num_dims):
            _cap *= _shape[i]

        let shape = Pointer[Int].alloc(_num_dims)
        for i in range(_num_dims):
            shape.store(i,_shape[i])

        let skips = Pointer[Int].alloc(_num_dims)
        memset_zero(skips,_num_dims)
        skips.store(_num_dims-1,1)
        for i in range(_num_dims-1):
            skips.store(_num_dims - i - 2, skips.load(_num_dims - i - 1) * _shape[_num_dims - i - 1])

        let data = DTypePointer[DType.float32].alloc(_cap)
        memset_zero(data, _cap)

        let gradient = DTypePointer[DType.float32].alloc(_cap)
        memset_zero(gradient, _cap)
    	
        let velocity = DTypePointer[DType.float32].alloc(_cap)
        memset_zero(velocity, _cap)

        let parents = Pointer[Int].alloc(64)
        memset_zero(parents, 64)
        let num_parents = 0 

        let name = StringRef('none')

        return Tensor{
            id: 1,
            name: name,
            num_dims: _num_dims,
            cap: _cap,
            shape: shape,
            skips: skips,
            data: data,
            gradient: gradient,
            velocity: velocity,
            parents: parents,
            num_parents: num_parents,
            inTensors: False,
            visited: False,
            requiresGradient: True
        }

    @always_inline
    fn copyFrom(inout self, borrowed other: Tensor):
        self.id = other.id
        self.name = other.name
        self.num_dims = other.num_dims
        self.cap = other.cap
        self.skips = other.skips
        self.data = other.data
        self.gradient = other.gradient
        self.parents = other.parents
        self.num_parents = other.num_parents
        self.inTensors = other.inTensors
        self.visited = other.visited
        self.requiresGradient = other.requiresGradient

    @always_inline
    fn getId(self) -> Int:
        return self.id

    @always_inline
    fn setId(inout self: Self, newId: Int):
        self.id = newId

    @always_inline
    fn printId(self):
        print(self.id)

    @always_inline
    fn getName(self) -> StringRef:
        return self.name

    @always_inline
    fn setName(inout self, newName: StringRef):
        self.name = newName

    @always_inline
    fn printName(self):
        print(self.name)

    @always_inline
    fn getNum_dims(self) -> Int:
        return self.num_dims

    @always_inline
    fn printNum_dims(self):
        print(self.num_dims)

    @always_inline
    fn getCap(self) -> Int:
        return self.cap

    @always_inline
    fn printCap(self):
        print(self.cap)

    @always_inline
    fn getShape(self, index: Int) -> Int:
        return self.shape.load(index)

    @always_inline
    fn printShape(self):
        print_no_newline("[ ")
        let len = self.getNum_dims()
        for i in range(len):
            print_no_newline(self.shape.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn getSkips(self, index: Int) -> Int:
        return self.skips.load(index)

    @always_inline
    fn printSkips(self):
        print_no_newline("[ ")
        let len = self.getNum_dims()
        for i in range(len):
            print_no_newline(self.skips.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn setInTensors(inout self: Self, val: Bool):
        self.inTensors = val

    @always_inline
    fn getInTensors(self) -> Bool:
        return self.inTensors

    @always_inline
    fn setVisited(inout self: Self, val: Bool):
        self.visited = val

    @always_inline
    fn getVisited(self) -> Bool:
        return self.visited

    @always_inline
    fn loadData(self, index: Int) -> Float32:
        return self.data.load(index)

    @always_inline
    fn setDataAll(self, val: Float32):
        if(val == 0):
            memset_zero(self.data,self.getCap())
        else:
            for i in range(self.getCap()):
                self.data.store(i,val)

    @always_inline
    fn setData(self, val: DTypePointer[DType.float32]):
        memcpy(self.data, val, self.getCap())

    @always_inline
    fn setData(self, index: Int, val: Float32):
        self.data.store(index, val)

    fn initRandom(self, min: Float32, max: Float32):
        seed()
        rand(self.data, self.cap)
        for i in range(self.cap):
            self.setData(i, self.getData(i) * (max - min) + min)

    fn initRandomHe(self):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap) 
        let u2 = DTypePointer[DType.float32].alloc(self.cap) 
        rand(u1, self.cap)
        rand(u2, self.cap)
        for i in range(self.cap):
            let z = sqrt(-Float32(2.0) * log(u1.load(i))) * cos(Float32(2.0) * pi * u2.load(i))
            let sigma = sqrt( Float32(2.0) / self.shape[self.num_dims-1]) 
            self.setData(i, z * sigma)


    @always_inline
    fn getData(self, index: Int) -> Float32:
        return self.data.load(index)

    @always_inline
    fn getData(self) -> DTypePointer[DType.float32]:
        return self.data    

    @always_inline
    fn setData(self, pos: DynamicVector[Int], val: Float32):
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        self.data.store(index, val)

    @always_inline
    fn setData(self, _pos: Vec, val: Float32):
        let pos = _pos.get()
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        self.data.store(index, val)

    @always_inline
    fn getData(self, _pos: Vec) -> Float32:
        let pos = _pos.get()
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        return self.data.load(index)

    @always_inline
    fn getData(self, *_pos: Int) -> Float32:
        let pos = VariadicList[Int](_pos)
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        return self.data.load(index)


    @always_inline
    fn printData(self):
        let num_dims = self.getNum_dims()
        let row: Int = self.getShape(num_dims-2)
        let cols: Int = self.getShape(num_dims-1)
        let col_skips: Int = (self.getSkips(0) * self.getShape(0)) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_skips):
            if(col_skips > 6 and i > 2 and i < col_skips - 3):
                if(i == 3):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-1):
                    if(cols * i % self.getSkips(d) == 0):
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
                        print_no_newline(self.loadData(idx))
                        if(j != cols-1):
                            print_no_newline(', ')

                for d in range(num_dims-2,-1,-1):
                    if(cols * (i + 1) % self.getSkips(d) == 0):
                        print_no_newline(" ]")

                if(i < col_skips-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.getShape(i))
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], Data>\n\n")  

    @always_inline
    fn setGradientAll(self, val: Float32):
        if(val == 0):
            memset_zero(self.gradient,self.getCap())
        else:
            for i in range(self.getCap()):
                self.gradient.store(i,val)

    @always_inline
    fn setGradient(self, val: DTypePointer[DType.float32]):
        memcpy(self.gradient, val, self.getCap())

    @always_inline
    fn setGradient(self, index: Int, val: Float32):
        self.gradient.store(index, val)

    @always_inline
    fn getGradient(self, index: Int) -> Float32:
        return self.gradient.load(index)

    @always_inline
    fn getGradient(self) -> DTypePointer[DType.float32]:
        return self.gradient  

    @always_inline
    fn printGradient(self):
        let num_dims = self.getNum_dims()
        let cols: Int = self.getShape(num_dims-1)
        let col_skips: Int = (self.getSkips(0) * self.getShape(0)) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_skips):
            if(col_skips > 6 and i > 2 and i < col_skips - 3):
                if(i == 3):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-1):
                    if(cols * i % self.getSkips(d) == 0):
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
                        print_no_newline(self.getGradient(idx))
                        if(j != cols-1):
                            print_no_newline(', ')

                for d in range(num_dims-2,-1,-1):
                    if(cols * (i + 1) % self.getSkips(d) == 0):
                        print_no_newline(" ]")

                if(i < col_skips-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.getShape(i))
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], Gradient>\n\n")  

    @always_inline
    fn setvelocityAll(self, val: Float32):
        if(val == 0):
            memset_zero(self.velocity,self.getCap())
        else:
            for i in range(self.getCap()):
                self.velocity.store(i,val)

    @always_inline
    fn setVelocity(self, val: DTypePointer[DType.float32]):
        memcpy(self.velocity, val, self.getCap())

    @always_inline
    fn setVelocity(self, index: Int, val: Float32):
        self.velocity.store(index, val)

    @always_inline
    fn getVelocity(self, index: Int) -> Float32:
        return self.velocity.load(index)

    @always_inline
    fn getVelocity(self) -> DTypePointer[DType.float32]:
        return self.velocity    

    @always_inline
    fn setVelocity(self, pos: DynamicVector[Int], val: Float32):
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        self.velocity.store(index, val)

    @always_inline
    fn setVelocity(self, _pos: Vec, val: Float32):
        let pos = _pos.get()
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        self.velocity.store(index, val)

    @always_inline
    fn getVelocity(self, _pos: Vec) -> Float32:
        let pos = _pos.get()
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        return self.velocity.load(index)

    @always_inline
    fn getVelocity(self, *_pos: Int) -> Float32:
        let pos = VariadicList[Int](_pos)
        let len = len(pos)
        var index = 0
        for j in range(len):
            index += self.skips[j] * pos[j]

        return self.velocity.load(index)


    @always_inline
    fn printVelocity(self):
        let num_dims = self.getNum_dims()
        let row: Int = self.getShape(num_dims-2)
        let cols: Int = self.getShape(num_dims-1)
        let col_skips: Int = (self.getSkips(0) * self.getShape(0)) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_skips):
            if(col_skips > 6 and i > 2 and i < col_skips - 3):
                if(i == 3):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-1):
                    if(cols * i % self.getSkips(d) == 0):
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
                        print_no_newline(self.getVelocity(idx))
                        if(j != cols-1):
                            print_no_newline(', ')

                for d in range(num_dims-2,-1,-1):
                    if(cols * (i + 1) % self.getSkips(d) == 0):
                        print_no_newline(" ]")

                if(i < col_skips-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.getShape(i))
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], velocity>\n\n")  

    @always_inline
    fn getNum_parents(self) -> Int:
        return self.num_parents

    @always_inline
    fn setNum_parents(inout self, val: Int):
        self.num_parents = val                

    @always_inline
    fn addParent(inout self, parentId: Int):
        let index = self.getNum_parents()
        self.parents.store(index, parentId)
        self.num_parents += 1

    @always_inline
    fn getParent(self, index: Int) -> Int:
        return self.parents.load(index)

    @always_inline
    fn printParents(self):
        print_no_newline("[ ")
        let len = self.getNum_parents()
        for i in range(len):
            print_no_newline(self.parents.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn setRequiresGradient(inout self, val: Bool):
        self.requiresGradient = val

    @always_inline
    fn getRequiresGradient(self) -> Bool:
        return self.requiresGradient


    
    @always_inline
    fn printArgMax(self):
        let num_dims = self.getNum_dims()
        let row: Int = self.getShape(num_dims-2)
        let cols: Int = self.getShape(num_dims-1)
        let col_skips: Int = (self.getSkips(0) * self.getShape(0)) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_skips):
            if(col_skips > 6 and i > 2 and i < col_skips - 3):
                if(i == 3):
                    print("                 ... ")
                continue
            else:
                if(i > 0):
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(num_dims-2):
                    if(cols * i % self.getSkips(d) == 0):
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
                    if(self.loadData(idx) > max):
                        max = self.loadData(idx)
                        max_idx = max_counter                   
                print_no_newline(max_idx)
                for d in range(num_dims-2,0,-1):
                    if(cols * (i + 1) % self.getSkips(d) == 0):
                        print_no_newline(" ]")

                if(i < col_skips-1):
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(num_dims):
                        print_no_newline(self.getShape(i))
                        if(i < num_dims-1):
                            print_no_newline(",")                        
                    print_no_newline("], Data>\n\n")  