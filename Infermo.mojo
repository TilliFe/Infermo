from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from memory import memset_zero, memcpy
from random import rand
from runtime.llcl import Runtime
from algorithm import vectorize, parallelize
from random import rand, random_si64, seed
from math import sin, cos, log, sqrt


alias nelts = simdwidthof[DType.float32]()

@always_inline
fn mul(inout C: Tensor, A: Tensor, B: Tensor, rt: Runtime):
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

        @parameter
        fn calc_row(m: Int):
            for k in range(K):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.data.simd_store[nelts](offset_C + m*N+n, C.data.simd_load[nelts](offset_C + m*N+n) + A.data.load(offset_A + m*K+k) * B.data.simd_load[nelts](offset_B + k*N+n))
                vectorize[nelts, dot](N)
        parallelize[calc_row](rt, M)

@always_inline
fn add(inout C: Tensor, A: Tensor, B: Tensor):
    let num_dims = A.getNum_dims()
    var matrix_size = A.getShape(num_dims-2) * A.getShape(num_dims-1)
    if(num_dims >= 3):
        matrix_size = A.getSkips(num_dims-3)

    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)

    if(C.getCap() > nelts):
        for i in range(0,C.getCap() - (nelts), nelts):
            C.data.simd_store[nelts](i, A.data.simd_load[nelts](i) + B.data.simd_load[nelts](i))
        for i in range(C.getCap() - nelts, C.getCap()):
            C.data.store(i, A.data.load(i) + B.data.load(i))
    else:
        for i in range(C.getCap()):
            C.data.store(i, A.data.load(i) + B.data.load(i))

@always_inline
fn sum(inout B: Tensor, A: Tensor):
    var sum: Float32 = 0
    for i in range(A.getCap()):
        sum += A.getData(i)
    B.setData(0,sum)

@always_inline
fn ReLU(inout B: Tensor, A: Tensor):
    for i in range(A.getCap()):
        let val = A.getData(i)
        if(val < 0):
            B.setData(i,0)
        else:
            B.setData(i,val)

@always_inline
fn MSE(inout C: Tensor, A: Tensor, B: Tensor):
    for index in range(A.getCap()):
        let error = (A.getData(index) - B.getData(index)) * (A.getData(index) - B.getData(index))
        C.setData(0, C.getData(0) + error)
    C.setData(0, C.getData(0) / A.getCap())

@always_inline
fn reshape(inout B: Tensor, A: Tensor):
    return


@always_inline
fn mul_grad(C: Tensor, inout A: Tensor, inout B: Tensor, rt: Runtime):

    let num_dims = A.getNum_dims()
    let A_matrix_size = A.shape[num_dims-2] * A.shape[num_dims-1]
    let B_matrix_size = B.shape[num_dims-2] * B.shape[num_dims-1]
    let C_matrix_size = C.shape[num_dims-2] * C.shape[num_dims-1] 

    let M = A.shape[num_dims-2]
    let K = B.shape[num_dims-2]  
    let N = B.shape[num_dims-1]

    for s in range(A.getCap() // A_matrix_size):
        let offset_C = s * C_matrix_size
        let offset_B = s * B_matrix_size
        let offset_A = s * A_matrix_size 

        if (A.getRequiresGradient()):

            # for m in range(M):
            #     for k in range(K):
            #         for n in range(N):
            #             let index_A = offset_A + m * K + k
            #             let index_C = offset_C + m * N + n
            #             let index_B = offset_B + k * N + n
            #             let a = A.getGradient(index_A) + C.getGradient(index_C) * B.getData(index_B) 
            #             A.setGradient(index_A, a) 
            
            @parameter
            fn calc_row_1(m: Int):
                for n in range(N):
                    @parameter
                    fn dot[nelts: Int](k: Int):
                        let index_A = offset_A + m * K + k
                        let index_C = offset_C + m * N + n
                        let index_B = offset_B + k * N + n
                        let a = A.getGradient(index_A) + C.getGradient(index_C) * B.getData(index_B) 
                        A.setGradient(index_A, a) 
                    vectorize[nelts, dot](K)
            parallelize[calc_row_1](rt, M)

        # if (B.getRequiresGradient()): 
        #     for k in range(K):
        #         for n in range(N): 
        #             for m in range(M): 
        #                 let index_B = offset_B + k * N + n
        #                 let index_A = offset_A + m * K + k
        #                 let index_C = offset_C + m * N + n
        #                 let b = B.getGradient(index_B) + A.getData(index_A) * C.getGradient(index_C)  
        #                 B.setGradient(index_B, b) 

            @parameter
            fn calc_row_2(k: Int):
                for m in range(M):
                    @parameter
                    fn dot[nelts: Int](n: Int):
                        let index_B = offset_B + k * N + n
                        let index_A = offset_A + m * K + k
                        let index_C = offset_C + m * N + n
                        let b = B.getGradient(index_B) + A.getData(index_A) * C.getGradient(index_C)  
                        B.setGradient(index_B, b) 
                    vectorize[nelts, dot](N)
            parallelize[calc_row_2](rt, K)

@always_inline        
fn add_grad(C: Tensor, inout A: Tensor, inout B: Tensor):
    A.setGradient(C.getGradient())
    B.setGradient(C.getGradient())

@always_inline
fn sum_grad(B: Tensor, inout A: Tensor):
    A.setGradientAll(1)

@always_inline
fn ReLU_grad(B: Tensor, inout A: Tensor):
    A.setGradient(B.getGradient())
    for i in range(A.getCap()):
        let val = A.getData(i)
        if val < 0:
            A.setGradient(i, 0)

@always_inline
fn MSE_grad(C: Tensor, inout A: Tensor, inout B: Tensor): # A: TrueVals, B: Logits
    let num_dims = A.getNum_dims()
    let M = A.getShape(num_dims-2)
    let N = A.getShape(num_dims-1)
    var matrix_size = M*N
    if(num_dims >= 3):
        matrix_size = A.getSkips(num_dims-3)

    for index in range(A.getCap()):
        let grad = Float32(2) * (B.getData(index) - A.getData(index)) / A.getCap()
        A.setGradient(index, grad) 
        B.setGradient(index, grad) 

@always_inline
fn reshape_grad(B: Tensor, inout A: Tensor):
    A.setGradient(B.getGradient())



struct Vec:
    var shape: DynamicVector[Int]

    fn __init__(inout self, *_shape: Int):
        let v = VariadicList[Int](_shape)
        let len = len(v)
        self.shape = DynamicVector[Int](0)
        for i in range(len):
            self.shape.push_back(v[i])

    fn get(self) -> DynamicVector[Int]:
        return self.shape

fn shape(*_shape: Int) -> DynamicVector[Int]:
    let v = VariadicList[Int](_shape)
    let len = len(v)
    var shape = DynamicVector[Int](0)
    if(len == 1):
        shape.push_back(1)
    for i in range(len):
        shape.push_back(v[i])
    return shape



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



struct Module:
    var Tensors: DynamicVector[Tensor]
    var counter: Int
    var forwardTape: DynamicVector[Int]
    var backwardTape: DynamicVector[Int]

    fn __init__(inout self):
        self.Tensors = DynamicVector[Tensor](0)
        self.counter = 0
        self.forwardTape = DynamicVector[Int]()
        self.backwardTape = DynamicVector[Int]()

    @always_inline
    fn addForward(inout self, TensorId: Int):
        self.forwardTape.push_back(TensorId)

    @always_inline
    fn addBackward(inout self, TensorId: Int):
        self.backwardTape.push_back(TensorId)

    @always_inline
    fn addTensor(inout self, inout a: Tensor):
        a.setId(self.counter)
        a.setInTensors(True)
        self.counter += 1
        self.Tensors.push_back(a)

    @always_inline
    fn tensor(inout self, *s: Int) -> Tensor:
        let v = VariadicList[Int](s)
        let len = len(v)
        var shape = DynamicVector[Int](0)
        for i in range(len):
            shape.push_back(v[i])

        var newTensor = Tensor(shape)

        return newTensor 

    @always_inline
    fn printTensor(self, index: Int):
        self.Tensors[index].printData()

    @always_inline
    fn getTensor(inout self, index: Int) -> Tensor:
        return self.Tensors[index]

    @always_inline
    fn getCounter(self) -> Int:
        return self.counter

    @always_inline
    fn printForwardTape(self):
        print_no_newline("[ ")
        let len = len(self.forwardTape)
        for i in range(len):
            print_no_newline(self.forwardTape[i])
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")
    
    @always_inline
    fn printBackwardTape(self):
        print_no_newline("[ ")
        let len = len(self.backwardTape)
        for i in range(len):
            print_no_newline(self.backwardTape[i])
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn mul(inout self, inout A: Tensor, inout B: Tensor) -> Tensor:

        # check dimensions
        if(A.getNum_dims() != B.getNum_dims()):
            print("Error (a mul): number of dimensions are not equal")
        if(A.getNum_dims() < 2):
            print("Error (at mul): Tensors must be of shape at least 2, in order to perform matrix multiplication")
        let num_dims = A.getNum_dims()
        if(A.getShape(num_dims-1) != B.getShape(num_dims-2)):
            print("Error (at mul): For Matrix Multiplication, Matrices need to in the following shape: C[mxn] = A[mxk] * B[kxn]")

        # init result Tensor 
        var new_shape = DynamicVector[Int](0)
        for i in range(num_dims-1):
            new_shape.push_back(A.shape[i])
        new_shape.push_back(B.shape[num_dims-1])
        var C = Tensor(new_shape)

        C.setName('mul')

        if(not A.getInTensors()):
            C.addParent(self.counter)
            self.addTensor(A)
        else:
            C.addParent(A.getId())

        if(not B.getInTensors()):
            C.addParent(self.counter)
            self.addTensor(B)
        else:
            C.addParent(B.getId())
        self.addTensor(C)

        return C 
        
    @always_inline
    fn add(inout self, inout A: Tensor, inout B: Tensor) -> Tensor:

        # check dimensions
        if(A.getNum_dims() != B.getNum_dims()):
            print("Error (at add): number of dimensions are not equal")
        let num_dims = A.getNum_dims()
        if(A.getShape(num_dims-2) != B.getShape(num_dims-2) or A.getShape(num_dims-1) != B.getShape(num_dims-1)):
            print("Error (at add): For Matrix ADdition, Matrices need to in the following shape: C[mxn] = A[mxn] + B[mxn]")

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(A.getShape(i))
        var C = Tensor(new_shape)

        C.setName('add')

        if(not A.getInTensors()):
            C.addParent(self.counter)
            self.addTensor(A)
        else:
            C.addParent(A.getId())

        if(not B.getInTensors()):
            C.addParent(self.counter)
            self.addTensor(B)
        else:
            C.addParent(B.getId())
        self.addTensor(C)

        return C 

    @always_inline
    fn ReLU(inout self, inout A: Tensor) -> Tensor: 
        var new_shape = DynamicVector[Int]()
        for i in range(A.getNum_dims()):
            new_shape.push_back(A.getShape(i))

        var B = Tensor(new_shape)

        B.setName('ReLU')

        if(not A.getInTensors()):
            B.addParent(self.counter)
            self.addTensor(A)
        else:
            B.addParent(A.getId())
        self.addTensor(B)

        return B


    @always_inline
    fn sum(inout self, inout A: Tensor) -> Tensor: 

        var B = Tensor(shape(1,1))

        B.setName('sum')

        if(not A.getInTensors()):
            B.addParent(self.counter)
            self.addTensor(A)
        else:
            B.addParent(A.getId())
        self.addTensor(B)

        return B

    @always_inline
    fn MSE(inout self, inout A: Tensor, inout B: Tensor) -> Tensor:

        # check dimensions
        if(A.getNum_dims() != B.getNum_dims()):
            print("Error (at MSE): number of dimensions are not equal")
        let num_dims = A.getNum_dims()
        if(A.getShape(num_dims-2) != B.getShape(num_dims-2) or A.getShape(num_dims-1) != B.getShape(num_dims-1)):
            print("Error (at MSE): For MSE computation, Matrices need to in the following shape: C[mxn] = (A[mxn] - B[mxn])^2")

        # init result Tensor 
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(A.getShape(i))
        var C = Tensor(shape(1))

        C.setName('MSE')

        if(not A.getInTensors()):
            C.addParent(self.counter)
            self.addTensor(A)
        else:
            C.addParent(A.getId())

        if(not B.getInTensors()):
            C.addParent(self.counter)
            self.addTensor(B)
        else:
            C.addParent(B.getId())
        self.addTensor(C)

        return C 

    @always_inline
    fn reshape(inout self, inout A: Tensor, dir: Int) -> Tensor: 
        if(dir != -1):
            print("Error (at reshape)!")
        var new_shape = DynamicVector[Int](0)
        new_shape.push_back(A.getShape(0))
        new_shape.push_back(1)
        for i in range(1, A.getNum_dims()):
            new_shape[1] *= A.getShape(i)
        var B = Tensor(new_shape)

        B.setName('reshape')

        if(not A.getInTensors()):
            B.addParent(self.counter)
            self.addTensor(A)
        else:
            B.addParent(A.getId())
        self.addTensor(B)

        return B

    @always_inline
    fn reshape(inout self, inout A: Tensor, newShape: DynamicVector[Int]) -> Tensor: 
        var B = Tensor(newShape)
        B.setData(A.getData())

        B.setName('reshape')

        if(not A.getInTensors()):
            B.addParent(self.counter)
            self.addTensor(A)
        else:
            B.addParent(A.getId())
        self.addTensor(B)

        if(A.getCap() != B.getCap()):
            print("Error (at reshape): The Product of the dimensions in shapes must be the same!")

        return B


    fn topOrder(inout self, inout Tensor: Tensor):  
        if not Tensor.getVisited():
            for i in range(Tensor.getNum_parents()):
                let nextTensorId = Tensor.getParent(i)
                var nextTensor = self.Tensors[nextTensorId]
                self.topOrder(nextTensor)
            self.forwardTape.push_back(Tensor.getId())
            Tensor.setVisited(True)

    @always_inline
    fn forward(inout self, inout computingTensor: Tensor):
        for i in range(self.counter):
            self.Tensors[i].setVisited(False)
            if(self.Tensors[i].getName() != 'none'):
                self.Tensors[i].setDataAll(0)
        self.forwardTape = DynamicVector[Int]()
        self.topOrder(computingTensor)

        for i in range(self.counter):
            var curr = self.Tensors[i]
            if(curr.getName() == 'mul'):
                let par1 = self.Tensors[curr.getParent(0)]
                let par2 = self.Tensors[curr.getParent(1)]
                with Runtime() as rt:
                    mul(curr,par1,par2,rt)
            if(curr.getName() == 'add'):
                let par1 = self.Tensors[curr.getParent(0)]
                let par2 = self.Tensors[curr.getParent(1)]
                add(curr,par1,par2)
            if(curr.getName() == 'sum'):
                let par1 = self.Tensors[curr.getParent(0)]
                sum(curr,par1)
            if(curr.getName() == 'ReLU'):
                let par1 = self.Tensors[curr.getParent(0)]
                ReLU(curr,par1) 
            if(curr.getName() == 'MSE'):
                let par1 = self.Tensors[curr.getParent(0)]
                let par2 = self.Tensors[curr.getParent(1)]
                MSE(curr,par1,par2) 
            if(curr.getName() == 'reshape'):
                let par1 = self.Tensors[curr.getParent(0)]
                reshape(curr,par1)

    fn backwardOrder(inout self, Tensor: Tensor):
        self.backwardTape = DynamicVector[Int](0)
        self.backwardTape.push_back(Tensor.getId())
        var it = 0
        while(it < len(self.backwardTape)):
            let currId = self.backwardTape[it]
            let curr = self.Tensors[currId]
            for i in range(curr.getNum_parents()):
                let parId = curr.getParent(i)
                let par = self.Tensors[parId]
                if(par.getRequiresGradient()):
                    self.backwardTape.push_back(parId)
            it += 1

    @always_inline
    fn backward(inout self, inout lastTensor: Tensor):
        if(lastTensor.cap != 1):
            print("Error: Gradient can be implicitly created only for scalar outputs")
            return
        self.backwardOrder(lastTensor)
        for i in range(self.counter):
            if(self.Tensors[i].requiresGradient):
                self.Tensors[i].setGradientAll(0)

        for i in range(len(self.backwardTape)):
            let currId = self.backwardTape[i]
            let curr = self.Tensors[currId]
            if(curr.getName() == 'mul'):
                var par1 = self.Tensors[curr.getParent(0)]
                var par2 = self.Tensors[curr.getParent(1)]
                with Runtime() as rt:
                    mul_grad(curr,par1,par2,rt)
            if(curr.getName() == 'add'):
                var par1 = self.Tensors[curr.getParent(0)]
                var par2 = self.Tensors[curr.getParent(1)]
                add_grad(curr,par1,par2)
            if(curr.getName() == 'sum'):
                var par1 = self.Tensors[curr.getParent(0)]
                sum_grad(curr,par1)
            if(curr.getName() == 'ReLU'):
                var par1 = self.Tensors[curr.getParent(0)]
                ReLU_grad(curr,par1)
            if(curr.getName() == 'MSE'):
                var par1 = self.Tensors[curr.getParent(0)]
                var par2 = self.Tensors[curr.getParent(1)]
                MSE_grad(curr,par1,par2)
            if(curr.getName() == 'reshape'):
                var par1 = self.Tensors[curr.getParent(0)]
                reshape_grad(curr,par1)

    fn optimize(inout self, optType: String, lr: Float32 = 0.001, momentum: Float32 = 0.9):
        
        if(optType == "sgd"):
            for i in range(len(self.backwardTape)):
                let id = self.Tensors[self.backwardTape[i]].id
                for index in range(self.Tensors[id].getCap()):
                    self.Tensors[id].setData(index, self.Tensors[id].getData(index) - lr * self.Tensors[id].getGradient(index))
        
        if(optType == "sgd_momentum"):
            for i in range(len(self.backwardTape)):
                let id = self.Tensors[self.backwardTape[i]].id
                for index in range(self.Tensors[id].getCap()):
                    self.Tensors[id].setVelocity(index, momentum * self.Tensors[id].getVelocity(index) + lr * self.Tensors[id].getGradient(index))
                    self.Tensors[id].setData(index, self.Tensors[id].getData(index) - self.Tensors[id].getVelocity(index))


    @always_inline
    fn printTensors(self): 
        print("Printing all Tensors of the Computational Graph .....\n")
        for i in range(self.counter):
            let n = self.Tensors[i]
            print("Tensor ID: ", n.getId(), ", Name: ", n.getName(), ", rquiresGrad: ", n.getRequiresGradient(), ", cap = ", n.getCap())
            n.printData()
            n.printGradient()
        print("End of Printing all Tensors of the Computational Graph.")


# define one layer of an MLP
fn Linear(inout nn: Module, inout x: Tensor, num_neurons: Int, addBias : Bool = True, activation: String = 'ReLU') -> Tensor:
    let x_rows = x.getShape(x.num_dims - 2)
    let x_cols = x.getShape(x.num_dims - 1)
    var W = Tensor(shape(x_cols,num_neurons))

    W.initRandomHe()
    if(addBias):
        var ones = Tensor(shape(x_rows,1))
        var bias = Tensor(shape(1,num_neurons))
        ones.setDataAll(1)
        ones.requiresGradient = False
        bias.setDataAll(0.1)
        var xW = nn.mul(x,W)    
        var ob = nn.mul(ones,bias)
        x = nn.add(xW,ob)
        if(activation == 'ReLU'):
            x = nn.ReLU(x)
    else:
        x = nn.mul(x,W)
        if(activation == 'ReLU'):
            x = nn.ReLU(x)
    return x


# Simple introductory example ############################################################################

# define the model and its behaviour
struct model:
    var nn: Module
    var input: Tensor
    var trueVals: Tensor
    var logits: Tensor
    var loss: Tensor

    fn __init__(inout self):
        self.input = Tensor(shape(512,1))
        self.input.requiresGradient = False
        self.trueVals = Tensor(shape(512,1))
        self.trueVals.requiresGradient = False
        self.nn = Module()

        # define model architecture
        var x = Linear(self.nn,self.input, num_neurons=8, addBias=True, activation='ReLU')
        for i in range(2):
            x = Linear(self.nn,x, num_neurons=32, addBias=True, activation='ReLU')
        self.logits = Linear(self.nn,x,1,True,'none')
        self.loss = self.nn.MSE(self.trueVals,self.logits)

    @always_inline     
    fn forward(inout self, _input: DTypePointer[DType.float32], _trueVals: DTypePointer[DType.float32]) -> Tensor:
        self.nn.Tensors[0].setData(_input) # this is a bug, why cant we assign to self.input directly ? -> the id changes to two, dont know why
        self.trueVals.setData(_trueVals)
        self.nn.forward(self.logits)
        return self.logits

    @always_inline
    fn backward(inout self):
        self.nn.backward(self.loss)

    @always_inline
    fn step(inout self):
        self.nn.optimize('sgd_momentum', lr = 0.1, momentum = 0.9)


# Data Generator for a simple regression problem
struct DataGenerator:
    var size: Int
    var x: DTypePointer[DType.float32]
    var y: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.size = size
        self.x = DTypePointer[DType.float32].alloc(self.size)
        self.y = DTypePointer[DType.float32].alloc(self.size)

    @always_inline
    fn random(self, it: Int):
        seed(it)
        rand(self.x, self.size)
        let min = -1
        let max = 1
        for i in range(self.size):
            let x_rand = self.x.load(i) * (max - min) + min
            self.x.store(i, x_rand)
            let res = 0.5 + 0.5*sin(5*x_rand)
            self.y.store(i, res) 


# train the model
fn main():

    let dataset = DataGenerator(512)
    var model = model()
    let num_epochs = 1000

    var lossSum: Float32 = 0
    let every = 100

    for epoch in range(0,num_epochs):
        dataset.random(epoch)
        let logits = model.forward(dataset.x,dataset.y)
        model.backward()
        model.step()

        lossSum += model.loss.getData(0)
        if( epoch % every == 0 and epoch > 0):
            print("Epoch", epoch,", AvgLoss = ", lossSum / every)
            lossSum = 0      
            # logits.printData()
            # model.trueVals.printData()
            # model.nn.printTensors()


# fn main():
#     # init
#     var nn = Module()
#     var A = Tensor(shape(2,3))
#     var B = Tensor(shape(3,4))

#     # specify tensor entries
#     A.setDataAll(2)
#     B.setDataAll(3)

#     # perform computation and print result
#     var C = nn.mul(A,B)
#     var D = nn.sum(C)

#     # perform computation and print result
#     nn.forward(C)
#     C.printData()

#     # compute gradients of A and B
#     nn.backward(D)
#     A.printGradient()
#     B.printGradient()
