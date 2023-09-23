from memory import memset_zero, memcpy
from memory.unsafe import Pointer

@register_passable("trivial")
struct Node:
    var id: Int
    var num_dims: Int
    var cap: Int
    var shape: Pointer[Int]
    var skips: Pointer[Int]
    var data: DTypePointer[DType.float32]
    var gradient: DTypePointer[DType.float32]    
    var parents: Pointer[Int]
    var num_parents: Int
    var children: Pointer[Int]
    var num_children: Int
    var name: StringRef
    var inNodes: Bool
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
    	
        # lets say a node can have maximum of 64 parents and children
        let parents = Pointer[Int].alloc(64)
        memset_zero(parents, 64)
        let num_parents = 0 
        let children = Pointer[Int].alloc(64)
        memset_zero(children, 64)
        let num_children = 0 

        let name = StringRef('none')

        return Node{
            id: 1,
            name: name,
            num_dims: _num_dims,
            cap: _cap,
            shape: shape,
            skips: skips,
            data: data,
            gradient: gradient,
            parents: parents,
            num_parents: num_parents,
            children: children,
            num_children: num_children,
            inNodes: False,
            visited: False,
            requiresGradient: True
        }

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
    fn setInNodes(inout self: Self, val: Bool):
        self.inNodes = val

    @always_inline
    fn getInNodes(self) -> Bool:
        return self.inNodes

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

    @always_inline
    fn getData(self, index: Int) -> Float32:
        return self.data.load(index)

    @always_inline
    fn getData(self) -> DTypePointer[DType.float32]:
        return self.data    

    @always_inline
    fn printData(self):
        let num_dims = self.getNum_dims()
        let cols: Int = self.getShape(num_dims-1)
        let col_skips: Int = (self.getSkips(0) * self.getShape(0)) // cols
        print_no_newline("<Tensor: ")
        for i in range(col_skips):

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
    fn getNum_parents(self) -> Int:
        return self.num_parents

    @always_inline
    fn setNum_parents(inout self, val: Int):
        self.num_parents = val                

    @always_inline
    fn addParent(inout self, parentId: Int):
        let index = self.getNum_parents()
        self.parents.store(index, parentId)
        self.num_parents = index + 1

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
    fn getNum_children(self) -> Int:
        return self.num_children#.load(0)

    @always_inline
    fn addChild(inout self, childId: Int):
        let index = self.getNum_children()
        self.children.store(index, childId)
        self.num_children = index + 1

    @always_inline
    fn getChild(self, index: Int) -> Int:
        return self.children.load(index)

    @always_inline
    fn printChildren(self):
        print_no_newline("[ ")
        let len = self.getNum_children()
        for i in range(len):
            print_no_newline(self.children.load(i))
            if (i < len-1):
                print_no_newline(", ")
        print_no_newline(" ]\n")

    @always_inline
    fn setRequiresGradient(inout self, val: Bool):
        self.requiresGradient = val

    @always_inline
    fn getRequiresGradient(self) -> Bool:
        return self.requiresGradient