from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from node import Node
from tensorOps import mul, add, ReLU, reshape
from tensorOpsGradients import mul_grad, add_grad, ReLU_grad, reshape_grad
from vector import Vec

struct nn:
    var nodes: DynamicVector[Pointer[Node]]
    var counter: Int
    var forwardTape: DynamicVector[Int]
    var forwardTapeGenerated: Bool
    var backwardTape: DynamicVector[Int]
    var backwardTapeGenerated: Bool

    fn __init__(inout self):
        self.nodes = DynamicVector[Pointer[Node]](0)
        self.counter = 0
        self.forwardTape = DynamicVector[Int](0)
        self.forwardTapeGenerated = False
        self.backwardTape = DynamicVector[Int](0)
        self.backwardTapeGenerated = False

    @always_inline
    fn addForward(inout self, nodeId: Int):
        self.forwardTape.push_back(nodeId)

    @always_inline
    fn addBackward(inout self, nodeId: Int):
        self.backwardTape.push_back(nodeId)

    @always_inline
    fn addNode(inout self, inout a: Node):
        self.nodes.push_back(Pointer[Node].address_of(a))
        a.setId(self.counter)
        a.setInNodes(True)
        self.counter += 1

    @always_inline
    fn tensor(inout self, *s: Int) -> Node:
        let v = VariadicList[Int](s)
        let len = len(v)
        var shape = DynamicVector[Int](0)
        for i in range(len):
            shape.push_back(v[i])

        let newNode = Node(shape)

        return newNode #Pointer[Node].address_of(newNode)

    @always_inline
    fn printNode(self, index: Int):
        self.nodes[index].load().printData()

    @always_inline
    fn getNode(inout self, index: Int) -> Pointer[Node]:
        return self.nodes[index]

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
    fn mul(inout self, inout A: Node, inout B: Node) -> Node:

        # check dimensions
        if(A.getNum_dims() != B.getNum_dims()):
            print("Error (a mul): number of dimensions are not equal")
        if(A.getNum_dims() < 2):
            print("Error (at mul): Nodes must be of shape at least 2, in order to perform matrix multiplication")
        let num_dims = A.getNum_dims()
        if(A.getShape(num_dims-1) != B.getShape(num_dims-2)):
            print("Error (at mul): For Matrix Multiplication, Matrices need to in the following shape: C[mxn] = A[mxk] * B[kxn]")

        # init result Node 
        var new_shape = DynamicVector[Int](0)
        for i in range(num_dims-1):
            new_shape.push_back(A.shape[i])
        new_shape.push_back(B.shape[num_dims-1])
        var C = Node(new_shape)
   
        if(not A.getInNodes()):
            self.addNode(A)
        if(not B.getInNodes()):
            self.addNode(B)
        self.addNode(C)
        C.setName('mul')

        C.addParent(A.getId())
        C.addParent(B.getId())
        A.addChild(C.getId())
        B.addChild(C.getId())

        return C #ret
        
    @always_inline
    fn add(inout self, inout A: Node, inout B: Node) -> Node:

        # check dimensions
        if(A.getNum_dims() != B.getNum_dims()):
            print("Error (at add): number of dimensions are not equal")
        let num_dims = A.getNum_dims()
        if(A.getShape(num_dims-2) != B.getShape(num_dims-2) or A.getShape(num_dims-1) != B.getShape(num_dims-1)):
            print("Error (at add): For Matrix ADdition, Matrices need to in the following shape: C[mxn] = A[mxn] + B[mxn]")

        # init result Node 
        var new_shape = DynamicVector[Int](0)
        for i in range(num_dims):
            new_shape.push_back(A.getShape(i))
        var C = Node(new_shape)

        if(not A.getInNodes()):
            self.addNode(A)
        if(not B.getInNodes()):
            self.addNode(B)
        self.addNode(C)
        C.setName('add')

        C.addParent(A.getId())
        C.addParent(B.getId())
        A.addChild(C.getId())
        B.addChild(C.getId())

        return C 

    @always_inline
    fn ReLU(inout self, inout A: Node) -> Node: 
        var new_shape = DynamicVector[Int](0)
        for i in range(A.getNum_dims()):
            new_shape.push_back(A.getShape(i))
        var B = Node(new_shape)

        if(not A.getInNodes()):
            self.addNode(A)
        self.addNode(B)
        B.setName('ReLU')

        B.addParent(A.getId())
        A.addChild(B.getId())

        return B

    @always_inline
    fn reshape(inout self, inout A: Node,dir: Int) -> Node: 
        if(dir != -1):
            print("Error (at reshape)!")
        var new_shape = DynamicVector[Int](0)
        new_shape.push_back(A.getShape(0))
        new_shape.push_back(1)
        for i in range(1, A.getNum_dims()):
            new_shape[1] *= A.getShape(i)
        var B = Node(new_shape)

        if(not A.getInNodes()):
            self.addNode(A)
        self.addNode(B)
        B.setName('reshape')

        B.addParent(A.getId())
        A.addChild(B.getId())

        return B

    @always_inline
    fn reshape(inout self, inout A: Node, newShape: Vec) -> Node: 
        var B = Node(newShape.get())

        if(not A.getInNodes()):
            self.addNode(A)
        self.addNode(B)
        B.setName('reshape')

        B.addParent(A.getId())
        A.addChild(B.getId())
        if(A.getCap() != B.getCap()):
            print("Error (at reshape): The Product of the dimensions in shapes must be the same!")

        return B


    fn topOrder(inout self, inout node: Node):  
        if not node.getVisited():
            for i in range(node.getNum_parents()):
                let nextNodeId = node.getParent(i)
                var nextNode = self.nodes[nextNodeId].load()
                self.topOrder(nextNode)
            self.forwardTape.push_back(node.getId())
            node.setVisited(True)

    @always_inline
    fn forward(inout self):
        let numNodes = self.getCounter()
        if(not self.forwardTapeGenerated):
            var lastNode = self.nodes[self.getCounter()-1].load()
            self.topOrder(lastNode)
            self.forwardTapeGenerated = True

        for i in range(numNodes):
            var curr = self.nodes[i].load()
            if(curr.name == 'mul'):
                let par1 = self.nodes[curr.getParent(0)].load()
                let par2 = self.nodes[curr.getParent(1)].load()
                mul(curr,par1,par2)
            if(curr.name == 'add'):
                let par1 = self.nodes[curr.getParent(0)].load()
                let par2 = self.nodes[curr.getParent(1)].load()
                add(curr,par1,par2)
            if(curr.name == 'ReLU'):
                let par1 = self.nodes[curr.getParent(0)].load()
                ReLU(curr,par1)  
            if(curr.name == 'reshape'):
                let par1 = self.nodes[curr.getParent(0)].load()
                reshape(curr,par1)     

    fn backwardOrder(inout self, node: Node):
        self.backwardTape.push_back(node.getId())
        var it = 0
        while(it < len(self.backwardTape)):
            let currId = self.backwardTape[it]
            let curr = self.nodes[currId].load()
            for i in range(curr.getNum_parents()):
                let parId = curr.getParent(i)
                let par = self.nodes[parId].load()
                if(par.getRequiresGradient()):
                    self.backwardTape.push_back(parId)
            it += 1

    @always_inline
    fn backward(inout self):
        let numNodes = self.getCounter()
        if(not self.backwardTapeGenerated):
            let lastNode = self.nodes[self.getCounter()-1].load()
            self.backwardOrder(lastNode)
            self.backwardTapeGenerated = True

        for i in range(len(self.backwardTape)):
            let currId = self.backwardTape[i]
            let curr = self.nodes[currId].load()
            if(curr.getName() == 'mul'):
                var par1 = self.nodes[curr.getParent(0)].load()
                var par2 = self.nodes[curr.getParent(1)].load()
                mul_grad(curr,par1,par2)
            if(curr.getName() == 'add'):
                var par1 = self.nodes[curr.getParent(0)].load()
                var par2 = self.nodes[curr.getParent(1)].load()
                add_grad(curr,par1,par2)
            if(curr.getName() == 'ReLU'):
                var par1 = self.nodes[curr.getParent(0)].load()
                ReLU_grad(curr,par1)
            if(curr.getName() == 'reshape'):
                var par1 = self.nodes[curr.getParent(0)].load()
                reshape_grad(curr,par1)

    @always_inline
    fn printNodes(self): 
        print("Printing all Nodes of the Computational Graph .....\n")
        for i in range(self.counter):
            let n = self.nodes[i].load()
            print("Node ID: ", n.getId(), ", Name: ", n.getName())
            n.printData()
            n.printGradient()
        print("End of Printing all Nodes of the Computational Graph.")

                    
