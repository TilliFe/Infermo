from memory import memset_zero, memcpy
from memory.unsafe import Pointer
from node import Node
from tensorOps import mul, add, ReLU, reshape
from tensorOpsGradients import mul_grad, add_grad, ReLU_grad, reshape_grad
from vector import Vec

struct nn:
    var nodes: DynamicVector[Node]
    var counter: Int
    var forwardTape: DynamicVector[Int]
    var forwardTapeGenerated: Bool
    var backwardTape: DynamicVector[Int]
    var backwardTapeGenerated: Bool

    fn __init__(inout self):
        self.nodes = DynamicVector[Node](0)
        self.counter = 0
        self.forwardTape = DynamicVector[Int]()
        self.forwardTapeGenerated = False
        self.backwardTape = DynamicVector[Int]()
        self.backwardTapeGenerated = False

    @always_inline
    fn addForward(inout self, nodeId: Int):
        self.forwardTape.push_back(nodeId)

    @always_inline
    fn addBackward(inout self, nodeId: Int):
        self.backwardTape.push_back(nodeId)

    @always_inline
    fn addNode(inout self, inout a: Node):
        a.setId(self.counter)
        a.setInNodes(True)
        self.counter += 1
        self.nodes.push_back(a)

    @always_inline
    fn tensor(inout self, *s: Int) -> Node:
        let v = VariadicList[Int](s)
        let len = len(v)
        var shape = DynamicVector[Int](0)
        for i in range(len):
            shape.push_back(v[i])

        var newNode = Node(shape)
        # self.addNode(newNode)

        return newNode #Pointer[Node].address_of(newNode)

    @always_inline
    fn printNode(self, index: Int):
        self.nodes[index].printData()

    @always_inline
    fn getNode(inout self, index: Int) -> Node:
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

        C.setName('mul')

        if(not A.getInNodes()):
            C.addParent(self.counter)
            self.addNode(A)
        else:
            C.addParent(A.getId())

        if(not B.getInNodes()):
            C.addParent(self.counter)
            self.addNode(B)
        else:
            C.addParent(B.getId())
        self.addNode(C)

        return C 
        
    @always_inline
    fn add(inout self, inout A: Node, inout B: Node) -> Node:

        # check dimensions
        if(A.getNum_dims() != B.getNum_dims()):
            print("Error (at add): number of dimensions are not equal")
        let num_dims = A.getNum_dims()
        if(A.getShape(num_dims-2) != B.getShape(num_dims-2) or A.getShape(num_dims-1) != B.getShape(num_dims-1)):
            print("Error (at add): For Matrix ADdition, Matrices need to in the following shape: C[mxn] = A[mxn] + B[mxn]")

        # init result Node 
        var new_shape = DynamicVector[Int]()
        for i in range(num_dims):
            new_shape.push_back(A.getShape(i))
        var C = Node(new_shape)

        C.setName('add')

        if(not A.getInNodes()):
            C.addParent(self.counter)
            self.addNode(A)
        else:
            C.addParent(A.getId())

        if(not B.getInNodes()):
            C.addParent(self.counter)
            self.addNode(B)
        else:
            C.addParent(B.getId())
        self.addNode(C)

        return C 

    @always_inline
    fn ReLU(inout self, inout A: Node) -> Node: 
        var new_shape = DynamicVector[Int]()
        for i in range(A.getNum_dims()):
            new_shape.push_back(A.getShape(i))

        var B = Node(new_shape)

        B.setName('ReLU')

        if(not A.getInNodes()):
            B.addParent(self.counter)
            self.addNode(A)
        else:
            B.addParent(A.getId())
        self.addNode(B)

        return B

    @always_inline
    fn reshape(inout self, inout A: Node, dir: Int) -> Node: 
        if(dir != -1):
            print("Error (at reshape)!")
        var new_shape = DynamicVector[Int](0)
        new_shape.push_back(A.getShape(0))
        new_shape.push_back(1)
        for i in range(1, A.getNum_dims()):
            new_shape[1] *= A.getShape(i)
        var B = Node(new_shape)

        B.setName('reshape')

        if(not A.getInNodes()):
            B.addParent(self.counter)
            self.addNode(A)
        else:
            B.addParent(A.getId())
        self.addNode(B)

        return B

    @always_inline
    fn reshape(inout self, inout A: Node, newShape: DynamicVector[Int]) -> Node: 
        var B = Node(newShape)
        B.setData(A.getData())

        B.setName('reshape')

        if(not A.getInNodes()):
            B.addParent(self.counter)
            self.addNode(A)
        else:
            B.addParent(A.getId())
        self.addNode(B)

        if(A.getCap() != B.getCap()):
            print("Error (at reshape): The Product of the dimensions in shapes must be the same!")

        return B


    fn topOrder(inout self, inout node: Node):  
        if not node.getVisited():
            for i in range(node.getNum_parents()):
                let nextNodeId = node.getParent(i)
                var nextNode = self.nodes[nextNodeId]
                self.topOrder(nextNode)
            self.forwardTape.push_back(node.getId())
            node.setVisited(True)

    @always_inline
    fn forward(inout self):
        let numNodes = self.getCounter()
        if(not self.forwardTapeGenerated):
            var lastNode = self.nodes[self.getCounter()-1]
            self.topOrder(lastNode)
            self.forwardTapeGenerated = True

        for i in range(self.counter):
            var curr = self.nodes[i]
            if(curr.getName() == 'mul'):
                let par1 = self.nodes[curr.getParent(0)]
                let par2 = self.nodes[curr.getParent(1)]
                mul(curr,par1,par2)
            if(curr.getName() == 'add'):
                let par1 = self.nodes[curr.getParent(0)]
                let par2 = self.nodes[curr.getParent(1)]
                add(curr,par1,par2)
            if(curr.getName() == 'ReLU'):
                let par1 = self.nodes[curr.getParent(0)]
                ReLU(curr,par1)  
            if(curr.getName() == 'reshape'):
                let par1 = self.nodes[curr.getParent(0)]
                reshape(curr,par1)

    fn backwardOrder(inout self, node: Node):
        self.backwardTape.push_back(node.getId())
        var it = 0
        while(it < len(self.backwardTape)):
            let currId = self.backwardTape[it]
            let curr = self.nodes[currId]
            for i in range(curr.getNum_parents()):
                let parId = curr.getParent(i)
                let par = self.nodes[parId]
                if(par.getRequiresGradient()):
                    self.backwardTape.push_back(parId)
            it += 1

    @always_inline
    fn backward(inout self):
        let numNodes = self.getCounter()
        if(not self.backwardTapeGenerated):
            let lastNode = self.nodes[self.getCounter()-1]
            self.backwardOrder(lastNode)
            self.backwardTapeGenerated = True

        for i in range(len(self.backwardTape)):
            let currId = self.backwardTape[i]
            let curr = self.nodes[currId]
            if(curr.getName() == 'mul'):
                var par1 = self.nodes[curr.getParent(0)]
                var par2 = self.nodes[curr.getParent(1)]
                mul_grad(curr,par1,par2)
            if(curr.getName() == 'add'):
                var par1 = self.nodes[curr.getParent(0)]
                var par2 = self.nodes[curr.getParent(1)]
                add_grad(curr,par1,par2)
            if(curr.getName() == 'ReLU'):
                var par1 = self.nodes[curr.getParent(0)]
                ReLU_grad(curr,par1)
            if(curr.getName() == 'reshape'):
                var par1 = self.nodes[curr.getParent(0)]
                reshape_grad(curr,par1)

    @always_inline
    fn printNodes(self): 
        print("Printing all Nodes of the Computational Graph .....\n")
        for i in range(self.counter):
            let n = self.nodes[i]
            print("Node ID: ", n.getId(), ", Name: ", n.getName())
            n.printData()
            n.printGradient()
        print("End of Printing all Nodes of the Computational Graph.")

                    
