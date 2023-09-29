from memory import memset_zero, memcpy
from Tensor import Tensor, shape, Vec
from tensorOps import mul, add, ReLU, sum, softmax, MSE, CE, reshape
from tensorOpsGradients import mul_grad, add_grad, ReLU_grad, sum_grad, softmax_grad, MSE_grad, CE_grad, reshape_grad
from runtime.llcl import Runtime

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

    # @always_inline
    # fn tensor(inout self, *s: Int) -> Tensor:
    #     let v = VariadicList[Int](s)
    #     let len = len(v)
    #     var shape = DynamicVector[Int](0)
    #     for i in range(len):
    #         shape.push_back(v[i])

    #     var newTensor = Tensor(shape)

    #     return newTensor

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
    fn softmax(inout self, inout A: Tensor) -> Tensor: 

        var new_shape = DynamicVector[Int]()
        for i in range(A.getNum_dims()):
            new_shape.push_back(A.getShape(i))

        var B = Tensor(new_shape)

        B.setName('softmax')

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
    fn CE(inout self, inout A: Tensor, inout B: Tensor) -> Tensor:

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

        C.setName('CE')

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
            if(curr.getName() == 'ReLU'):
                let par1 = self.Tensors[curr.getParent(0)]
                ReLU(curr,par1) 
            if(curr.getName() == 'sum'):
                let par1 = self.Tensors[curr.getParent(0)]
                sum(curr,par1)
            if(curr.getName() == 'softmax'):
                let par1 = self.Tensors[curr.getParent(0)]
                softmax(curr,par1)
            if(curr.getName() == 'MSE'):
                let par1 = self.Tensors[curr.getParent(0)]
                let par2 = self.Tensors[curr.getParent(1)]
                MSE(curr,par1,par2) 
            if(curr.getName() == 'CE'):
                let par1 = self.Tensors[curr.getParent(0)]
                let par2 = self.Tensors[curr.getParent(1)]
                CE(curr,par1,par2) 
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
            if(curr.getName() == 'ReLU'):
                var par1 = self.Tensors[curr.getParent(0)]
                ReLU_grad(curr,par1)
            if(curr.getName() == 'sum'):
                var par1 = self.Tensors[curr.getParent(0)]
                sum_grad(curr,par1)
            if(curr.getName() == 'softmax'):
                var par1 = self.Tensors[curr.getParent(0)]
                softmax_grad(curr,par1)
            if(curr.getName() == 'MSE'):
                var par1 = self.Tensors[curr.getParent(0)]
                var par2 = self.Tensors[curr.getParent(1)]
                MSE_grad(curr,par1,par2)
            if(curr.getName() == 'CE'):
                var par1 = self.Tensors[curr.getParent(0)]
                var par2 = self.Tensors[curr.getParent(1)]
                CE_grad(curr,par1,par2)
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

                    
