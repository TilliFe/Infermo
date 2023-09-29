from python import Python
from python.object import PythonObject
from memory.buffer import Buffer
from Tensor import shape
from random import randint, seed
from memory import memset_zero
from Tensor import Tensor

struct DataLoader:
    var indeces: DTypePointer[DType.int32]
    var data: DTypePointer[DType.float32]
    var file_path: String
    var rows: Int
    var cols: Int
    var counter: Int

    fn __init__(inout self, file_path: String)raises:
        print("Loading Dataset...")
        let np = Python.import_module("numpy")
        let py = Python.import_module("builtins")

        self.file_path = file_path
        let np_data = np.loadtxt(self.file_path)
        let np_shape = np_data.shape # ndim = 2 every time
        
        self.rows = np_shape[0].to_float64().to_int()
        self.cols = np_shape[1].to_float64().to_int()
        self.data = DTypePointer[DType.float32].alloc(self.rows * self.cols)
        self.indeces = DTypePointer[DType.int32].alloc(self.rows)
        self.counter = 0

        for i in range(self.rows):
            for j in range(self.cols):
                self.data.store( i * self.cols + j, np_data[i][j].to_float64().cast[DType.float32]())

        seed()
        randint[DType.int32](self.indeces,self.rows,0,self.rows-1)
    
    fn load(inout self, batchSize: Int, start: Int, end: Int, scalingFactor: Float32 = Float32(1.0)) raises -> DTypePointer[DType.float32]:
        # print(self.counter)
        var _start = start
        var _end = end
        let _batchSize = batchSize
        if(_start < 0):
            _start = 0
        if(_end>self.cols):
            _end = self.cols

        let batch = DTypePointer[DType.float32].alloc(_batchSize * (_end - _start))
        if(_batchSize < self.rows and _batchSize * (self.counter+1) < self.rows):
            self.counter += 1
            for i in range(_batchSize):
                let sampleIndex = self.indeces.load((self.counter-1) * _batchSize + i).to_int()
                for j in range(_start,_end):
                    batch.store(i * (_end-_start) + j - _start, scalingFactor * self.data.load(sampleIndex * self.cols + j))
        elif(_batchSize < self.rows):
            seed()
            randint[DType.int32](self.indeces,self.rows,0,self.rows-1)
            self.counter = 1
            for i in range(_batchSize):
                let sampleIndex = self.indeces.load((self.counter-1) * _batchSize + i).to_int()
                for j in range(_start,_end):
                    batch.store(i * (_end-_start) + j - _start, scalingFactor * self.data.load(sampleIndex * self.cols + j))
        else:
            print("Error: BatchSize exceeds the number of samples in the data set!")

        # if(self.counter == 0):
        #     for i in range(_batchSize):
        #         for j in range(0,_end-_start):
        #             if((_end-_start) > 30 and j >=10 and j <= ((_end-_start) - 10)):
        #                 if(j == 10):
        #                     print_no_newline("... ")
        #                 continue
        #             else:
        #                 print_no_newline(batch.load(i * (_end-_start) + j), '')
        #         put_new_line()
        #     put_new_line()

        return batch

    fn load_again(inout self, batchSize: Int, start: Int, end: Int, scalingFactor: Float32 = Float32(1.0)) raises -> DTypePointer[DType.float32]:
        var _start = start
        var _end = end
        let _batchSize = batchSize
        if(_start < 0):
            _start = 0
        if(_end>self.cols):
            _end = self.cols

        let batch = DTypePointer[DType.float32].alloc(_batchSize * (_end - _start))
        if(_batchSize < self.rows and _batchSize * (self.counter) < self.rows):
            for i in range(_batchSize):
                let sampleIndex = self.indeces.load((self.counter-1) * _batchSize + i).to_int()
                for j in range(_start,_end):
                    batch.store(i * (_end-_start) + j - _start, scalingFactor * self.data.load(sampleIndex * self.cols + j))

        return batch

    fn oneHot(inout self, batchSize: Int, index: Int, ndims: Int)raises -> DTypePointer[DType.float32]:
        let _batchSize = batchSize
        let batch = DTypePointer[DType.float32].alloc(_batchSize * ndims)

        for i in range(_batchSize):
            let sampleIndex = self.indeces.load((self.counter-1) * _batchSize + i).to_int()
            let entry = self.data.load(sampleIndex * self.cols + index).to_int()
            for j in range(ndims):
                if(entry == j):
                    batch.store(i * ndims + j, 1)
                else:
                    batch.store(i * ndims + j, 0)

        return batch

    fn print(inout self)raises:
        print("Dataset:",self.file_path)
        print("NumSamples:",self.rows)
        print("SampleSize:",self.cols)
        print("Example Sample:")
        let exampleBatch = self.load(1,0,self.cols)
        let exampleBatch2 = self.load(1,0,self.cols)
        let exampleBatch3 = self.load(1,0,self.cols)
        print("\n")
