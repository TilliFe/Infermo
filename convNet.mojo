from dv import *

fn main():
	var nn = Module()

	var A = Tensor(shape(1,2,4,4))
	A.setDataAll(0.1)
	
	var B = Conv2d(
			nn=nn,
			x=A,
			out_channels=2,
			kernel_width=3,
			kernel_height=3,
			stride=1,
			padding=0,
			use_bias=True     
		)
	
	var C = nn.sum(B)

	nn.forward(C)
	nn.backward(C)

	nn.printTensors()

	




    # let batch_size = 1
    # let in_channels = 1
    # let width = 4
    # let height = 4

    # let out_channels = 3
    # let kernel_width = 2
    # let kernel_height = 2
    # let padding = 1
    # let stride = 1

    # var A = Tensor(shape(batch_size,in_channels, width, height))
    # for batch in range(batch_size):
    #     for channel in range(in_channels):
    #         for i in range(width):
    #             for j in range(height):
    #                 let index = batch * in_channels * width * height + channel * width * height + i * height + j
    #                 A.data.store(index, Float32(index + 1) * Float32(0.1))

    # var B = Tensor(shape(out_channels,in_channels, kernel_width, kernel_height))
    # for filter in range(out_channels):
    #     for channel in range(in_channels):
    #         for i in range(kernel_width):
    #             for j in range(kernel_height):
    #                 let index = filter * in_channels * kernel_width * kernel_height + channel * kernel_width * kernel_height + i * kernel_height + j
    #                 B.data.store(index,Float32(0.1))
    
    # A.printData()
    # B.printData()

    # var C = nn.conv2d(A,B,padding=padding,stride=stride)

    # var D = nn.sum(C)

    # nn.forward(D)
    # nn.backward(D)

    # C.printData()
    # D.printData()

    # A.printGradient()

    
