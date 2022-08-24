import torch
import torch.nn as nn

class yolo_net(nn.Module):
	'''
	作用：输出7*7*30的YOLOv1结果
	nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
	是进行卷积运算的常用接口。nn.MaxPool2d(kernel_size,stride)进行最大
	池化的常用接口。nn.Linear(in_channels,out_channels)全连接层接口。
	'''
	def __init__(self):
		super(yolo_net, self).__init__()

		self.Conv_layers1 = nn.Sequential(
			nn.Conv2d(3, 64, 7, 2, 3),
			nn.LeakyReLU(),
			nn.MaxPool2d(2, 2)
		)

		self.Conv_layers2 = nn.Sequential(
			nn.Conv2d(64, 256, 3, 1, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d( 2, 2)
		)

		self.Conv_layers3 = nn.Sequential(
			nn.Conv2d(256, 128, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(256, 256, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2, 2)
		)

		self.Conv_layers4 = nn.Sequential(
			nn.Conv2d(512, 256, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(512, 256, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(512, 256, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(512, 256, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(512, 512, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(512, 1024, 3, 1, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2, 2)
		)

		self.Conv_layers5 = nn.Sequential(
			nn.Conv2d(1024, 512, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(512, 1024, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(1024, 512, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(512, 1024, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(1024, 1024, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(1024, 1024, 3, 2, 1),
			nn.LeakyReLU()
		)

		self.Conn_layer1 = nn.Sequential(
			nn.Linear(7 * 7 * 1024, 4096),
			nn.LeakyReLU()
		)

		self.Conn_layer2 = nn.Sequential(
			nn.Linear(4096, 7 * 7 * 11),
			nn.Sigmoid()# 将输出结果转化为0~1之间的数
		)

	def forward(self, input):
		'''
		对网络进行前向传播，并返回每层网络的返回结果
		'''
		input = self.Conv_layers1(input)
		# print('Conv_layers1')
		input1 = input
		input = self.Conv_layers2(input)
		# print('Conv_layers2')
		input2 = input
		input = self.Conv_layers3(input)
		# print('Conv_layers3')
		input3 = input
		input = self.Conv_layers4(input)
		# print('Conv_layers4')
		input4 = input
		input = self.Conv_layers5(input)
		# print('Conv_layers5')
		input5 = input

		# 将输入数据维度转换为一维
		input = input.view(input.size()[0], -1)

		input = self.Conn_layer1(input)
		# print('Conv_layers6')
		input6 = input
		input = self.Conn_layer2(input)
		# print('Conv_layers7')
		input7 = input
		output = input.reshape(-1, 11, 7, 7)
		#return input1, input2, input3, input4, input5, input6, input7, output
		return output

if __name__ == '__main__':
	'''
	测试网络准确情况
	'''
	x = torch.randn((1, 3, 448, 448))# 随机设置一个四维的初试输入数据
	net = yolo_net()# 新建一个网络对象
	print(net)# 打印出所构建的网络
	#output1, output2, output3, output4, output5, output6, output7, output = net(x)
	output = net(x)
	# 分别答应出每层网络各维度的数据量
	'''
	print(output1.size())
	print(output2.size())
	print(output3.size())
	print(output4.size())
	print(output5.size())
	print(output6.size())
	print(output7.size())
	'''
	print(output.size())
	print(output)