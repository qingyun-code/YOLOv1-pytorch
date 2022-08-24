import torch
import dataset as D
import yolo_net as YN
import yolo_loss as YL

epoch = 2# 设置迭代次数
batch_size = 5# 设置批量的大小
lr = 0.01# 设置学习率

train_data = D.dataset()
data = D.Data_Loader(train_data, batch_size)# 加载数据

net = YN.yolo_net()# 初始化一个网络对象

# 随机梯度下降
#optimizer = torch.optim.SGD(net.parameters(), lr = lr)
optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)

# optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay = 0.0005)

for e in range(epoch):
	for (inputs, labels) in data:
		inputs = torch.from_numpy(inputs)
		labels = torch.from_numpy(labels)
		inputs = inputs.float()
		#print(inputs.size())
		#print(inputs)
		predict = net(inputs)
		labels = labels.float()
		loss = YL.yolo_loss(predict, labels)
		print(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

