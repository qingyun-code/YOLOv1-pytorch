import numpy as np
import cv2
import random
import utils as U
import drawing as dr

'''
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
			'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
			'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
'''

CLASSES = ['cup']

image_format = '.jpeg'

NUM_BBOX = 2# 自己设定的bounding box的数目

class dataset():
	def __init__(self, is_train = True):
		self.file_names = []# 设置存储着文件名的列表
		self.file_names_backup = []# 备份一个文件名列表

		# 如果是要进行训练读取训练数据文档，否则读取验证数据文档
		if is_train:
			with open("ImageSets/Main/train.txt", 'r') as f:# 打开训练文档
				self.file_names = [x.strip() for x in f]# 将训练数据的文档名存储到文件列表中,.strip()去除字符串前后的空格或换行符
				self.file_names_backup = self.file_names[:]
		else:
			with open("ImageSets/Main/val.txt", 'r') as f:# 打开验证数据文档
				self.file_names = [x.strip() for x in f]# 将验证数据的文档名存储到文件列表中
				self.file_names_backup = self.file_names[:]

		self.img_path = "JPEGImages/"# 图片存储路径
		self.label_path = "labels/"# label数据文档存储数据

	def __len__(self):
		'''
		功能：返回文件名的个数
		'''
		return len(self.file_names)# 返回文件名的个数

	def image_process(self, picture_index):
		'''
		功能：对图像数据进行规范化处理
		参数：
		——picture_index：图片在文件名当中的索引
		'''
		img = cv2.imread(self.img_path + self.file_names[picture_index] + image_format)# 读取需要图像处理和数据转换的图像

		with open(self.label_path+self.file_names[picture_index] + ".txt") as f:# 根据索引打开需要转换的图片的labels中的文档
			bbox = f.read().split('\n')# 读取文档中bbox数值，一行代表一个bbox数据，以'\n'为分隔符读取

		bbox = [x.split() for x in bbox]
		bbox = [float(x) for y in bbox for x in y]

		if len(bbox) % 5 != 0:
			raise ValueError("File:" + self.label_path + self.filenames[picture_index] + ".txt"+"——bbox Extraction Error!")

		img, bbox = zoom(img, bbox)# 规范化图片和bbox
		# dr.show_labels_img2(img, bbox)# 画出缩小后的图片
		labels = convert(bbox)# 转换训练时需要用的labels数据

		return img, labels

	def batches(self, batch_size):
		'''
		功能：打乱存储的文件名
		参数：
		——batch_size：每一批训练的数目
		'''
		n = len(self.file_names)# 获取文件名列表的长度
		random.shuffle(self.file_names_backup)# 随机打乱文件名列表中的内容

		# 将file_names按照批次分开成n/batch_size组列表
		mini_batches = [
			self.file_names_backup[k:k + batch_size]
			for k in range(0, n, batch_size)]

		return mini_batches

def zoom(img, bbox):
	'''
	功能：将图片的数据规范化为448*448
	参数：
	——img：利用cv2.imread返回的img的值
	——bbox：初始的bbox数据，从labels中读到的数据
	'''
	h, w = img.shape[0:2]# 获取原始图像的高和宽
	input_size = 448# 定义YOLOv1的输入尺寸为448*448
	padw, padh = 0, 0# 给padw和padh初始化定义为0

	if h > w:
		padw = (h - w) // 2# 宽度的左右两边分别加上(h-w)/2维数据
		img = np.pad(img, ((0, 0), (padw, padw),
		(0, 0)), 'constant', constant_values = 0)# 在img中增加pad
	elif w > h:
		padh = (w - h) // 2
		img = np.pad(img, ((padh, padh), (0, 0),
		(0, 0)), 'constant', constant_values = 0)# 在img中增加pad

	img = cv2.resize(img, (input_size, input_size))# 把img数据变成448*448尺寸的图片

	for i in range(len(bbox) // 5):
		if padw != 0:
			bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h# 计算中心点xc的转换值，h=(w+2*pad)
			bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h# 计算中心点w的转换值，h=(w+2*pad)
		elif padh != 0:
			bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
			bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w

	return img, bbox

def convert(bbox):
	'''
	功能：将规范成448*448图片后的bbox里的数据(cls_id,xc,yc,w,h)转换成
	神经网络要输出的(7, 7, 5 * NUM_BBOX + len(CLASSES))的labels数据
	'''
	grid_size = 1.0 / 7# 定义一个grid_size
	labels = np.zeros((7, 7, 5 * NUM_BBOX + len(CLASSES)))# 将labels内的数据清空

	for i in range(len(bbox) // 5):# 遍历每个bounding box中的五个数据

		# 巧妙利用向下取整的语法来确定bbox的中心所在的单元格坐标
		gridx = int(bbox[i * 5 + 1] // grid_size)
		gridy = int(bbox[i * 5 + 2] // grid_size)

		# (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
		gridpx = bbox[i * 5 + 1] / grid_size - gridx
		gridpy = bbox[i * 5 + 2] / grid_size - gridy

		# 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
		labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
		labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
		labels[gridy, gridx, 10 + int(bbox[i * 5])] = 1

	return labels

def Data_Loader(train_data, batch_size):
	'''
	功能：通过获取的train_data和batch_size通过处理输出训练
	网络需要用的inputs和labels数据
	参数：
	——train_data：一个VOC2012类的对象
	——batch_size：一个批次所存在的数据量
	'''
	data = []# 创建数据列表
	mini_batches = train_data.batches(batch_size)# 获取一个批次的所有数据

	# 计算一个批次的inputs和labels
	for mini_batch in mini_batches:
		labels = np.zeros((batch_size, 5 * NUM_BBOX + len(CLASSES), 7, 7))
		inputs = np.zeros((batch_size, 3, 448, 448))

		for i in range(batch_size):
			img, label = train_data.image_process(int(mini_batch[i]) - 1)# 文件名-1就是这个文件所在的索引

			# 将label数据传给labels
			for wide in range(7):
				for high in range(7):
					for j in range(5 * NUM_BBOX + len(CLASSES)):
						labels[i, j, wide, high] = label[wide, high, j]

			img = np.array(img, dtype = np.float32)# 将img数据转换成numpy数组
			img = img / 255.0# 通过广播将img内的数据进行规范化

			# 将img的数据转换成输出数据
			for wide in range(448):
				for high in range(448):
					for channel in range(3):
						inputs[i, channel, wide, high] = img[high, wide, channel]
						# inputs[i, channel, wide, high] = U.normal_img(img[high, wide, channel])

		data.append((inputs, labels))

	return data





if __name__ == '__main__':
	image_format = '.jpeg'
	voc = dataset()
	# voc.image_process(0)
	img, labels = voc.image_process(0)
	print(img)
	print(img.shape)
	# dr.show_labels_img2(img, bbox)
	#print('img={}\nlabels={}'.format(img, labels))
	# print(voc.__len__())
