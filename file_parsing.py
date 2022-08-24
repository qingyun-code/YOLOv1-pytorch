import xml.etree.ElementTree as ET
import utils as U
import drawing as dr
import os
import cv2

'''
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
			'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
			'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
'''
CLASSES = ['cup']

def convert_annotation(image_id):
	'''
	功能：将Annotations文件夹中的名字为image_id的xml文档打开并解析
	然后在labels文件夹中创建一个和image_id同名的.txt文档，把解析出来
	的class_id,x,y,w,h写入labels文件夹里面的对应名字的文档中。
	参数：
	image_id：一个带有.xml的字符串，例如"2007_000027.xml"
	'''
	with open('Annotations/%s' % (image_id)) as in_file:
		image_id = image_id.split('.')[0]# 用'.''000000.txt'文件名分成image_id和'txt'两个字符串
		out_file = open('labels/%s.txt' % (image_id), 'w')# 打开相应文档，如果没有没有就会创个相关文档并打开文档进行写模式同时清空原文档内容
		tree = ET.parse(in_file)# 分析指定xml文件
		root = tree.getroot()# 获取第一标签
		size = root.find('size')# 查找第一标签中'size'标签
		w = int(size.find('width').text)# 查找size标签下的'width'子标签
		h = int(size.find('height').text)# 查找size标签下的'height'子标签
		pixel = (w, h)# 定义一张图片的宽高像素

		for obj in root.iter('object'):# 遍历所有的object标签
			difficult = obj.find('difficult').text# 如果发现difficult标签，则把difficult标签的内容赋值
			cls = obj.find('name').text# 如果发现name标签，则把name标签的内容赋值

			if cls not in CLASSES or int(difficult) == 1:
				continue# 如果没有在对象集合里发现相应的对象则跳过循环

			cls_id = CLASSES.index(cls)# 返回cls在class类中所表示的对象的序列号
			xmlbox = obj.find('bndbox')# 查找object标签下的bndbox标签
			box_points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
				float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))# 将bndbox标签的坐标值取出来放到points中
			box_points_normal = U.normal(pixel, box_points)# 将bounding box的坐标和长宽进行归一化
			out_file.write(str(cls_id) + " " + " ".join([str(a) for a in box_points_normal]) + '\n')# 将归一化后的数据写入新建的文档中
			# label内放入(cls_id,x,y,w,h),其中(x,y)为方框中点坐标

	return image_id, w, h

def make_label_txt():
	'''
	功能：在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息
	'''
	file_names = os.listdir('Annotations')# 返回存储Annotations文件夹里的所有文件名的列表

	for file in file_names:
		convert_annotation(file)

	return file_names

if __name__ == '__main__':
	'''
	image_id = '2007_000027.xml'
	test = convert_annotation(image_id)
	print(test)
	test2 = make_label_txt()
	print(test2)
	'''
	'''
	test = convert_annotation('2007_000256.xml')
	test2 = make_label_txt()
	dr.show_labels_img('2007_000256')
	'''
	for i in range(50):
		convert_annotation(str(i + 1) + '.xml')