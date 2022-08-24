import numpy as np
import torch

def iou(box1, box2):
	'''
	功能：计算box1和box2两个方框的交并集
	参数：
	box1：第一个框，其值为左上角坐标和右下角坐标(x1, x2, y1, y2)
	box2：第二个框，其值为左上角坐标和右下角坐标(x1, x2, y1, y2)
	'''
	# 分别找出找左上角坐标的最大的x,y值
	xi1 = tensor_max(box1[0], box2[0])
	yi1 = tensor_max(box1[1], box2[1])

	# 分别找出找右下角坐标的最小的x,y值
	xi2 = tensor_min(box1[2], box2[2])
	yi2 = tensor_min(box1[3], box2[3])

	# 计算出两个框的交集面积
	inter_area = (yi2 - yi1) * (xi2 - xi1)

	# 分别求出两个框的面积
	box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
	box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])

	# 计算两个框的并集面积
	union_area = box1_area + box2_area - inter_area

	# 算出交并比
	iou = inter_area / union_area

	return iou

def normal(pixel, box):
	'''
	功能：计算box的中心点坐标并将box内的中心点坐标和h,w进行归一化
	参数：
	pixel：一个存储(w,h)像素点的字典，w代表图片宽的像素点，h代表高。
	box：存储坐标的框，坐标(x1,x2,y1,y2)
	'''
	# 计算box的中心点坐标(x,y)
	x = (box[0] + box[1]) / 2.0
	y = (box[2] + box[3]) / 2.0

	# 计算box的宽和高的大小
	w = box[1] - box[0]
	h = box[3] - box[2]

	# 将x,y,w,h进行归一化
	dw = 1. / pixel[0]
	dh = 1. / pixel[1]
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh

	return (x, y, w, h)

def normal_img(img_value):
	return img_value / 255.0

def tensor_max(num1, num2):
	'''
	功能：输出num1和num2中的最大值
	参数：
	——num1：第一个待比较的tensor数值
	——num2：第二个待比较的tensor数值
	'''
	if num1 > num2:
		return num1
	else:
		return num2

def tensor_min(num1, num2):
	'''
	功能：输出num1和num2中的最小值
	参数：
	——num1：第一个待比较的tensor数值
	——num2：第二个待比较的tensor数值
	'''
	if num1 > num2:
		return num2
	else:
		return num1