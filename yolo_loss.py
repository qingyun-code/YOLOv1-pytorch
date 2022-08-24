import torch
import utils as U
import numpy as np

def yolo_loss(predict, labels):
	'''
	功能：通过yolov1的损失函数计算yolo的损失
	参数：
	——predict：yolo_net的预测结果(batchsize,30,7,7)
	——labels：样本数据(batchsize,30,7,7)
	'''

	num_gridx, num_gridy = labels.size()[-2:]# 赋值网格的宽和高
	num_bbox  = 2# 将bounding box的数量设置为2
	num_class = 20# 展示类别的数量
	class_loss = 0.# 含有目标网格的类别损失
	coord_loss = 0.# 含有目标网格的bbox的坐标损失
	noobj_confidence_loss  = 0.# 不含目标的网格损失(只有置信度损失)
	object_confidence_loss = 0.# 含有目标的bbox的置信度损失
	batch = labels.size()[0]# 获取batchsize的大小

	for batch_size in range(batch):
		for wide in range(num_gridx):
			for high in range(num_gridy):

				# 如果所在网格有对象
				if labels[batch_size, 4, wide, high] == 1:
					'''
					根据(cx-w/2,cy-h/2,cx+w/2,cy+h/2)计算bbox坐标
					其中(cx,cy)为bbox的中心坐标，计算公式为dateset.py文件
					里的convert函数中计算labels公式的逆运算
					'''
					# bbox1的预测坐标
					bbox1_predict = ((predict[batch_size, 0, wide, high] + wide) / num_gridx - predict[batch_size, 2, wide, high] / 2,
									(predict[batch_size, 1, wide, high] + high) / num_gridy - predict[batch_size, 3, wide, high] / 2,
									(predict[batch_size, 0, wide, high] + wide) / num_gridx + predict[batch_size, 2, wide, high] / 2,
									(predict[batch_size, 1, wide, high] + high) / num_gridy + predict[batch_size, 3, wide, high] / 2)

					# bbox2的预测坐标
					bbox2_predict = ((predict[batch_size, 5, wide, high] + wide) / num_gridx - predict[batch_size, 7, wide, high] / 2,
									(predict[batch_size, 6, wide, high] + high) / num_gridy - predict[batch_size, 8, wide, high] / 2,
									(predict[batch_size, 5, wide, high] + wide) / num_gridx + predict[batch_size, 7, wide, high] / 2,
									(predict[batch_size, 6, wide, high] + high) / num_gridy + predict[batch_size, 8, wide, high] / 2)

					# 真实的bbox的坐标
					bbox_real = ((labels[batch_size, 0, wide, high] + wide) / num_gridx - labels[batch_size, 2, wide, high] / 2,
								(labels[batch_size, 1, wide, high] + high) / num_gridy - labels[batch_size, 3, wide, high] / 2,
								(labels[batch_size, 0, wide, high] + wide) / num_gridx + labels[batch_size, 2, wide, high] / 2,
								(labels[batch_size, 1, wide, high] + high) / num_gridy + labels[batch_size, 3, wide, high] / 2)

					# 根据得到的预测坐标与真实坐标计算iou，坐标值都是正规化后的
					iou1 = U.iou(bbox1_predict, bbox_real)
					iou2 = U.iou(bbox2_predict, bbox_real)

					# 如果iou1>iou2则将bbox1视作负责物体，反之相反
					if iou1 >= iou2:

						# 就算坐标和宽高的损失，因为是重要信息，所以进行加整数权处理
						coord_loss = coord_loss + 5 * (torch.sum((predict[batch_size, 0:2, wide, high] - labels[batch_size, 0:2, wide, high]) ** 2) \
								+ torch.sum((predict[batch_size, 2:4, wide, high].sqrt() - labels[batch_size, 2:4, wide, high].sqrt()) ** 2))

						# 将iou1视作负责体然后计算bbox1的confidence损失
						object_confidence_loss = object_confidence_loss + (predict[batch_size, 4, wide, high] - iou1) ** 2

						# 因为iou2比较小所以bbox并不是很重要然后用0.5权重对其confidence损失进行抑制
						noobj_confidence_loss = noobj_confidence_loss + 0.5 * ((predict[batch_size, 9, wide, high] - iou2) ** 2)
					else:
						coord_loss = coord_loss + 5 * (torch.sum((predict[batch_size, 5:7, wide, high] - labels[batch_size, 5:7, wide, high]) ** 2) \
								+ torch.sum((predict[batch_size, 7:9, wide, high].sqrt() - labels[batch_size, 7:9, wide, high].sqrt()) ** 2))
						object_confidence_loss = object_confidence_loss + (predict[batch_size, 9, wide, high] - iou2) ** 2
						noobj_confidence_loss = noobj_confidence_loss + 0.5 * ((predict[batch_size, 4, wide, high] - iou1) ** 2)

					# 计算对象概率的损失函数
					class_loss = class_loss + torch.sum((predict[batch_size, 10:, wide, high] - labels[batch_size, 10:, wide, high]) **2 )

				else:
					# 如果不包含对象物体，将两个bbox的confidence的损失进行抑制
					noobj_confidence_loss = noobj_confidence_loss + 0.5 * torch.sum(predict[batch_size, [4, 9], wide, high] ** 2)

	# 计算总的损失
	loss = coord_loss + object_confidence_loss + noobj_confidence_loss + class_loss

	return loss/batch

if __name__ == '__main__':
	predict = np.random.rand(1, 30, 7, 7)
	labels = np.random.rand(1, 30, 7, 7)
	predict = torch.from_numpy(predict)
	labels = torch.from_numpy(labels)
	#print(predict)
	#print(predict.size()[-2:])
	#num_gridx, num_gridy = labels.size()[-2:]
	#print(num_gridx)
	loss = yolo_loss(predict, labels)
	print(loss)



