import cv2

'''
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
			'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
			'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
'''
CLASSES = ['cup']

image_format = '.jpeg'

def show_labels_img(imgname):
	'''
	功能：将id为imgname的图片画出相关对象的bounding box，并显示出来。
	'''
	img = cv2.imread('JPEGImages/' + imgname + image_format)# 从指定文件夹中读取指定文件
	h, w = img.shape[:2]# 把图像宽和高的像素赋值给h,w变量
	print(w, h)# 将宽和高的像素值打印出来
	label =[]

	with open("labels/" + imgname + ".txt", 'r') as flabel:
		for label in flabel:
			label = label.split(' ')# 用空格把label中的内容分开并存储到label中
			label = [float(x.strip()) for x in label]# 将label中的数据装入列表中，并去除数据两端的空格
			print(CLASSES[int(label[0])])# 打印出标签id对应的对象名
			pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))# 算出左上角坐标,中点坐标减去盒子宽高的二分之一
			pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))# 算出右下角坐标,中点坐标加上盒子宽高的二分之一
			cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))# 在框框的左上角出打出对象的名字
			cv2.rectangle(img,pt1,pt2,(0,0,255,2))# 画出框框

	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def show_labels_img2(img, label):
	'''
	功能：cv2.imread的img数据的图片画出相关对象的bounding box，并显示出来。
	'''
	h, w = img.shape[:2]# 把图像宽和高的像素赋值给h,w变量
	print(CLASSES[int(label[0])])# 打印出标签id对应的对象名
	pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))# 算出左上角坐标,中点坐标减去盒子宽高的二分之一
	pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))# 算出右下角坐标,中点坐标加上盒子宽高的二分之一
	cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))# 在框框的左上角出打出对象的名字
	cv2.rectangle(img,pt1,pt2,(0,0,255,2))# 画出框框

	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()




if __name__ == '__main__':
	image_format = '.jpeg'
	test = show_labels_img('1')