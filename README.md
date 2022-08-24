# pytorch实现YOLOv1
**操作步骤**

1. 在 JPEGImages 目录中存放要训练的图片

2. Annotations 目录中存放 JPEGImages 目录中图片对应的标签，为 xml 文档，格式如下：

   ```xml
   <annotation>
   	<folder>JPEGImages</folder>
   	<filename>1.jpeg</filename>
   	<path>/Users/yun/Desktop/python_work/my_yolo_code/JPEGImages/1.jpeg</path>
   	<source>
   		<database>Unknown</database>
   	</source>
   	<size>
   		<width>1080</width>
   		<height>1440</height>
   		<depth>3</depth>
   	</size>
   	<segmented>0</segmented>
   	<object>
   		<name>cup</name>
   		<pose>Unspecified</pose>
   		<truncated>0</truncated>
   		<difficult>0</difficult>
   		<bndbox>
   			<xmin>398</xmin>
   			<ymin>273</ymin>
   			<xmax>670</xmax>
   			<ymax>978</ymax>
   		</bndbox>
   	</object>
   </annotation>
   ```

3. labels 目录下为解析后的标签文档，文档为 “.txt” 后缀

4. ImageSets/Main/ 中 放入要训练和验证的图片名字

5. 训练并验证结果
