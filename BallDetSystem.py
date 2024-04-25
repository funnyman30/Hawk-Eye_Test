import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from yolo import YOLO
from PIL import Image
from utils.models import BallLocateModel
import torch
import torchvision.transforms as transforms

class BallDetSystem():
    
    def __init__(self) -> None:
        self.yolo = YOLO()
        self.plateLocateModel = BallLocateModel()
        # 加载模型权重
        path = "/Users/chyou/Documents/Academic/A-Projects/ensembleSystem/logs/resnet/model_weights_resnet18_v3_2.pth"
        self.plateLocateModel.load_state_dict(torch.load(path,map_location=torch.device('cpu'),))
        
    def get_FPS(self,image,det_interval):
        
        return self.yolo.get_FPS(image,det_interval)
    def get_relative_cors(self, video_path, track_saved_path = None, corrected_plate_saved_path = None):
        
        centers = self.detect_video(video_path,track_saved_path)

        key_frame_ind, _ = self.find_opt_frame(centers)
        
        key_frame = self.get_frame_from_video(video_path,key_frame_ind)
        
        # 对图像进行预处理
        trans = transforms.Compose([
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Resize((216, 384)),  # 调整大小
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

        # 执行预处理
        img_tensor = trans(key_frame)

        # 添加批次维度（batch dimension）
        image_tensor = img_tensor.unsqueeze(0)

        # 选择设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #print("device:" + str(device))
        
        
        # 获取模型输出
        with torch.no_grad():
            self.plateLocateModel.eval()
            image_tensor = image_tensor.to("cpu")
            output = self.plateLocateModel(image_tensor)


        predicted_coords = output.squeeze().detach().cpu().numpy()
    
        image = img_tensor.numpy()
        normalized_image = (image * np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]) + np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
        normalized_image = np.clip(normalized_image, 0, 1)
        # 将像素值重新缩放到0到255的范围
        normalized_image = (normalized_image * 255).astype(np.uint8)
        image = normalized_image.transpose(1, 2, 0)  # 从 (C, H, W) 到 (H, W, C)，以便matplotlib显示

        # 定义源和目标点坐标
        source_points = np.array([[predicted_coords[0], predicted_coords[1]],
                                [predicted_coords[2], predicted_coords[3]],
                                [predicted_coords[4], predicted_coords[5]],
                                [predicted_coords[6], predicted_coords[7]]], dtype=np.float32)
        plate_width = 800
        plate_height = 800
        target_points = np.array([[0, 0], [plate_width - 1, 0], [0, plate_height - 1], [plate_width - 1, plate_height - 1]], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(source_points, target_points)

        # 进行透视变换
        corrected_plate = cv2.warpPerspective(image, M, (plate_width, plate_height))
        
        # 是否保存矫正后的图片
        if corrected_plate_saved_path is not None:
            image_pil = Image.fromarray(np.uint8(corrected_plate))
            # 保存为jpg图片
            image_pil.save(corrected_plate_saved_path)
            
        # 获取靶盘中心点坐标
        center_cor = self.get_center_cor(corrected_plate)
        
        # 过滤原图中的像素
        corrected_plate = self.process_plate(corrected_plate)
        
        # 获取小球圆心坐标
        ball_cor = self.get_ball_cor(corrected_plate)
        
        return center_cor,ball_cor
    
    def get_ball_cor(self,corrected_plate):
        
        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(corrected_plate, cv2.COLOR_RGB2HSV)

        # 定义蓝色HSV范围
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # 根据蓝色范围进行阈值化
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 对阈值图像进行形态学处理
        kernel = np.ones((17, 17), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 对阈值图像进行形态学处理
        kernel = np.ones((21, 21), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 使用椭圆拟合功能
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化最大面积和对应的椭圆
        max_area = 0
        max_ellipse = None

        # 遍历所有检测到的椭圆
        for contour in contours:
            # 拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            # 计算椭圆的面积
            area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
            # 如果当前椭圆的面积大于之前记录的最大面积，则更新最大面积和椭圆
            if area > max_area:
                max_area = area
                max_ellipse = ellipse

        # 返回最大椭圆圆心坐标
        if max_ellipse is not None:
            return max_ellipse[0]
        
    def process_plate(self,corrected_plate):
        
        # 定义RGB范围
        lower_rgb = np.array([40, 80, 110])  # 最小阈值
        upper_rgb = np.array([100, 160, 220])  # 最大阈值

        # 创建掩码
        mask = cv2.inRange(corrected_plate, lower_rgb, upper_rgb)
        
        ##############################################################
        #           对mask进行形态学去噪处理
        
        # 定义形态学操作的内核大小
        kernel_size_closing = 24
        kernel_size_opening = 9
        # 定义形态学操作的内核
        kernel_closing = np.ones((kernel_size_closing, kernel_size_closing), np.uint8)
        kernel_opening = np.ones((kernel_size_opening, kernel_size_opening), np.uint8)
        # 使用闭运算消除噪声
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closing)
        # 使用开运算消除噪声
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening)
        
        # 对形态学处理过后的掩码进行膨胀
        # 定义膨胀操作的内核大小
        kernel_size = 11

        # 定义膨胀操作的内核
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 对掩码进行膨胀操作
        dilated_mask = cv2.dilate(opening, kernel, iterations=1)   
                        
        # 进行颜色过滤
        
        # 将原图转换为RGB格式（确保原图和掩码具有相同的颜色通道）

        # 将掩码取反，即将过滤出来的区域设置为0，其余区域设置为255
        filtered_area = cv2.bitwise_not(dilated_mask)

        # 将原图中需要过滤的像素设置为浅红色
        corrected_plate[filtered_area == 0] = [255, 255, 255]  # 设置为白色
        
        return corrected_plate
    
    def get_center_cor(self,np_corrected_plate):
        
        image = np_corrected_plate

        # 对图像进行高斯滤波来降噪
        image = cv2.GaussianBlur(image, (7, 7), 0)

        # 边缘检测
        edges = cv2.Canny(image, threshold1=30, threshold2=135)

        # 定义形态学核
        kernel = np.ones((11, 11), np.uint8)

        # 执行闭运算操作
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 定义形态学核
        kernel = np.ones((3, 3), np.uint8)
        # 执行开运算操作
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        ###################################################################
        #                        寻找最大矩形找到中心点                       #
        ###################################################################
        
        # 寻找图像中的轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化最大矩形相关变量
        max_area = 0
        max_rect = None

        # 遍历轮廓
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            
            # 如果面积足够大，则考虑它作为候选矩形
            if area > 1000:  # 根据需要调整阈值
                # 计算轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 如果当前矩形的面积更大，则更新最大矩形
                if area > max_area:
                    max_area = area
                    max_rect = (x, y, w, h)

        # 计算中心点
        if max_rect is not None:
            x, y, w, h = max_rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 计算矩形的中心点坐标
            center_x = x + w // 2
            center_y = y + h // 2
            
        return (center_x, center_y)
    
    def detect_video(self, video_path, track_saved_path):
        
        capture = cv2.VideoCapture(video_path)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        centers = []
        while(True):
            # 读取一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame, result_center = self.yolo.detect_image_center(frame)
            
            frame = np.array(frame)
            centers.append(result_center)
    
        capture.release()
        if track_saved_path is not None:
            np.save(track_saved_path, np.array(centers))
            
        return centers
    
    def detect_image(self, image_np):
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(np.uint8(image_np))

        # 进行目标检测
        _, result_center = self.yolo.detect_image_center(image)

        # 返回检测到的目标中心点坐标
        return result_center

    def find_opt_frame(self, coors):
        if len(coors) < 3:
            return None  # 需要至少3个点才能满足条件

        # 将输入数据转换为 NumPy 数组
        points = np.array(coors)

        # 寻找局部最小值的点
        local_minima_indices = []

        for i in range(1, len(coors) - 1):
            
            x, y = coors[i]
            # 跳过[0,0]点
            
            if x==0 and y==0:
                continue
            
            prev_y = coors[i - 1][1]
            next_y = coors[i + 1][1]
            
            if y > prev_y and y > next_y:
                local_minima_indices.append(i)
        
        if len(local_minima_indices) == 0:
            return None  # 没有找到局部最小值点

        local_minima_indices = np.array(local_minima_indices)
        # 计算每个局部最小值点与前后相邻2个点的y坐标距离
        distances1 = points[local_minima_indices + 1, 1] - points[local_minima_indices - 1, 1]
        distances2 = points[local_minima_indices + 2, 1] - points[local_minima_indices - 2, 1]
        distances = distances1 + distances2

        # 找到距离最大的局部最小值点的索引
        optimal_index = np.argmax(distances)

        # 获取满足条件的点的坐标
        optimal_point = points[local_minima_indices[optimal_index]]

        return local_minima_indices[optimal_index]+1, optimal_point
    
    def get_frame_from_video(self, video_path, frame_number):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        # 获取视频帧总数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 检查帧序号是否在有效范围内
        if frame_number < 0 or frame_number >= total_frames:
            print("Error: Invalid frame number.")
            cap.release()
            return None

        # 设置视频帧的位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # 读取视频帧
        ret, frame = cap.read()

        # 检查帧是否成功读取
        if not ret:
            print("Error: Failed to read frame.")
            cap.release()
            return None

        # 将BGR颜色空间转换为RGB颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 释放视频文件
        cap.release()

        # 返回帧的NumPy数组表示（以RGB颜色空间）
        return frame_rgb

    def plot_points(self, coordinates):
        # 将原点移到左下角
        flipped_coordinates = [(x, -y) for x, y in coordinates]

        # 提取 x 和 y 坐标
        x_values, y_values = zip(*flipped_coordinates)

        # 绘制散点图
        plt.scatter(x_values, y_values, color='red', marker='o')

        # 设置坐标轴标签
        plt.xlabel('X')
        plt.ylabel('Y')

        # 显示图形
        plt.show()
        