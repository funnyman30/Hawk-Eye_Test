from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import json
import torch

class plateDataset(Dataset):
    def __init__(self, labels_folder_path, transform=None):
        self.labels_folder_path = labels_folder_path
        self.transform = transform
        self.image_paths, self.labels = self._load_image_paths_and_labels()

    def _load_image_paths_and_labels(self):
        image_paths = []
        labels = []
        
        json_files = [f for f in os.listdir(self.labels_folder_path) if f.endswith('.json')]
        
        # 遍历JSON文件读取图像路径和标签
        for json_file in json_files:
            file_path = os.path.join(self.labels_folder_path,json_file)
            
            # 打开JSON文件并加载数据
            with open(file_path,'r') as file:
                data = json.load(file)
            
            label = []
            for point in data['points']:
                for cor in point:
                    # 对坐标进行缩放
                    cor = cor/5
                    
                    label.append(cor)
            # 保存路径和标签
            image_paths.append(data['imagePath'])
            labels.append(label)
            
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        label = torch.tensor(label)
        try:
            image = Image.open(image_path)
            if self.transform is not None:
                image = self.transform(image)
            return image, label
        except (IOError, SyntaxError):
            print(f"Error opening image file: {image_path}")