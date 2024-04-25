import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

class BallLocateModel(nn.Module):
    def __init__(self,num_output=8):
        super(BallLocateModel, self).__init__()
        # 定义模型结构
        # 加载预训练的 ResNet 模型
        self.pretrained_model = models.resnet18(pretrained=True)
        num_features = self.pretrained_model.fc.in_features
        # 替换最后一层全连接层
        self.pretrained_model.fc = nn.Linear(num_features, num_output)

        # 使用Kaiming初始化对最后一层全连接层进行权重初始化
        init.kaiming_normal_(self.pretrained_model.fc.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # 定义前向传播过程
        x = self.pretrained_model(x)
        
        return x