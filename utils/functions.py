import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs, device,lr_scheduler,print_every=10, record_every=1):
    model.train()  # 设置模型为训练模式
    losses = []
    accuracies = []
    running_loss = 0.0
    correct = 0
    total = 0
    
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        if lr_scheduler is not None:
            # 在每个epoch开始前更新学习率
            lr_scheduler.step()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # 将输入数据移至 GPU
            labels = labels.to(device)  # 将标签移至 GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1

            if batch_count % print_every == 0:
                batch_loss = loss.item()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Loss: {batch_loss:.4f}")


            if batch_count % record_every == 0:
                batch_loss = loss.item()
                losses.append(batch_loss)
                

    return losses

def train_and_save_best_model(model, train_loader, criterion, optimizer, num_epochs, device, lr_scheduler, print_every=10, record_every=1, save_path='best_model.pth'):
    model.train()
    losses = []
    accuracies = []
    running_loss = 0.0
    correct = 0
    total = 0
    best_loss = float('inf')  # 初始化为正无穷大
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        if lr_scheduler is not None:
            # 在每个epoch开始前更新学习率
            lr_scheduler.step()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1

            if batch_count % print_every == 0:
                batch_loss = loss.item()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Loss: {batch_loss:.4f}")

            if batch_loss < best_loss:
                # 如果当前batch的loss更低，保存模型参数
                best_loss = batch_loss
                torch.save(model.state_dict(), save_path)

            if batch_count % record_every == 0:
                batch_loss = loss.item()
                losses.append(batch_loss)

    return losses


# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    # 计算测试集中每个图像的损失
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_samples += len(labels)

    average_loss = total_loss / num_samples
    print(f"Average Loss on Test Set: {average_loss:.4f}")

def visualize_random_images(model, test_loader, device, num_samples=9):
    model.eval()
    num_images = len(test_loader.dataset)
    indices = random.sample(range(num_images), num_samples)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)

    for i, idx in enumerate(indices):
        image, target = test_loader.dataset[idx]
        image = image.to(device)
        target = target.to(device)
        
        output = model(image.unsqueeze(0))
        predicted_coords = output.squeeze().detach().cpu().numpy()

        image = image.permute(1, 2, 0).cpu().numpy()

        axes[i // 3, i % 3].imshow(image)
        axes[i // 3, i % 3].set_title(f"Image {i + 1}")
        axes[i // 3, i % 3].axis('off')

        # 标注预测的坐标
        axes[i // 3, i % 3].scatter(predicted_coords[::2], predicted_coords[1::2], color='red', marker='x')

    plt.show()



# 可视化类别分布函数
def visualize_class_distribution(dataset,yline,title):
    # 统计每个类别的样本数量
    class_counts = Counter(dataset.labels)
    
    # 创建DataFrame以便绘制柱状图
    class_df = pd.DataFrame({'Class': class_counts.keys(), 'Count': class_counts.values()})
    class_df = class_df.sort_values('Class')
    
    # 使用Seaborn绘制柱状图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Class', y='Count', data=class_df)
    
    # 添加类别标签
    for index, row in class_df.iterrows():
        plt.text(row.name, row['Count'], str(row['Count']), ha='center', va='bottom')
    
    plt.axhline(y=yline, color='r', linestyle='--')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()




# 预测并可视化函数


def visualize_predictions(model, folder_path,device):
    # 定义预测结果的类别标签（根据您的实际情况进行修改）
    labels_eng = ['Speed limit (5km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (70km/h)',
               'Dont Go Left','Dont overtake from Left','No Car','No horn',
               'Go Right','keep Right','watch out for cars','Bicycles crossing',
               'Zebra Crossing','Children crossing','Go right or straight','Fences',
               'Stop','No stopping','No entry','To give away']
    
    labels_cn = ['限速 (5km/h)', '限速 (30km/h)', '限速 (50km/h)', '限速 (70km/h)',
               '禁止左转','禁止向左超车','禁止机动车驶入','禁止鸣笛',
               '右转','保持向右行驶','小心机动车','小心非机动车',
               '小心行人','小心儿童','直行或右转车道','铁道口',
               '停车让行','禁止停车','禁止驶入','让行']
    
    labels = labels_eng
    
    # 数据预处理和转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((124,124)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 获取文件夹中的所有图像文件路径
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png'))]

    # 从图像路径中随机选择20张图像
    selected_paths = random.sample(image_paths, 20)

    # 创建网格化可视化
    # 设置中文字体
    #plt.rcParams['font.family'] = 'SimHei'  # 设置使用黑体字体
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 12))
    fig.tight_layout()

    
    # 遍历选择的图像路径
    for i, image_path in enumerate(selected_paths):
        # 加载图像并应用转换
        image = Image.open(image_path).convert("RGB")
        trans_image = transform(image)

        # 增加一个维度，以符合模型输入的要求（batch_size=1）
        trans_image = trans_image.unsqueeze(0)
        # 将数据移动到计算设备上
        trans_image = trans_image.to(device)
        model = model.to(device)
        # 模型预测
        output = model(trans_image)
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.item()
        predicted_class = labels[predicted_label]

        # 可视化预测结果
        row = i // 5
        col = i % 5
        # 将图像转换为NumPy数组
        image = np.array(image)
        axes[row, col].imshow(image)
        axes[row, col].set_title(predicted_class)
        axes[row, col].axis('off')

    # 显示图像网格
    plt.show()



# 显示热力图函数

def visualize_activation_area(model, folder_path,device):
    # 设置模型为评估模式
    model.eval()

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 获取文件夹中的子文件夹列表
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 计算行数
    num_rows = len(subfolders)

    # 创建网格布局
    fig, axs = plt.subplots(num_rows, 2, figsize=(8, num_rows * 3))

    # 遍历每个子文件夹
    for i, subfolder in enumerate(subfolders):
        images = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            continue

        # 随机选择一张图像
        random_image = random.choice(images)

        # 加载图像
        image = Image.open(random_image).convert('RGB')

        # 预处理图像
        img_tensor = preprocess(image)
        img_tensor = img_tensor.unsqueeze(0)
        
        img_tensor = img_tensor.to(device)

        # 向前传递并计算梯度
        img_tensor.requires_grad_()

        output = model(img_tensor)
        target = torch.argmax(output)
        output[0, target].backward()

        # 提取梯度
        gradients = img_tensor.grad.squeeze()
        gradients = torch.abs(gradients).sum(dim=0)  # 求绝对值后按通道求和

        # 后处理梯度图
        gradients = gradients.detach().cpu().numpy()

        # 计算当前图像在网格中的位置
        row_idx = i

        # 可视化源图像
        axs[row_idx, 0].imshow(image)
        axs[row_idx, 0].axis('off')
        axs[row_idx, 0].set_title('Source Image')

        # 可视化热力图
        axs[row_idx, 1].imshow(gradients, cmap='gray')
        axs[row_idx, 1].axis('off')
        axs[row_idx, 1].set_title('Activation Area')

    plt.tight_layout()
    plt.show()


def visualize_filter_img(model,image_path,device):

    model.eval()

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(124),
        transforms.CenterCrop(124),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    input_batch = input_batch.to(device)
    # 使用模型获取特征
    with torch.no_grad():
        features = model.pretrained_model.conv1(input_batch)

    # 将特征可视化
    feature_maps = features.squeeze(0)
    num_channels = feature_maps.shape[0]
    rows = (num_channels - 1) // 8 + 1
    fig, axarr = plt.subplots(rows, 8, figsize=(16, rows*2))

    for idx, ax in enumerate(axarr.flat):
        if idx < num_channels:
            ax.imshow(feature_maps[idx].cpu(), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()




def visualize_fc_weights_cluster(fc_weights, num_clusters):

    # 创建t-SNE对象并将权重进行降维到2维
    tsne = TSNE(n_components=2)
    tsne_weights = tsne.fit_transform(fc_weights)

    # 使用K-means对降维后的权重进行聚类
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(tsne_weights)

    # 可视化聚类结果
    plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], c=labels)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Visualization of FC Weights Clusters')
    plt.show()