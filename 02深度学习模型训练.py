import warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image
from torchvision import datasets, models, transforms,utils
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from tqdm import tqdm
import logging
import torch.optim as optim
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.metrics import confusion_matrix
import seaborn as sns
# 配置日志记录
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 检查 CUDA 是否可用
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your installation.")
else:
    print("CUDA is available. Using GPU for training.")
#设置随机种子
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(20)

root = './'


num_epochs = 30   
batch_size = 10
learning_rate = 0.00005  
momentum = 0.96 
num_classes = len(os.listdir('F:/Machine learning/飞机型号识别/飞机型号识别/数据集')) 


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):  
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  
        imgs = [] 
        for line in fh:  
            line = line.rstrip() 
            words = line.split()  
            imgs.append((line[:-2], int(words[-1])))  
          
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  
        fn, label = self.imgs[index] 
        img = Image.open(fn).convert('RGB')  
        img = img.resize((224,224))

        if self.transform is not None:
            img = self.transform(img) 
        return img, label 

    def __len__(self): 
        return len(self.imgs)


class FocalLoss(nn.Module):  # 定义 Focal Loss 类
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)  # 计算交叉熵损失
        pt = torch.exp(-BCE_loss)  # 计算预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # 计算 Focal Loss

        if self.reduction == 'mean':
            return F_loss.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return F_loss.sum()  # 返回总损失
        else:
            return F_loss  # 返回原始损失


class SpatialAttentionLayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionLayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.spatial_attention = SpatialAttentionLayer()

    def forward(self, x):
        spatial_attention = self.spatial_attention(x)
        return x * spatial_attention


if __name__ == '__main__':
    # 定义数据转换，包括归一化
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机缩放裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = MyDataset(datatxt=root + 'train.txt', transform=transform)
    test_data = MyDataset(datatxt=root + 'val.txt', transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation.")
    else:
        print("CUDA is available. Using GPU for training.")

    # 使用 EfficientNet V3 作为基础模型
    class EfficientNetV3Model(nn.Module):
        def __init__(self, num_classes=num_classes):  
            super(EfficientNetV3Model, self).__init__()
            self.model = models.efficientnet_v2_l(pretrained=True)
            # 获取特征提取器
            self.features = self.model.features
            # 获取分类器的输入特征数
            in_features = self.model.classifier[-1].in_features
            # 添加注意力层
            self.attention = AttentionLayer(in_channels=1280)
            # 修改分类器
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )

        def forward(self, x):
            # 1. 首先通过特征提取器
            x = self.features(x)  # 这会得到特征图
            
            # 2. 应用注意力机制
            x = self.attention(x)
            
            # 3. 全局平均池化
            x = torch.mean(x, dim=[2, 3])
            
            # 4. 通过分类器
            x = self.classifier(x)
            
            return x

    net = EfficientNetV3Model(num_classes).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.1, gamma=2)  # 适合类别不平衡且样本差异小的情况

    # 使用 AdamW 优化器
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 用于记录损失和准确率
    train_losses = []
    test_losses = []
    test_accuracies = []
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loader = tqdm(train_loader)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录每个批次的损失
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # 每10批次打印一次
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0  # 重置损失

        # 更新学习率
        scheduler.step()

        net.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = net(batch_x)
                loss2 = criterion(out, batch_y)
                test_loss += loss2.item()
                pred = torch.max(out, 1)[1]
                num_correct = (pred == batch_y).sum()
                test_acc += num_correct.item()
            test_losses.append(test_loss / len(test_data))
            test_accuracies.append(test_acc / len(test_data))
            print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, test_loss / (len(test_data)), test_acc / (len(test_data))))
            logging.info(f'Epoch {epoch + 1}, Test Loss: {(test_loss / len(test_data)):.4f}, Test acc: {(test_acc / len(test_data)):.4f}')    

        torch.save(net.state_dict(), 'model.ckpt')

    # 绘制损失和准确率变化图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    plt.savefig('training_results.png')  # 保存图像
    plt.show()  # 显示图像

    # 获取类别名称
    class_names = os.listdir('F:/Machine learning/飞机型号识别/飞机型号识别/数据集')

    # 计算混淆矩阵
    all_preds = []
    all_labels = []
    net.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = net(batch_x)
            pred = torch.max(out, 1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  # 使用类别名称
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # 保存混淆矩阵图像
    plt.show()  # 显示混淆矩阵图像

    # 可视化最终训练出的网络
    def visualize_model(model, num_images=12):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
        class_samples = {class_name: 0 for class_name in class_names}  # 记录每个类别的样本数量
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for j in range(inputs.size()[0]):
                    label = class_names[labels[j].item()]
                    if class_samples[label] < num_images // len(class_names):  # 每个类别最多显示指定数量的样本
                        images_so_far += 1
                        ax = plt.subplot(num_images // 2, 2, images_so_far)
                        ax.imshow(inputs.cpu().data[j].permute(1, 2, 0).clamp(0, 1))  # 确保图像在[0, 1]范围内
                        ax.set_title(f'Predicted: {class_names[preds[j].item()]}, Actual: {label}')  # 使用类别名称
                        ax.axis('off')  # 关闭坐标轴
                        class_samples[label] += 1  # 更新样本数量
                        if images_so_far == num_images:
                            model.train(mode=was_training)
                            plt.show()  # 显示图像
                            return
        model.train(mode=was_training)
        plt.show()  # 显示图像

    visualize_model(net)  # 可视化最终训练出的网络
