import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图片
# 设置日志记录
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{log_dir}/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 训练历史记录类
class TrainingHistory:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'learning_rates': [],
            'epoch_times': [],
            'class_accuracies': [],
            'confusion_matrices': []
        }
        self.start_time = time.time()
        
    def add_epoch(self, train_loss, test_loss, train_acc, test_acc, lr, epoch_time, class_acc=None, conf_matrix=None):
        self.history['train_loss'].append(train_loss)
        self.history['test_loss'].append(test_loss)
        self.history['train_acc'].append(train_acc)
        self.history['test_acc'].append(test_acc)
        self.history['learning_rates'].append(lr)
        self.history['epoch_times'].append(epoch_time)
        if class_acc:
            self.history['class_accuracies'].append(class_acc)
        if conf_matrix is not None:
            self.history['confusion_matrices'].append(conf_matrix.tolist())
    
    def save_history(self, filename='training_history.json'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_history_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"训练历史已保存到: {filename}")
        
    def plot_metrics(self):
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 损失曲线
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['test_loss'], label='Test Loss', marker='s')
        plt.title('train & test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        plt.subplot(2, 3, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(self.history['test_acc'], label='Test Accuracy', marker='s')
        plt.title('train & test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        plt.subplot(2, 3, 3)
        plt.plot(self.history['learning_rates'], marker='o', color='red')
        plt.title('Learning Rate Change')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 4. 每个epoch训练时间
        plt.subplot(2, 3, 4)
        plt.bar(range(len(self.history['epoch_times'])), self.history['epoch_times'])
        plt.title('Each Epoch Training Time')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # 5. 损失差异（过拟合检测）
        plt.subplot(2, 3, 5)
        loss_diff = np.array(self.history['test_loss']) - np.array(self.history['train_loss'])
        plt.plot(loss_diff, marker='o', color='purple')
        plt.title('Test-Train Loss Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 6. 准确率差异
        plt.subplot(2, 3, 6)
        acc_diff = np.array(self.history['train_acc']) - np.array(self.history['test_acc'])
        plt.plot(acc_diff, marker='o', color='orange')
        plt.title('Train-Test Accuracy Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Difference')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/training_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 绘制最终混淆矩阵
        if self.history['confusion_matrices']:
            self.plot_confusion_matrix(timestamp)
    
    def plot_confusion_matrix(self, timestamp):
        class_names = ['bloody', 'normal', 'porn', 'smoke']
        cm = np.array(self.history['confusion_matrices'][-1])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'results/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()


# 使用torchvision.transforms进行数据增强和预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),                                   # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(),                                   # 随机水平翻转
        transforms.RandomRotation(20),                                       # 随机旋转
        transforms.ToTensor(),                                               # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # 标准化
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 新增自定义数据集类处理损坏文件
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f"\n跳过损坏文件: {self.samples[index][0]}，错误: {e}")
                index = (index + 1) % len(self)

# 修改数据集创建部分
data_dir = 'harmful-video-dataset'
image_datasets = {
    x: SafeImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x]
    ) for x in ['train', 'test']
}

# 数据加载器（根据显存调整batch_size）
batch_size = 32  # 如果显存不足可减少到16
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size,
        shuffle=True, num_workers=4)
    for x in ['train', 'test']
}

# 初始化模型（使用预训练的ResNet）
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4个输出类别
model = model.to(device)

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# 在evaluate_model函数之后添加这个新函数

def generate_classification_report_table(all_labels, all_preds, class_names, test_acc, timestamp=None):
    """生成美观的分类报告表格图片"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 生成分类报告字典
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # 计算各类别准确率
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracies[class_name] = class_correct[i] / class_total[i]
        else:
            class_accuracies[class_name] = 0.0
    
    # 创建表格数据
    table_data = []
    for class_name in class_names:
        table_data.append([
            class_name,
            f"{report[class_name]['precision']:.4f}",
            f"{report[class_name]['recall']:.4f}",
            f"{report[class_name]['f1-score']:.4f}",
            f"{int(report[class_name]['support'])}"
        ])
    
    # 添加总体指标
    table_data.append(['', '', '', '', ''])  # 空行
    table_data.append([
        'accuracy',
        '',
        '',
        f"{report['accuracy']:.4f}",
        f"{int(report['macro avg']['support'])}"
    ])
    table_data.append([
        'macro avg',
        f"{report['macro avg']['precision']:.4f}",
        f"{report['macro avg']['recall']:.4f}",
        f"{report['macro avg']['f1-score']:.4f}",
        f"{int(report['macro avg']['support'])}"
    ])
    table_data.append([
        'weighted avg',
        f"{report['weighted avg']['precision']:.4f}",
        f"{report['weighted avg']['recall']:.4f}",
        f"{report['weighted avg']['f1-score']:.4f}",
        f"{int(report['weighted avg']['support'])}"
    ])
    
    # 创建DataFrame
    df = pd.DataFrame(table_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个表格：分类报告
    ax1.axis('tight')
    ax1.axis('off')
    
    # 创建表格
    table1 = ax1.table(cellText=df.values, colLabels=df.columns, 
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(df.columns)):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置类别行样式
    for i in range(1, len(class_names) + 1):
        table1[(i, 0)].set_facecolor('#E8F5E8')
        table1[(i, 0)].set_text_props(weight='bold')
    
    # 设置总体指标行样式
    for i in range(len(class_names) + 2, len(table_data) + 1):
        for j in range(len(df.columns)):
            table1[(i, j)].set_facecolor('#F0F0F0')
            if j == 0:  # 第一列加粗
                table1[(i, j)].set_text_props(weight='bold')
    
    ax1.set_title('Classification Report', fontsize=16, fontweight='bold', pad=20)
    
    # 第二个表格：各类别准确率
    class_acc_data = []
    for class_name in class_names:
        class_acc_data.append([
            class_name,
            f"{class_accuracies[class_name]:.4f}",
            f"{class_correct[class_names.index(class_name)]}/{class_total[class_names.index(class_name)]}"
        ])
    
    df_acc = pd.DataFrame(class_acc_data, columns=['Class', 'Accuracy', 'Correct/Total'])
    
    ax2.axis('tight')
    ax2.axis('off')
    
    table2 = ax2.table(cellText=df_acc.values, colLabels=df_acc.columns,
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(df_acc.columns)):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(class_names) + 1):
        table2[(i, 0)].set_facecolor('#E3F2FD')
        table2[(i, 0)].set_text_props(weight='bold')
        
        # 根据准确率设置颜色
        acc_value = class_accuracies[class_names[i-1]]
        if acc_value >= 0.9:
            color = '#C8E6C9'  # 绿色
        elif acc_value >= 0.8:
            color = '#FFF9C4'  # 黄色
        else:
            color = '#FFCDD2'  # 红色
        
        table2[(i, 1)].set_facecolor(color)
        table2[(i, 2)].set_facecolor(color)
    
    ax2.set_title('Individual Class Accuracies', fontsize=16, fontweight='bold', pad=20)
    
    # 添加整体信息
    info_text = f"Overall Test Accuracy: {test_acc:.4f}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # 保存图片
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/classification_report_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"分类报告表格已保存到: results/classification_report_{timestamp}.png")

# 修改evaluate_model函数，返回更多信息
def evaluate_model(model, dataloader, phase='test'):
    model.eval()  # 设置为评估模式
    running_corrects = 0
    total_samples = 0

    # 初始化类别统计字典
    class_correct = {i: 0 for i in range(4)}
    class_total = {i: 0 for i in range(4)}
    class_names = ['bloody', 'normal', 'porn', 'smoke']
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader[phase], desc=f'评估{phase}集'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 统计总体准确率
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            # 收集所有预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 统计每个类别的准确率
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    # 计算总体准确率
    overall_acc = running_corrects.double() / total_samples
    print(f'\n{phase}集总体准确率: {overall_acc:.4f}')

    # 打印每个类别的准确率
    print("\n各类别准确率:")
    for i in range(4):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_names[i]}: 无样本")

    # 生成分类报告表格图片
    generate_classification_report_table(all_labels, all_preds, class_names, overall_acc)

    return overall_acc


# 修改训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    history = TrainingHistory()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    logger.info(f"开始训练，总共 {num_epochs} 个epoch")
    logger.info(f"数据集大小 - 训练: {len(image_datasets['train'])}, 测试: {len(image_datasets['test'])}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        logger.info(f'开始第 {epoch} 个epoch')

        epoch_results = {}
        cm = None  # 初始化混淆矩阵

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0
            
            # 类别统计
            class_correct = {i: 0 for i in range(4)}
            class_total = {i: 0 for i in range(4)}
            all_preds = []
            all_labels = []

            # 使用tqdm进度条
            pbar = tqdm(dataloaders[phase], desc=phase)
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化仅在训练阶段进行
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 收集预测结果用于混淆矩阵
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 类别统计
                for label, pred in zip(labels, preds):
                    if label == pred:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1
                
                # 更新进度条信息
                current_loss = running_loss / ((pbar.n + 1) * batch_size)
                current_acc = running_corrects.double() / ((pbar.n + 1) * batch_size)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}'
                })

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            # 计算类别准确率
            class_acc = {}
            class_names = ['bloody', 'normal', 'porn', 'smoke']
            for i in range(4):
                if class_total[i] > 0:
                    class_acc[class_names[i]] = class_correct[i] / class_total[i]
                else:
                    class_acc[class_names[i]] = 0.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            logger.info(f'{phase} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
            
            # 记录结果
            epoch_results[f'{phase}_loss'] = epoch_loss
            epoch_results[f'{phase}_acc'] = float(epoch_acc)
            
            if phase == 'test':
                # 生成混淆矩阵
                cm = confusion_matrix(all_labels, all_preds)
                
                # 打印详细分类报告
                print("\n分类报告:")
                print(classification_report(all_labels, all_preds, target_names=class_names))
                
                # 打印类别准确率
                print("各类别准确率:")
                for class_name, acc in class_acc.items():
                    print(f"  {class_name}: {acc:.4f}")

            # 深拷贝最佳模型权重
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info(f"新的最佳准确率: {best_acc:.4f}")

        # 记录这个epoch的历史
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history.add_epoch(
            epoch_results['train_loss'], 
            epoch_results['test_loss'],
            epoch_results['train_acc'], 
            epoch_results['test_acc'],
            current_lr, 
            epoch_time,
            class_acc,
            cm
        )
        
        scheduler.step()
        
        print(f"Epoch时间: {epoch_time:.2f}秒")
        print(f"当前学习率: {current_lr:.2e}")
        print()

    total_time = time.time() - history.start_time
    logger.info(f"训练完成！总用时: {total_time:.2f}秒")
    logger.info(f"最佳验证准确率: {best_acc:.4f}")
    
    # 保存训练历史和绘制图表
    history.save_history()
    history.plot_metrics()

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

# 生成最终报告
def generate_final_report(model, history, test_acc):
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'results/training_report_{timestamp}.txt'
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("模型训练完整报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {device}\n")
        f.write(f"总训练轮数: {len(history.history['train_loss'])}\n")
        f.write(f"最佳验证准确率: {max(history.history['test_acc']):.4f}\n")
        f.write(f"最终测试准确率: {test_acc:.4f}\n")
        f.write(f"总训练时间: {sum(history.history['epoch_times']):.2f}秒\n")
        f.write(f"平均每轮时间: {np.mean(history.history['epoch_times']):.2f}秒\n")
        f.write("\n训练参数:\n")
        f.write(f"- 批大小: {batch_size}\n")
        f.write(f"- 初始学习率: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"- 优化器: {type(optimizer).__name__}\n")
        f.write(f"- 损失函数: {type(criterion).__name__}\n")
        f.write(f"- 数据增强: 是\n")
        
        # 过拟合分析
        final_train_acc = history.history['train_acc'][-1]
        final_test_acc = history.history['test_acc'][-1]
        overfitting_gap = final_train_acc - final_test_acc
        f.write(f"\n过拟合分析:\n")
        f.write(f"- 训练-验证准确率差: {overfitting_gap:.4f}\n")
        if overfitting_gap > 0.1:
            f.write("- 警告: 可能存在过拟合\n")
        else:
            f.write("- 模型泛化良好\n")
    
    logger.info(f"最终报告已保存到: {report_filename}")

# 开始训练（epoch数量根据实际情况调整）
model, training_history = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

# 在训练完成后添加评估代码
print("\n正在使用最佳模型评估测试集...")
final_model = model  # 已经加载了最佳权重
test_acc = evaluate_model(final_model, dataloaders)

# 创建模型保存目录
os.makedirs('models', exist_ok=True)

# 保存模型
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
torch.save(model.state_dict(), f'models/best_model_weights_{timestamp}_acc{test_acc:.4f}.pth')
torch.save(model, f'models/full_model_{timestamp}_acc{test_acc:.4f}.pth')

# 生成最终报告
generate_final_report(model, training_history, test_acc)

print("训练完成！所有结果已保存。")