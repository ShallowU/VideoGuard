import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import seaborn as sns
from datetime import datetime
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameters
img_height, img_width = 224, 224
batch_size = 16
epochs = 10
learning_rate = 0.001
sequence_length = 10
dataset_path = '/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create experiment directory
experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"experiment_{experiment_time}"
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(f"{experiment_dir}/plots", exist_ok=True)
os.makedirs(f"{experiment_dir}/models", exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(f'{experiment_dir}/tensorboard_logs')

# Transformations
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_frames(video_path, sequence_length, interval=10):
    """Extract frames from a video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (img_height, img_width))
            frames.append(frame)
        frame_count += 1
    cap.release()

    # Handle cases where there are fewer frames than sequence_length
    if len(frames) < sequence_length:
        while len(frames) < sequence_length:
            frames.append(frames[-1] if frames else cv2.resize(np.zeros((img_height, img_width, 3), dtype=np.uint8), (img_height, img_width)))
    else:
        frames = frames[:sequence_length]

    return frames


class VideoDataset(Dataset):
    """Dataset class for video data"""
    def __init__(self, video_files, labels, transform=None, sequence_length=10):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        frames = extract_frames(self.video_files[idx], self.sequence_length)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        label = self.labels[idx]
        return torch.stack(frames), label


def load_dataset(dataset_path):
    """Load dataset from directory structure"""
    video_files = []
    labels = []
    classes = os.listdir(dataset_path)
    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        videos = os.listdir(class_path)
        for video_name in videos:
            video_path = os.path.join(class_path, video_name)
            video_files.append(video_path)
            labels.append(label)
    return video_files, labels


class MobileNetGRU(nn.Module):
    """MobileNet + GRU model for video classification"""
    def __init__(self, hidden_dim, num_classes, num_layers=1):
        super(MobileNetGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier = nn.Identity()  # Remove the last fully connected layer

        # GRU
        self.gru = nn.GRU(input_size=1280, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()

        # Reshape to (batch_size * seq_length, c, h, w) to feed each frame into MobileNet
        x = x.view(batch_size * seq_length, c, h, w)
        with torch.no_grad():
            x = self.mobilenet(x)
        
        # Reshape back to (batch_size, seq_length, 1280) to feed into GRU
        x = x.view(batch_size, seq_length, -1)

        # GRU
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        x, _ = self.gru(x, h0)

        # Classification
        x = self.fc(x[:, -1, :])  # Use the output of the last GRU cell

        return x


class ExperimentTracker:
    """Track and record experiment metrics"""
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': [],
            'model_params': None,
            'best_val_acc': 0,
            'best_epoch': 0
        }
        self.start_time = time.time()
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Log metrics for an epoch"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rates'].append(lr)
        self.metrics['epoch_times'].append(epoch_time)
        
        # Update best metrics
        if val_acc > self.metrics['best_val_acc']:
            self.metrics['best_val_acc'] = val_acc
            self.metrics['best_epoch'] = epoch
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', lr, epoch)
        
    def save_metrics(self):
        """Save all metrics to files"""
        # Save as JSON
        with open(f'{self.experiment_dir}/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save as CSV
        df = pd.DataFrame({
            'epoch': range(len(self.metrics['train_loss'])),
            'train_loss': self.metrics['train_loss'],
            'train_acc': self.metrics['train_acc'],
            'val_loss': self.metrics['val_loss'],
            'val_acc': self.metrics['val_acc'],
            'learning_rate': self.metrics['learning_rates'],
            'epoch_time': self.metrics['epoch_times']
        })
        df.to_csv(f'{self.experiment_dir}/training_log.csv', index=False)


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_model_info(model, experiment_dir):
    """Save model architecture and parameters info"""
    total_params, trainable_params = count_parameters(model)
    
    model_info = {
        'model_name': 'MobileNetGRU',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': str(model)
    }
    
    with open(f'{experiment_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")
    return model_info


def plot_confusion_matrix(y_true, y_pred, class_names, experiment_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_scores, experiment_dir):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, tracker=None):
    """Enhanced training function with comprehensive monitoring"""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Log batch progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        if tracker:
            tracker.log_epoch(epoch, epoch_loss, epoch_acc.item(), val_epoch_loss, 
                            val_epoch_acc.item(), current_lr, epoch_time)
        
        logger.info(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, '
                   f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, '
                   f'LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')

    return model, tracker.metrics if tracker else None


def evaluate_model_comprehensive(model, test_loader, class_names, experiment_dir):
    """Comprehensive model evaluation with detailed metrics"""
    model.eval()
    test_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Save detailed metrics
    with open(f'{experiment_dir}/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, experiment_dir)
    
    # Plot ROC curve (for binary classification)
    if len(class_names) == 2:
        y_scores = [prob[1] for prob in all_probs]  # Probability of positive class
        roc_auc = plot_roc_curve(all_labels, y_scores, experiment_dir)
        logger.info(f'Test Accuracy: {test_acc:.4f}, ROC AUC: {roc_auc:.4f}')
    else:
        logger.info(f'Test Accuracy: {test_acc:.4f}')
    
    return test_acc, report


def plot_training_history_enhanced(metrics, experiment_dir):
    """Enhanced training history plots"""
    epochs_range = range(len(metrics['train_loss']))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs_range, metrics['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs_range, metrics['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs_range, metrics['train_acc'], 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs_range, metrics['val_acc'], 'r-', label='Val Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[0, 2].plot(epochs_range, metrics['learning_rates'], 'g-')
    axes[0, 2].set_xlabel('Epochs')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True)
    
    # Epoch time plot
    axes[1, 0].plot(epochs_range, metrics['epoch_times'], 'm-')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Epoch Training Time')
    axes[1, 0].grid(True)
    
    # Loss difference plot
    loss_diff = np.array(metrics['val_loss']) - np.array(metrics['train_loss'])
    axes[1, 1].plot(epochs_range, loss_diff, 'orange')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Accuracy difference plot
    acc_diff = np.array(metrics['train_acc']) - np.array(metrics['val_acc'])
    axes[1, 2].plot(epochs_range, acc_diff, 'purple')
    axes[1, 2].set_xlabel('Epochs')
    axes[1, 2].set_ylabel('Train Acc - Val Acc')
    axes[1, 2].set_title('Accuracy Gap')
    axes[1, 2].grid(True)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/plots/training_history_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_experiment_summary(metrics, model_info, test_results, experiment_dir):
    """Save comprehensive experiment summary"""
    summary = {
        'experiment_info': {
            'timestamp': experiment_time,
            'device': str(device),
            'total_epochs': len(metrics['train_loss']),
            'best_epoch': metrics['best_epoch'],
            'best_val_accuracy': metrics['best_val_acc'],
            'total_training_time': sum(metrics['epoch_times'])
        },
        'model_info': model_info,
        'final_metrics': {
            'final_train_loss': metrics['train_loss'][-1],
            'final_train_acc': metrics['train_acc'][-1],
            'final_val_loss': metrics['val_loss'][-1],
            'final_val_acc': metrics['val_acc'][-1],
            'test_accuracy': float(test_results) if isinstance(test_results, torch.Tensor) else test_results
        },
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'sequence_length': sequence_length,
            'img_height': img_height,
            'img_width': img_width
        }
    }
    
    with open(f'{experiment_dir}/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)


def main():
    """Enhanced main training pipeline"""
    logger.info(f"Starting experiment: {experiment_time}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_dir)
    
    # Load dataset
    video_files, labels = load_dataset(dataset_path)
    class_names = ['NonViolence', 'Violence']  # Adjust based on your dataset
    
    # Split dataset
    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        video_files, labels, test_size=0.3, random_state=42
    )
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=0.5, random_state=42
    )
    
    # Create datasets
    train_dataset = VideoDataset(train_videos, train_labels, transform=data_transforms, sequence_length=sequence_length)
    val_dataset = VideoDataset(val_videos, val_labels, transform=data_transforms, sequence_length=sequence_length)
    test_dataset = VideoDataset(test_videos, test_labels, transform=data_transforms, sequence_length=sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    hidden_dim = 512
    num_classes = 2
    model = MobileNetGRU(hidden_dim, num_classes).to(device)
    
    # Save model info
    model_info = save_model_info(model, experiment_dir)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    logger.info("Starting training...")
    model, metrics = train_model(model, criterion, optimizer, train_loader, val_loader, epochs, tracker)
    
    # Save training metrics
    tracker.save_metrics()
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_acc, test_report = evaluate_model_comprehensive(model, test_loader, class_names, experiment_dir)
    
    # Plot enhanced training history
    plot_training_history_enhanced(metrics, experiment_dir)
    
    # Save model
    model_path = f'{experiment_dir}/models/best_violence_detect.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save experiment summary
    save_experiment_summary(metrics, model_info, test_acc, experiment_dir)
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info(f"Training completed! Results saved in: {experiment_dir}")
    logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.4f} at epoch {metrics['best_epoch']}")
    logger.info(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()