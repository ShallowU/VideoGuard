import torch
import torch.nn as nn
from torchvision import models
from .attention import CBAM


class MobileNetCBAMGRU(nn.Module):
    """MobileNet + CBAM + GRU - 专注暴力检测"""
    def __init__(self, hidden_dim=512, num_classes=2, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 预训练的MobileNetV2骨干
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # 移除分类头，获取特征提取器
        self.features = self.backbone.features
        
        # 在最终特征上添加CBAM - 1280通道
        self.cbam = CBAM(1280, reduction_ratio=16)
        
        # 全局平均池化 - 将2D特征转为1D
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # GRU进行时序建模
        self.gru = nn.GRU(
            input_size=1280, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 简单的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def extract_frame_features(self, x):
        """提取单帧特征 - 带注意力增强"""
        # x: (B*T, C, H, W)
        features = self.features(x)  # (B*T, 1280, 7, 7)
        
        # 应用CBAM注意力
        enhanced_features = self.cbam(features)  # (B*T, 1280, 7, 7)
        
        # 全局池化得到每帧的特征向量
        frame_features = self.global_pool(enhanced_features).squeeze(-1).squeeze(-1)  # (B*T, 1280)
        
        return frame_features
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 重塑为处理所有帧
        x = x.view(batch_size * seq_len, *x.shape[2:])  # (B*T, C, H, W)
        
        # 提取增强特征
        with torch.no_grad():
            frame_features = self.extract_frame_features(x)  # (B*T, 1280)
        
        # 重塑回序列形式
        sequence_features = frame_features.view(batch_size, seq_len, -1)  # (B, T, 1280)
        
        # GRU处理时序信息
        gru_out, _ = self.gru(sequence_features)  # (B, T, hidden_dim)
        
        # 使用最后一个时间步的输出
        final_features = gru_out[:, -1, :]  # (B, hidden_dim)
        
        # 分类
        output = self.classifier(final_features)  # (B, num_classes)
        
        return output


def create_model(hidden_dim=512, num_classes=2):
    """工厂函数 - 创建模型实例"""
    return MobileNetCBAMGRU(hidden_dim, num_classes)