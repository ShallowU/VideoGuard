# --------------- model_loader.py ---------------
import torch
import logging
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms
from funasr import AutoModel
from paddleocr import PaddleOCR
import sys
import os

# PyTorch性能优化设置
# torch.backends.cudnn.benchmark = True      # 图像尺寸固定(224x224)，这个很有效
# torch.backends.cudnn.deterministic = False # 生产环境，追求速度而非完全可重现性
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'   # 异步执行，提高并发性能

logging.getLogger('ppocr').setLevel(logging.ERROR)

"""
类：模型加载器
功能：加载各种模型，包括ResNet50、CLIP、ASR、OCR和暴力检测模型
"""
class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet_model = None
        self.clip_processor = None
        self.clip_model = None
        self.asr_model = None
        self.ocr_model = None
        self.violence_model = None

    def load_all_models(self):
        """加载所有需要的模型"""
        self._load_resnet_model()
        # self._load_clip_model() # 目前该clip暴力检测模型不用，使用violence_model
        self._load_asr_model()
        self._load_ocr_model()
        self._load_violence_model()

    def _load_resnet_model(self):
        """加载血腥/色情/烟雾/正常分类模型"""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.fc = torch.nn.Linear(512, 4)  # 适配原始代码结构
        model.load_state_dict(torch.load('ckpt/best_model_weights.pth', map_location=self.device))
        self.resnet_model = model.to(self.device).eval()

    def _load_clip_model(self):
        """加载暴力检测模型"""
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device).eval()

    def _load_asr_model(self):
        """加载ASR语音识别模型（保持原始抑制输出逻辑）"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        # self.asr_model = AutoModel(
        #     model="/home/violet/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        #     vad_model="/home/violet/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        #     punc_model="/home/violet/.cache/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        #     disable_update=True
        # )
        self.asr_model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            disable_update=True,
        )
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    def _load_ocr_model(self):
        """加载OCR模型（保持原始参数）"""
        self.ocr_model = PaddleOCR(
            use_gpu=True,
            det_batch_num=8 ,# 256,
            rec_batch_num=8 ,# 512,      
            lang='ch',
            use_angle_cls=False,
            det_db_thresh=0.3,
            rec_thresh=0.5,
            precision='fp16',
        )

    def _load_violence_model(self):
        """加载自定义暴力检测模型"""
        try:
            # 定义模型结构
            class MobileNetGRU(torch.nn.Module):
                def __init__(self, hidden_dim=512, num_classes=2, num_layers=1):
                    super(MobileNetGRU, self).__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    
                    # 更新为与训练时相同的初始化方式
                    self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                    self.mobilenet.classifier = torch.nn.Identity()  # 移除分类层
                    # GRU层
                    self.gru = torch.nn.GRU(
                        input_size=1280,
                        hidden_size=hidden_dim, 
                        num_layers=num_layers, 
                        batch_first=True
                    ) 
                    # 分类层
                    self.fc = torch.nn.Linear(hidden_dim, num_classes)
                
                def forward(self, x):
                    # 完整的前向传播实现，确保与训练时一致
                    batch_size, seq_length, c, h, w = x.size()
                    # 重塑为(batch_size * seq_length, c, h, w)
                    x = x.view(batch_size * seq_length, c, h, w)
                    with torch.no_grad():
                        x = self.mobilenet(x)
                    # 重塑回(batch_size, seq_length, 1280)
                    x = x.view(batch_size, seq_length, -1)
                    # GRU处理
                    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
                    x, _ = self.gru(x, h0)
                    # 分类
                    x = self.fc(x[:, -1, :])
                    return x
            
            # 创建模型
            model = MobileNetGRU()
            # 加载权重
            model.load_state_dict(torch.load('ckpt/best_violence_detect.pth', map_location=self.device))
            self.violence_model = model.to(self.device).eval()
            # 尝试进行一次前向传播测试          
        except Exception as e:
            print(f"加载暴力检测模型失败: {e}")
            self.violence_model = None