# --------------- video_processor.py ---------------
import os
# import cv2
import tqdm
import time
import torch
import re
import torchaudio
import numpy as np
from PIL import Image
from typing import List, Dict
from torchvision import transforms
from collections import deque
from difflib import SequenceMatcher
from util import  deepseek_text
# 配置常量
CONFIG = {
    'FRAME_RATE': 8,                # 视频处理帧率
    'BATCH_SIZE': 32,                # 图像处理批量大小  
    'MAX_VIOLATION_IMAGES': 5,      # 最大违规图片展示数量
    'ASR_BATCH_SIZE': 300,          # ASR语音识别批量大小，初始300
    'HOTWORD': '魔搭',               # ASR语音识别热词
    'NUM_WORKERS': 4                # 数据加载器工作进程数
}
"""
视频处理类，负责视频帧提取、OCR、音频处理、暴力检测等功能
"""
class VideoProcessor:
    def __init__(self, video_path: str, model_loader, frame_rate: int = 4, batch_size: int = 32):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.batch_size = batch_size
        self.ml = model_loader  # 使用预加载的模型加载器实例
        self.device = model_loader.device
        self.audio_text = ""
        self.image_txt = ""
        
        # 初始化路径（完全保持原始路径结构）
        self.video_dir = os.path.dirname(video_path)
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.image_folder = os.path.join(self.video_dir, f"data/video-frame/{self.video_name}_frames")
        self.audio_file = os.path.join(self.video_dir, f"data/audio/{self.video_name}.wav")
        self.output_file = os.path.join(self.video_dir, f"data/txt/{self.video_name}_ImageDetectInfo.txt")
        self.audiotext_file = os.path.join(self.video_dir, f"data/audio-txt/{self.video_name}_audio.txt")
        self.violateAudio_file = os.path.join(self.video_dir, f"data/audio-txt/{self.video_name}_violate_audio.txt")
        self.ocrtext_file = os.path.join(self.video_dir, f"data/ocr-txt/{self.video_name}_ocr.txt")
        self.violateOcr_file = os.path.join(self.video_dir, f"data/ocr-txt/{self.video_name}_violate_ocr.txt")
        self.frontbackend_data_file = os.path.join(self.video_dir, f"data/frontbackend-data/{self.video_name}_frontbackend_data.json")
        # 创建目录（保持原始目录结构）
        os.makedirs(os.path.join(self.video_dir, "data/video-frame"), exist_ok=True)
        os.makedirs(os.path.join(self.video_dir, "data/audio"), exist_ok=True)
        os.makedirs(os.path.join(self.video_dir, "data/txt"), exist_ok=True)
        os.makedirs(os.path.join(self.video_dir, "data/audio-txt"), exist_ok=True)
        os.makedirs(os.path.join(self.video_dir, "data/ocr-txt"), exist_ok=True)
        os.makedirs(os.path.join(self.video_dir, "data/frontbackend-data"), exist_ok=True)
        # 图像预处理（保持原始参数）
        self.resnet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 确保所有模型已加载
        if not all([self.ml.resnet_model,  self.ml.asr_model, self.ml.ocr_model,self.ml.violence_model]):
            print("警告：部分模型未预加载，正在加载缺失的模型...")
            self.ml.load_all_models()

    # def extract_frames(self):
    #     """使用CUDA加速的FFmpeg提取视频帧"""
    #     start_time = time.time()
    #     if os.path.exists(self.image_folder):
    #         os.system(f"rm -rf {self.image_folder}")
    #     os.makedirs(self.image_folder, exist_ok=True)
        
    #     # 使用CUDA硬件解码和缩放
    #     cmd = f"ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {self.video_path} -vf 'fps={self.frame_rate},scale_cuda=224:224:format=nv12,hwdownload,format=nv12,format=rgb24' -f image2 {self.image_folder}/%05d.png -loglevel error"
    #     os.system(cmd)
    #     print(f"该视频帧已提取至: {self.image_folder}, 用时: {time.time() - start_time:.2f}秒")

    # def extract_audio(self):
    #     """使用CUDA加速的音频提取"""
    #     start_time = time.time()
    #     # 使用CUDA硬件解码
    #     cmd = f"ffmpeg -hwaccel cuda -y -i {self.video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {self.audio_file} -loglevel error"
    #     os.system(cmd)
    #     print(f"音频已提取至: {self.audio_file}, 用时: {time.time() - start_time:.2f}秒")

    def extract_frames(self):
        """使用兼容的CUDA加速FFmpeg提取视频帧"""
        start_time = time.time()
        if os.path.exists(self.image_folder):
            os.system(f"rm -rf {self.image_folder}")
        os.makedirs(self.image_folder, exist_ok=True)
        
        # 首先尝试CUDA硬件解码（不使用scale_cuda）
        cuda_cmd = f"ffmpeg -hwaccel cuda -i '{self.video_path}' -vf 'fps={self.frame_rate},scale=224:224' -f image2 '{self.image_folder}/%05d.png' -loglevel error"
        
        print("使用CUDA硬件解码提取视频帧...")
        exit_code = os.system(cuda_cmd)
        
        if exit_code == 0:
            print(f"CUDA硬件解码帧提取完成: {self.image_folder}, 用时: {time.time() - start_time:.2f}秒")
        else:
            print("CUDA硬件解码失败，回退到CPU模式...")
            # CPU回退模式
            cpu_cmd = f"ffmpeg -i '{self.video_path}' -vf 'fps={self.frame_rate},scale=224:224' -f image2 '{self.image_folder}/%05d.png' -loglevel error"
            os.system(cpu_cmd)
            print(f"CPU模式帧提取完成: {self.image_folder}, 用时: {time.time() - start_time:.2f}秒")

    def extract_audio(self):
        """使用兼容的CUDA加速音频提取"""
        start_time = time.time()
        
        # 使用CUDA硬件解码音频
        cuda_cmd = f"ffmpeg -hwaccel cuda -y -i '{self.video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 '{self.audio_file}' -loglevel error"
        
        print("使用CUDA硬件解码提取音频...")
        exit_code = os.system(cuda_cmd)
        
        if exit_code == 0:
            print(f"CUDA硬件解码音频提取完成: {self.audio_file}, 用时: {time.time() - start_time:.2f}秒")
        else:
            print("CUDA音频解码失败，回退到CPU模式...")
            # CPU回退模式
            cpu_cmd = f"ffmpeg -y -i '{self.video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 '{self.audio_file}' -loglevel error"
            os.system(cpu_cmd)
            print(f"CPU模式音频提取完成: {self.audio_file}, 用时: {time.time() - start_time:.2f}秒")
    # def extract_frames(self):
    #     """使用FFmpeg提取视频帧（保持原始实现）"""
    #     start_time = time.time()
    #     if os.path.exists(self.image_folder):
    #         os.system(f"rm -rf {self.image_folder}")
    #     os.makedirs(self.image_folder, exist_ok=True)
        
    #     cmd = f"ffmpeg -i {self.video_path} -vf fps={self.frame_rate} {self.image_folder}/%05d.png -loglevel error"
    #     os.system(cmd)
    #     print(f"该视频帧已提取至: {self.image_folder}, 用时: {time.time() - start_time:.2f}秒")
    
    def extract_text_from_image(self):
        """OCR文字提取（保持原始逻辑）"""
        start_time = time.time()
        image_files = sorted(os.listdir(self.image_folder))  
        selected_images = image_files[::8]
        total = len(selected_images)     
        last_texts = deque(maxlen=3)
        
        pbar = tqdm.tqdm(total=total)
        for img_name in selected_images:
            img_path = os.path.join(self.image_folder, img_name)
            result = self.ml.ocr_model.ocr(img_path, cls=True)
            texts = self._extract_ocr_text(result)
            current_text = ','.join(texts)
            
            if not any(SequenceMatcher(None, current_text, prev).ratio() > 0.9 for prev in last_texts):
                self.image_txt += current_text + ","
                last_texts.append(current_text)
            
            pbar.update(1)
        pbar.close()
        with open(self.ocrtext_file, "w") as f:
            f.write(self.image_txt.strip(','))
        print(f"该视频的图像转文本结果: {self.image_txt}\n耗时: {time.time()-start_time:.2f}s")

    def _extract_ocr_text(self, result):
        """OCR结果解析（保持原始实现）"""
        texts = []
        for item in result:
            if isinstance(item, list):
                texts.extend(self._extract_ocr_text(item))
            elif isinstance(item, tuple) and len(item) >= 2:
                if isinstance(item[0], str):
                    texts.append(item[0])
        return texts

    # def extract_audio(self):
    #     """音频提取（保持原始FFmpeg命令）"""
    #     start_time = time.time()
    #     cmd = f"ffmpeg -y -i {self.video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {self.audio_file} -loglevel error"
    #     os.system(cmd)
    #     print(f"音频已提取至: {self.audio_file}, 用时: {time.time() - start_time:.2f}秒")

    def _process_batch(self, image_batch: List[Image.Image], batch_paths=None):
        """批量处理（使用真实帧序列进行暴力检测）"""
        # ResNet处理（保持不变）
        resnet_tensors = torch.stack([self.resnet_transform(img) for img in image_batch]).to(self.device)
        with torch.no_grad():
            resnet_outputs = self.ml.resnet_model(resnet_tensors)
            resnet_probs = torch.softmax(resnet_outputs, dim=1).cpu().numpy()

        # 尝试使用自定义暴力检测模型
        if hasattr(self.ml, 'violence_model') and self.ml.violence_model is not None:
            sequence_length = 10  # 与训练时保持一致
            violence_probs = []
            
            # 如果没有提供批处理路径，无法获取序列信息
            if batch_paths is None:
                print("警告：未提供批处理路径，暴力检测可能不准确")
                batch_paths = ["unknown"] * len(image_batch)
            
            # 获取所有视频帧路径（全局）
            all_frame_paths = sorted([
                os.path.join(self.image_folder, f) 
                for f in os.listdir(self.image_folder) 
                if f.endswith(".png")
            ])
            
            # MobileNet变换
            mobilenet_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # 为每个帧构建序列并分析
            for i, img_path in enumerate(batch_paths):
                try:
                    # 获取当前帧在所有帧中的位置
                    if os.path.exists(img_path):
                        frame_idx = all_frame_paths.index(img_path)
                    else:
                        # 如果路径不存在，尝试通过文件名匹配
                        frame_name = os.path.basename(img_path) if img_path != "unknown" else None
                        if frame_name:
                            matching_frames = [f for f in all_frame_paths if os.path.basename(f) == frame_name]
                            if matching_frames:
                                frame_idx = all_frame_paths.index(matching_frames[0])
                            else:
                                # 如果找不到匹配，使用批次中的索引作为粗略估计
                                frame_idx = min(i, len(all_frame_paths)-1)
                        else:
                            # 完全未知的情况，使用批次中的索引
                            frame_idx = min(i, len(all_frame_paths)-1)
                    
                    # 构建以当前帧为中心的序列
                    half_seq = sequence_length // 2
                    start_idx = max(0, frame_idx - half_seq)
                    end_idx = min(len(all_frame_paths), start_idx + sequence_length)
                    
                    # 如果序列末尾超出范围，从前面补充
                    if end_idx - start_idx < sequence_length:
                        start_idx = max(0, end_idx - sequence_length)
                    
                    # 获取序列帧的路径
                    seq_paths = all_frame_paths[start_idx:end_idx]
                    
                    # 加载和转换序列帧
                    seq_tensors = []
                    for path in seq_paths:
                        try:
                            img = Image.open(path).convert("RGB")
                            seq_tensors.append(mobilenet_transform(img))
                        except Exception as e:
                            # 加载失败时使用当前批次中的图像
                            print(f"加载序列帧失败: {e}，使用当前帧替代")
                            seq_tensors.append(mobilenet_transform(image_batch[i]))
                    
                    # 如果序列长度不足，复制最后一帧
                    while len(seq_tensors) < sequence_length:
                        if seq_tensors:
                            seq_tensors.append(seq_tensors[-1])
                        else:
                            seq_tensors.append(mobilenet_transform(image_batch[i]))
                    
                    # 转换为批处理张量
                    tensor_sequence = torch.stack(seq_tensors).unsqueeze(0).to(self.device)  # [1, seq_len, 3, 224, 224]
                    
                    # MobileNet特征提取
                    batch_size, seq_len, c, h, w = tensor_sequence.size()
                    tensor_flat = tensor_sequence.view(batch_size * seq_len, c, h, w)
                    
                    with torch.no_grad():
                        # 特征提取
                        features = self.ml.violence_model.mobilenet(tensor_flat)
                        features = features.view(batch_size, seq_len, -1)
                        
                        # GRU处理
                        h0 = torch.zeros(
                            self.ml.violence_model.num_layers, 
                            batch_size, 
                            self.ml.violence_model.hidden_dim
                        ).to(self.device)
                        
                        seq_output, _ = self.ml.violence_model.gru(features, h0)
                        output = self.ml.violence_model.fc(seq_output[:, -1, :])
                        prob = torch.softmax(output, dim=1)[:, 1].item()  # 暴力类别的概率
                        violence_probs.append(prob)
                
                except Exception as e:
                    print(f"暴力检测序列处理失败: {e}，使用备选方法单帧多次复制")
                    # 备选方法：使用单帧多次复制
                    img = image_batch[i]
                    sequence = [img] * sequence_length
                    tensor_sequence = torch.stack([mobilenet_transform(frame) for frame in sequence])
                    tensor_sequence = tensor_sequence.unsqueeze(0).to(self.device)
                    
                    batch_size, seq_len, c, h, w = tensor_sequence.size()
                    tensor_flat = tensor_sequence.view(batch_size * seq_len, c, h, w)
                    
                    with torch.no_grad():
                        features = self.ml.violence_model.mobilenet(tensor_flat)
                        features = features.view(batch_size, seq_len, -1)
                        
                        h0 = torch.zeros(
                            self.ml.violence_model.num_layers, 
                            batch_size, 
                            self.ml.violence_model.hidden_dim
                        ).to(self.device)
                        
                        seq_output, _ = self.ml.violence_model.gru(features, h0)
                        output = self.ml.violence_model.fc(seq_output[:, -1, :])
                        prob = torch.softmax(output, dim=1)[:, 0].item()
                        violence_probs.append(prob)
            
            violence_probs = np.array(violence_probs)
        else:
            print("警告：暴力检测模型未加载，无法进行暴力检测")
        # 在返回前添加以下代码
        # if len(violence_probs) > 0:
        #     print(f"暴力概率分布: 最小={np.min(violence_probs):.4f}, 最大={np.max(violence_probs):.4f}, 平均={np.mean(violence_probs):.4f}")
        #     print(f"超过0.5的帧数: {np.sum(violence_probs > 0.5)}/{len(violence_probs)}")
        #     print(f"超过0.8的帧数: {np.sum(violence_probs > 0.8)}/{len(violence_probs)}")

        return resnet_probs, violence_probs

    def analyze_frames(self):
        """帧分析（传递帧路径给处理函数）"""
        frame_paths = sorted([
            os.path.join(self.image_folder, f) 
            for f in os.listdir(self.image_folder) 
            if f.endswith(".png")
        ])
        
        start_time = time.time()
        results = []
        pbar = tqdm.tqdm(total=len(frame_paths), desc="分析视频帧")
        
        for i in range(0, len(frame_paths), self.batch_size):
            batch_paths = frame_paths[i:i+self.batch_size]
            try:
                image_batch = [Image.open(p).convert("RGB") for p in batch_paths]
                # 将帧路径传递给处理函数
                resnet_probs, violence_probs = self._process_batch(image_batch, batch_paths=batch_paths)
                # from concurrent.futures import ThreadPoolExecutor
                # with ThreadPoolExecutor(max_workers=CONFIG.get('NUM_WORKERS', 4)) as executor:
                #     image_batch = list(executor.map(
                #         lambda p: Image.open(p).convert("RGB"), 
                #         batch_paths
                #     ))
                # resnet_probs, violence_probs = self._process_batch(image_batch, batch_paths=batch_paths)

                for idx, path in enumerate(batch_paths):
                    class_id = np.argmax(resnet_probs[idx])
                    results.append({
                        "frame": os.path.basename(path),
                        "class": ["bloody", "normal", "porn", "smoke"][class_id],
                        "confidence": resnet_probs[idx][class_id],
                        "violence_prob": violence_probs[idx]
                    })
                
                pbar.update(len(batch_paths))
            except Exception as e:
                print(f"处理批次 {i//self.batch_size} 出错: {str(e)}")
        
        pbar.close()
        print(f"视频帧分析完成，用时: {time.time() - start_time:.2f}秒")
        return results

    def generate_report(self, results: List[Dict]):
        """报告生成（保持原始实现）"""
        categories = {
            "bloody": [],
            "normal": [],
            "porn": [],
            "smoke": [],
            "violence": []
        }

        for res in results:
            categories[res["class"]].append((res["frame"], res["confidence"]))
            if res["violence_prob"] >= 0.7: # 0.7为阈值,增大阈值使其他的类别更容易被识别
                categories["violence"].append((res["frame"], res["violence_prob"]))

        with open(self.output_file, "w") as f:
            for cat in categories:
                f.write(f"\n{'#'*37}\n# {cat.capitalize().ljust(16)} \n")
                for frame, prob in categories[cat]:
                    metric = "暴力概率" if cat == "violence" else "置信度"
                    f.write(f"{frame} ({metric}: {prob:.2%})\n")
        
        print(f"检测报告已生成: {self.output_file}")

    def audio_distinguish(self):
        """语音识别（适配模型加载器）"""
        start_time = time.time()
        result = self.ml.asr_model.generate(input=self.audio_file, batch_size_s=300, hotword='魔搭')
        print(f"音频处理时间: {time.time() - start_time:.2f}秒")
        for segment in result:
            self.audio_text += segment['text'] + " "
        self.audio_text = self.audio_text.strip()
        with open(self.audiotext_file, "w") as f:
            f.write(self.audio_text)
            
    # 以下方法保持原始实现（predict_audiotext等）
    def predict_audiotext(self):
        audio_phrases = re.split(r'[，。]', self.audio_text)
        audio_phrases = [phrase.strip() for phrase in audio_phrases if phrase.strip()]
        categories_json, texts_json = deepseek_text(audio_phrases)
        return categories_json, texts_json

    def predict_ocrtext(self):
        image_words = self.image_txt.split(',')
        image_words = [word.strip() for word in image_words if word.strip()]
        categories_json, texts_json = deepseek_text(image_words)
        return categories_json, texts_json

    def process(self):
        """完整处理流程（保持原始调用顺序）"""
        self.extract_frames()
        analysis_results = self.analyze_frames()
        self.generate_report(analysis_results)