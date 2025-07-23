# VideoGuard - 基于大模型与多模态融合的短视频平台有害内容检测系统

## 项目简介

VideoGuard 是一个基于深度学习和多模态融合技术的智能视频内容检测系统，能够自动识别视频中的有害内容，包括暴力、色情、血腥、吸烟等违规行为，同时支持音频语音识别和图像文字提取，检测语音和文字中的违规内容，提供全方位的内容安全检测服务。

![首页](https://bu.dusays.com/2025/06/14/684c5d266e24a.png)

## 主要功能

- **多模态内容检测**：支持图像、音频、文本的综合分析
- **智能分类识别**：检测血腥、色情、暴力、吸烟、正常等内容类别
- **语音识别转录**：自动提取音频中的文字内容并进行安全检测
- **图像文字识别**：OCR技术提取视频帧中的文字信息
- **PDF报告生成**：生成专业的学术风格检测报告
- **多种输入方式**：支持本地文件上传、视频URL下载等
- **RESTful API**：提供完整的Web API接口

## 系统架构

本系统为前后端分离项目，需要**两台电脑进行分别部署前后端**，后端的电脑推荐选用性能较好的且含有GPU的电脑。

```
VideoGuard/
├── ckpt/                  # 模型权重文件
├── uploads/               # 前端上传视频、音频、文字
├── pre-train/             # 模型训练文件
├── log/                   # 日志记录
├── front-deploy/          # 前端部署
├── main.py                # Flask Web服务主入口
├── model_loader.py        # 模型加载管理器
├── video_processor.py     # 视频处理核心模块
├── pdf.py                 # PDF报告生成器
├── analyze_prob.py        # 概率分析工具
├── down_video.py          # 视频下载工具
├── util.py                # 文本分析工具
├── SimSun.ttf             # pdf报告生成字体包
├── requirements_all.txt   # 所有Python依赖包，如遇环境错误可以查看此文件查看版本是否正确
└── requirements.txt       # Python依赖包
```



## 安装指南

## A.后端部署

### 0.后端环境要求

- Python 3.9.21
- CUDA 12.1 (推荐使用GPU)
- FFmpeg
- 8GB+ RAM
- 最少4GB+ GPU显存 

### 1. 创建虚拟环境

```shell
# 使用conda (推荐)

conda create -n videoguard python=3.9.21

conda activate videoguard

# 或使用venv

python -m venv videoguard

source videoguard/bin/activate  # Linux/Mac
```

### 2. 安装python环境依赖

初始安装核心依赖即可

```shell
pip install -r requirements.txt
```

如果后续环境错误或者需要完整功能，则安装所有依赖

```shell
pip install -r requirements_all.txt
```

### 3. 安装系统依赖

```shell
# Ubuntu/Debian

sudo apt update

sudo apt install ffmpeg

# CentOS/RHEL

sudo yum install ffmpeg

# macOS

brew install ffmpeg

# Windows
下载 FFmpeg 并添加到 PATH 环境变量
```

### 4. 模型文件

模型文件放置在 **ckpt**目录下，如已有则无需操作跳到下一步骤

- `best_model_weights.pth` - ResNet图像分类模型
- `best_violence_detect.pth` - 暴力检测模型

### 5. 配置模型路径

在你安装[funasr](https://github.com/modelscope/FunASR)时，终端会给出安装到你ubuntu中的路径，类似如下：

```
            model="/home/你的用户名/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="/home/你的用户名/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="/home/你的用户名/.cache/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
```

编辑 model_loader.py：50-55行中的模型路径，确保ASR模型路径正确：

```python
# 在 _load_asr_model 方法中修改路径

self.asr_model = AutoModel(

  model="your/local/path/to/asr/model", # 修改为实际路径

  vad_model="your/local/path/to/vad/model",

  punc_model="your/local/path/to/punc/model",

  disable_update=True

)
```

--------------------------------------------------------------------------------------------





## B.前端部署

> [!IMPORTANT]
> 推荐在另一台电脑上进行部署前端

### 1. 环境要求

- Node.js 16.0+ 
- npm 8.0+ 或 yarn 1.22+
- 现代浏览器（Chrome 88+, Firefox 85+, Safari 14+）

### 2. 安装依赖

```shell
# 进入前端项目目录
cd front-deploy

# 安装依赖包
npm install

# 或使用yarn
yarn install
```

### 3. 配置后端API地址

编辑 `src/api/Api.js` 文件，确保API地址指向后端服务：

```javascript
// 修改为实际的后端服务地址
const SERVER_BASE = 'http://your-backend-ip:8000';
const SERVER_VIDEO = `${SERVER_BASE}/process`;
const SERVER_AUDIO = `${SERVER_BASE}/process_audio`;
const SERVER_TEXT = `${SERVER_BASE}/process_text`;
```

### 4. 启动开发服务器

```shell
# 启动开发模式
npm start

# 或使用yarn
yarn start
```

服务启动后会在 `http://localhost:3000` 提供Web界面。

### 5. 生产环境部署

#### 5.1 构建生产版本

```shell
# 构建生产版本
npm run build

# 或使用yarn
yarn build
```

构建完成后，会在 `build/` 目录下生成静态文件。

#### 5.2 部署静态文件

**方式一：使用serve**

```shell
# 全局安装serve
npm install -g serve

# 启动生产服务器
serve -s build -l 3000
```

**方式二：使用Nginx**

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /path/to/your/build;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }
    
    # 代理API请求到后端
    location /api/ {
        proxy_pass http://your-backend-ip:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 6. Docker部署（可选）

#### 6.1 创建Dockerfile

```dockerfile
# 使用Node.js作为构建环境
FROM node:16-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# 使用Nginx作为生产环境
FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 6.2 构建和运行容器

```shell
# 构建镜像
docker build -t videoguard-frontend .

# 运行容器
docker run -d -p 3000:80 --name videoguard-frontend videoguard-frontend
```

-----------------------------------------------



## 使用指南

### 1.启动服务

```
python main.py
```

服务启动后会在 `http://localhost:8000` 提供Web API接口。

### 2.只通过后端API接口调用说明(下面调用只是在前端未配置的情况测试，推荐配置前端后使用前端界面)

#### 1. 处理视频文件

```
curl -X POST -F "video=@your_video.mp4" http://localhost:8000/process
```

#### 2. 处理视频URL

```
curl -X POST -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/video.mp4"}' \
     http://localhost:8000/process_url
```

#### 3. 处理音频文件

```
curl -X POST -F "audio=@your_audio.wav" http://localhost:8000/process_audio
```

#### 4. 处理文本内容

```
curl -X POST -H "Content-Type: application/json" \
     -d '{"text":"要检测的文本内容"}' \
     http://localhost:8000/process_text
```

#### 5. 响应格式

example：

```
{
  "audio_violation": ["正常"],
  "text_violation": ["正常"],
  "audio_text": ["识别出的音频文字"],
  "ocr_text": ["识别出的图像文字"],
  "video_main_category": {
    "name": "normal",
    "confidence": 0.95
  },
  "video_second_category": {
    "name": "smoke",
    "confidence": 0.03
  },
  "violation_images": [...],
  "pdf_data": "base64编码的PDF报告"
}
```

### 3.使用前端界面使用

![首页](https://bu.dusays.com/2025/06/14/684c5d266e24a.png)

与上述调用api功能一致，不再赘述。



-------------------------------------------------------------



## 配置说明

### 主要配置参数(选改)

在 main.py 中可以调整以下参数：

```
CONFIG = {
    'FRAME_RATE': 10,           # 视频帧提取率
    'BATCH_SIZE': 16,           # 批处理大小
    'MAX_VIOLATION_IMAGES': 5,  # 最大违规图片数量
    'ASR_BATCH_SIZE': 300,      # ASR批处理大小
    'HOTWORD': '魔搭'          # ASR热词
}
```

### 检测阈值调整(选改)

在 video_processor.py中可以调整检测阈值：

```
# 暴力检测阈值
if res["violence_prob"] >= 0.7:  # 默认0.7，可调整
    categories["violence"].append((res["frame"], res["violence_prob"]))
```

--------------------------------------------------



## 故障排除

### 常见问题

1. **模型加载失败**

   \# 检查模型文件是否存在

   **ls** -la ckpt/

   \# 检查CUDA是否可用

   **python** -c "import torch; print(torch.cuda.is_available())"

2. **FFmpeg未找到**

   \# 检查FFmpeg是否安装

   **ffmpeg** -version

   \# 或安装FFmpeg

   **sudo** apt install ffmpeg

3. **内存不足**

   \# 在main.py中减小批处理大小

   CONFIG['BATCH_SIZE'] = 8 # 从16减少到8

### 日志查看

系统运行日志保存在 log 目录下：

```
tail -f log/nsfw_detect_*.log
```

### GPU加速

确保安装了正确的CUDA版本和PyTorch GPU版本，可以与requirements_all.txt对齐。

\# 检查CUDA版本

**nvcc** --version

------

**注意**：首次运行时系统会自动下载所需的预训练模型，可能需要较长时间和稳定的网络连接。建议在生产环境中提前下载并配置好所有模型文件。