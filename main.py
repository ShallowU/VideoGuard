from flask import Flask, request, jsonify
import argparse
import time
import os
import sys
import base64  
import re
import json
import logging
import shutil
from datetime import datetime
from model_loader import ModelLoader
from video_processor import VideoProcessor
from analyze_prob import parse_report, calculate_dominant_category, calculate_category_confidence
from util import deepseek_text
from down_video import download_video
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

fps={
    'small': 8,    # 小帧率
    'medium': 16,  # 中帧率
    'large': 24,   # 大帧率
}
# 配置常量
CONFIG = {
    'FRAME_RATE': 4,                # 视频处理帧率
    'BATCH_SIZE': 32,                # 图像处理批量大小  
    'MAX_VIOLATION_IMAGES': 5,      # 最大违规图片展示数量
    'ASR_BATCH_SIZE': 300,          # ASR语音识别批量大小，初始300
    'HOTWORD': '魔搭',               # ASR语音识别热词
    'NUM_WORKERS': 4                # 数据加载器工作进程数
}

# 全局模型加载器
ml = None

"""
函数功能：统一的资源清理函数，清理中间过程产生的文件和目录
@param paths: 要清理的文件或目录路径列表    
"""
def cleanup_resources(*paths):
    for path in paths:
        if not path:
            continue
        try:
            if os.path.isfile(path):
                os.remove(path)
                print(f"已清理文件: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"已清理目录: {path}")
        except Exception as e:
            print(f"清理资源失败 {path}: {str(e)}")

"""
函数功能：提取违规图片的通用函数
@param output_file: 检测结果输出文件路径
@param image_folder: 存储图像的文件夹路径
@param maxclass: 最大违规类别
@param max_images: 最大提取违规图片数量，默认为配置中的最大违规图片数量
@return: 违规图片列表，每个元素包含图片名称和Base64编码的图片数据
"""
def extract_violation_images(output_file, image_folder, maxclass, max_images=None):
    if max_images is None:
        max_images = CONFIG['MAX_VIOLATION_IMAGES']
    
    # 如果maxclass是正常类别，直接返回空列表
    if maxclass.lower() in ['normal', '正常', 'safe']:
        return []
    
    # 定义违规类别
    violation_categories = {"bloody", "porn", "smoke", "violence"}
    target_category = maxclass.lower()
    
    # 如果maxclass不在违规类别中，返回空列表
    if target_category not in violation_categories:
        return []
    
    violation_images = []
    
    try:
        with open(output_file, "r") as f:
            # 寻找目标类别的部分
            found_target_category = False
            for line in f:
                if line.startswith("#"):
                    category = line.strip("# \n").lower()
                    if category == target_category:
                        found_target_category = True
                    elif found_target_category:
                        # 如果已经找到目标类别，遇到新的类别标题就停止
                        break
                elif found_target_category and line.strip():
                    # 在目标类别中提取图片
                    frame_name = line.split(" ")[0]
                    frame_path = os.path.join(image_folder, frame_name)
                    if os.path.exists(frame_path):
                        with open(frame_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode("utf-8")
                            violation_images.append({
                                "frame": frame_name, 
                                "data": img_data
                            })
                    
                    # 达到最大数量就停止
                    if len(violation_images) >= max_images:
                        break
    
    except Exception as e:
        print(f"提取违规图片失败: {str(e)}")
    
    return violation_images

"""
函数功能：生成PDF报告的通用函数
@param video_path: 视频文件路径
@param maxclass: 最大违规类别
@param violation_images: 违规图片列表
@return: 生成的PDF报告的Base64编码字符串
"""
def generate_pdf_report(video_path, maxclass=None, violation_images=None):
    try:
        from pdf import VideoReportGenerator
        
        # 清理视频路径
        clean_video_path = video_path.replace('"', '').strip()
        print(f"开始生成PDF报告，视频路径: {clean_video_path}")
        
        # 从视频路径提取video_name
        video_name = os.path.splitext(os.path.basename(clean_video_path))[0]
        
        print(f"使用参数: video_name={video_name}, maxclass={maxclass}, violation_images数量={len(violation_images) if violation_images else 0}")
        
        # 使用固定的uploads/data目录调用PDF生成器，传入violation_images参数
        generator = VideoReportGenerator(video_name=video_name, maxclass=maxclass, violation_images=violation_images)
        pdf_path = generator.generate_report()
        
        print(f"PDF报告生成完成: {pdf_path}")
        
        print(f"PDF文件路径: {pdf_path}")
        
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"找到PDF文件，大小: {file_size} 字节")
            
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = base64.b64encode(pdf_file.read()).decode('utf-8')
                if pdf_data:
                    print(f"✅ PDF Base64 数据生成成功（{len(pdf_data)} 字符）")
                    return pdf_data
                else:
                    print("❌ PDF 数据为空")
        else:
            print(f"❌ PDF文件不存在: {pdf_path}")
            # 列出可能的PDF文件位置
            possible_dirs = [
                os.path.join(os.path.dirname(clean_video_path), "data", "report"),
                os.path.join("uploads", "data", "report"),
                "data/report",
                "uploads/data/report"
            ]
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    print(f"目录 {dir_path} 中的文件: {files}")
                    

    except Exception as e:
        print(f"生成PDF报告失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
    
    return None

"""
函数功能：根据视频文件大小自动选择帧率
@param video_path: 视频文件路径
@return: 适合视频大小的帧率
"""
def get_adaptive_frame_rate(video_path):
    """根据视频文件大小自动选择帧率"""
    try:
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)  # 转换为MB

        if file_size_mb < 8:            # 小于8  MB
            return fps['small']         # 8 fps
        elif file_size_mb < 16:         # 8-16MB
            return fps['medium']        # 16 fps
        else:                           # 大于16MB
            return fps['large']         # 24 fps
            
    except Exception as e:
        print(f"获取视频大小失败: {e}")
        return CONFIG['FRAME_RATE']     # 回退到默认值4


"""
函数功能：视频处理的通用逻辑
@param video_path: 视频文件路径
@return: 处理结果字典，包含音频、图像文字处理结果和视频
"""
def process_video_common(video_path):
    try:
        # 初始化处理器
        processor = VideoProcessor(
            video_path=video_path,
            model_loader=ml,
            frame_rate=4,
            batch_size=CONFIG['BATCH_SIZE']
        )
        
        # 处理流程
        print("开始处理视频...")
        processor.process()
        print("视频处理完成")
        
        # 图像分析报告
        counts = parse_report(processor.output_file)
        total_frames = len([
            name for name in os.listdir(processor.image_folder) 
            if os.path.isfile(os.path.join(processor.image_folder, name))
        ])
        
        maxclass, probability, secondclass, secondprob = calculate_dominant_category(counts, total_frames)
        print(f"图像分析结果：最大可能违规类别是 {maxclass}，概率为 {probability:.2%}；"
              f"第二可能违规类别是 {secondclass}，概率为 {secondprob:.2%}")
        
        max_confidence, second_confidence = calculate_category_confidence(
            processor.output_file, maxclass, secondclass
        )
        print(f"根据置信度计算：{maxclass}类别平均置信度为 {max_confidence:.2%}；"
              f"{secondclass}类别平均置信度为 {second_confidence:.2%}")
        
        # 音频处理
        audio_cat, audio_violate_txt = None, None
        try:
            print("开始处理音频...")
            processor.extract_audio()
            if os.path.exists(processor.audio_file):
                processor.audio_distinguish()
                audio_cat, audio_violate_txt = processor.predict_audiotext()
                print("音频处理完成")
                with open(processor.violateAudio_file,"w") as f:
                    f.write(json.dumps(audio_violate_txt, ensure_ascii=False))
            else:
                print("未找到音频文件，跳过音频处理")
        except Exception as e:
            print(f"音频处理失败: {str(e)}")
        
        # 图像文字处理
        ocr_cat, ocr_violate_txt = None, None
        try:
            print("开始处理图像文字...")
            processor.extract_text_from_image()
            ocr_cat, ocr_violate_txt = processor.predict_ocrtext()
            print("图像文字处理完成")
            with open(processor.violateOcr_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(ocr_violate_txt, ensure_ascii=False))
        except Exception as e:
            print(f"图像文字处理失败: {str(e)}")
        
        # 提取违规图片
        violation_images = extract_violation_images(processor.output_file, processor.image_folder, maxclass)
        # 生成PDF报告，传入violation_images参数
        pdf_data = generate_pdf_report(video_path, maxclass, violation_images)
        
        # 整理结果
        result = {
            "audio_violation": json.loads(audio_cat)["violation_categories"],
            "text_violation": json.loads(ocr_cat)["violation_categories"],
            "audio_text": json.loads(audio_violate_txt)["violation_texts"] if audio_violate_txt else [],
            "ocr_text": json.loads(ocr_violate_txt)["violation_texts"] if ocr_violate_txt else [],
            "video_main_category": {"name": maxclass, "confidence": max_confidence},
            "video_second_category": {"name": secondclass, "confidence": second_confidence},
            "violation_images": violation_images,
            "pdf_data": pdf_data
        }

        result_for_save = {k: v for k, v in result.items() if k != "pdf_data"}
        # 保存到文件
        with open(processor.frontbackend_data_file, "w", encoding='utf-8') as f:
            json.dump(result_for_save, f, ensure_ascii=False, indent=4)
        return result
        
    except Exception as e:
        raise Exception(f"视频处理失败: {str(e)}")

"""
函数功能：处理视频URL的Flask路由
@param video_url: 视频链接
@return: 处理结果的JSON响应
"""
@app.route('/process_url', methods=['POST'])
def process_video_url():
    if not request.json or 'url' not in request.json:
        return jsonify({"error": "未提供视频URL"}), 400
    
    video_url = request.json['url']
    video_path = None
    
    try:
        # 下载视频
        print(f"开始下载视频: {video_url}")
        video_path = download_video(video_url)
        print(f"视频下载完成: {video_path}")
        
        # 处理视频
        result = process_video_common(video_path)
        
        print("处理结果已整理")
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 清理临时视频文件
        cleanup_resources(video_path)

"""
函数功能：处理上传的视频文件的Flask路由
@param video_file: 上传的视频文件
@return: 处理结果的JSON响应
"""
@app.route('/process', methods=['POST'])
def process_video():
    """处理上传的视频文件"""
    if 'video' not in request.files:
        return jsonify({"error": "未上传视频文件"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "未选择视频文件"}), 400
    
    video_path = None
    try:
        # 保存上传的视频文件
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        # 处理视频
        result = process_video_common(video_path)
        
        print("处理结果已整理")
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 清理临时视频文件
        cleanup_resources(video_path)

"""
函数功能：处理音频文件的Flask路由
@param audio_file: 上传的音频文件
@return: 处理结果的JSON响应
"""
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "未上传音频文件"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "未选择音频文件"}), 400
    
    # 创建临时目录结构
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_name = f"audio_{timestamp}"
    base_dir = os.path.join(app.config['UPLOAD_FOLDER'], audio_name)
    
    try:
        os.makedirs(os.path.join(base_dir, "data/audio"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/audio-txt"), exist_ok=True)
        
        # 保存上传的音频文件
        audio_path = os.path.join(base_dir, "data/audio", f"{audio_name}.wav")
        audio_file.save(audio_path)
        
        print(f"开始处理音频: {audio_path}")
        
        # 创建音频文本输出路径
        audiotext_file = os.path.join(base_dir, "data/audio-txt", f"{audio_name}_audio.txt")
        
        # 使用ASR模型处理音频
        start_time = time.time()
        result = ml.asr_model.generate(
            input=audio_path, 
            batch_size_s=CONFIG['ASR_BATCH_SIZE'], 
            hotword=CONFIG['HOTWORD']
        )
        print(f"音频处理时间: {time.time() - start_time:.2f}秒")
        
        # 提取文本
        audio_text = " ".join(segment['text'] for segment in result).strip()
        
        # 保存文本结果
        with open(audiotext_file, "w", encoding='utf-8') as f:
            f.write(audio_text)
        
        # 分析文本内容
        
        if audio_text.strip():
            try:
                audio_phrases = [phrase.strip() for phrase in re.split(r'[，。]', audio_text) if phrase.strip()]
                print(f"开始分析文本内容，共{len(audio_phrases)}个短语")
                
                if audio_phrases:
                    categories_json, texts_json = deepseek_text(audio_phrases)
                    print("文本分析完成")
                else:
                    print("警告：没有有效的文本短语可供分析")
                    
            except Exception as e:
                print(f"文本分析失败: {str(e)}")
                categories_json = json.dumps({"violation_categories": ["分析失败"]}, ensure_ascii=False)
        
        # 整理结果
        result = {
            "audio_violation": json.loads(categories_json)["violation_categories"],
            "audio_violation_texts":  json.loads(texts_json)["violation_texts"],
            "audio_text": audio_text,
        }
        
        print("音频处理结果已整理")
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500
    finally:
        # 清理临时文件和目录
        cleanup_resources(base_dir)

"""
函数功能：处理文本内容的Flask路由
@param text_content: 文本内容
@return: 处理结果的JSON响应
"""
@app.route('/process_text', methods=['POST'])
def process_text():
    """处理文本内容"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "未提供文本内容"}), 400
    
    text_content = request.json['text']
    
    try:
        print("开始处理文本内容")
        
        # 分割文本为短语
        text_phrases = [phrase.strip() for phrase in re.split(r'[，。]', text_content) if phrase.strip()]
        
        if not text_phrases:
            return jsonify({
                "violation_categories": ["正常"],
                "violation_texts": []
            })
        
        # 使用deepseek_text分析文本内容
        categories_json, texts_json = deepseek_text(text_phrases)
        
        # 整理结果
        result = {
            "violation_categories": json.loads(categories_json)["violation_categories"],
            "violation_texts": json.loads(texts_json)["violation_texts"]
        }
        
        print("文本处理结果已整理")
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    })


# 添加错误处理
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

"""
函数功能：设置日志系统，记录所有输出到日志文件
@return: 日志文件路径和文件句柄
"""
def setup_logging():
    """设置日志系统"""
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"nsfw_detect_{timestamp}.log")
    
    log_file_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
    
    class LoggerWriter:
        def __init__(self, file_handle):
            self.file_handle = file_handle
            
        def write(self, message):
            if not message or message.isspace():
                return
                
            if isinstance(message, bytes):
                try:
                    message = message.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        message = message.decode('gbk')
                    except UnicodeDecodeError:
                        message = message.decode('latin-1')
            
            # 移除ANSI颜色代码
            message = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', message)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                self.file_handle.write(f"{timestamp} - {message}")
                self.file_handle.flush()
            except UnicodeEncodeError:
                safe_message = message.encode('utf-8', errors='replace').decode('utf-8')
                self.file_handle.write(f"{timestamp} - [编码错误] {safe_message}")
                self.file_handle.flush()
            
        def flush(self):
            self.file_handle.flush()
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = LoggerWriter(log_file_handle)
    sys.stderr = LoggerWriter(log_file_handle)
    
    init_msg = f"日志系统已初始化，日志文件: {log_file}\n"
    log_file_handle.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {init_msg}")
    log_file_handle.flush()
    
    return log_file, log_file_handle, original_stdout, original_stderr

if __name__ == "__main__":
    log_file, log_handle, orig_stdout, orig_stderr = setup_logging()
    
    orig_stdout.write(f"所有输出将被记录到日志文件: {log_file}\n")
    orig_stdout.flush()
    
    try:
        # 模型预加载
        print("正在加载所有模型...")
        start_load = time.time()
        ml = ModelLoader()
        ml.load_all_models()
        print(f"模型加载完成，用时: {time.time() - start_load:.2f}秒\n")
        
        # 创建上传文件夹
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        app.config['UPLOAD_FOLDER'] = upload_folder
        
        # 启动Flask应用
        print("启动服务...")
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("服务已停止")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
        if log_handle:
            log_handle.close()
        
        print("日志系统已关闭")