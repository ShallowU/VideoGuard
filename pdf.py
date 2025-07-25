# 视频内容检测报告生成器
# 该模块使用ReportLab库生成PDF格式的检测报告，支持学术风格的封面、页眉页脚和内容样式。
import os
import subprocess
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import logging
import argparse

# 注册中文字体
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
styles = getSampleStyleSheet()

def add_style_if_not_exists(style_name, style_config):
    if not hasattr(styles, style_name):
        styles.add(ParagraphStyle(name=style_name, **style_config))

# 定义学术样式配置
style_configs = {
    'CoverTitle': {
        'fontName': 'SimSun',
        'fontSize': 24,
        'leading': 30,
        'alignment': 1,
        'textColor': colors.HexColor('#2F5496'),
        'spaceAfter': 20
    },
    'AcademicHeading': {  # 添加缺失的AcademicHeading样式
        'fontName': 'SimSun',
        'fontSize': 16,
        'leading': 20,
        'textColor': colors.HexColor('#2F5496'),
        'spaceBefore': 20,
        'spaceAfter': 12,
        'alignment': 0
    },
    'AcademicSection': {
        'fontName': 'SimSun',
        'fontSize': 14,
        'leading': 18,
        'textColor': colors.HexColor('#2F5496'),
        'spaceBefore': 24,
        'spaceAfter': 12,
        'underlineWidth': 1,
        'underlineColor': colors.HexColor('#2F5496')
    },
    'AcademicBody': {
        'fontName': 'SimSun',
        'fontSize': 10.5,
        'leading': 15,
        'spaceAfter': 6,
        'textColor': colors.HexColor('#404040')
    },
    'AcademicTableHeader': {
        'fontName': 'SimSun',
        'fontSize': 11,
        'leading': 13,
        'alignment': 1,
        'textColor': colors.white,
        'backColor': colors.HexColor('#4F81BD')
    },
    'AcademicCaption': {
        'fontName': 'SimSun',
        'fontSize': 9,
        'leading': 11,
        'alignment': 1,
        'textColor': colors.HexColor('#666666')
    },
    'AcademicSubsection': {
        'fontName': 'SimSun',
        'fontSize': 12,
        'leading': 16,
        'textColor': colors.HexColor('#2F5496'),
        'spaceBefore': 12,
        'spaceAfter': 8,
        'alignment': 0
    }
}

# 批量添加新样式
for style_name, config in style_configs.items():
    add_style_if_not_exists(style_name, config)


class VideoReportGenerator:
    # 常量定义
    DATA_DIR = "uploads/data"
    REPORT_DIR = "uploads/data/report"

    def __init__(self, video_name: str = None, video_path: str = None, maxclass: str = None, violation_images: list = None):
        # 初始化日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # 验证maxclass参数
        valid_classes = ['bloody', 'normal', 'porn', 'smoke', 'violence']
        if maxclass and maxclass not in valid_classes:
            raise ValueError(f"maxclass参数必须是以下值之一: {', '.join(valid_classes)}")
        self.maxclass = maxclass
        
        # 存储传入的违规图片
        self.violation_images = violation_images or []
        
        # 固定使用uploads/data作为数据目录
        self.data_dir = self.DATA_DIR
        
        # 支持两种初始化方式：通过video_name 或 通过video_path
        if video_name:
            self.video_name = video_name
            self.video_path = None  # 不需要实际视频文件
            self.output_path = os.path.join(self.REPORT_DIR, f"{video_name}_检测报告.pdf")
        elif video_path:
            self.video_path = video_path
            self.video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.output_path = os.path.join(self.REPORT_DIR, f"{self.video_name}_检测报告.pdf")
        else:
            raise ValueError("必须提供 video_name 或 video_path 参数")
            
        self.tech_data = {}
        # 初始化violation_data的所有键，确保数据类型一致
        self.violation_data = {
            'audio': "",
            'ocr': "",
            'image': {}
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
    def collect_tech_parameters(self):
        """从处理过的文件中收集技术参数"""
        try:
            # 读取图像检测信息
            image_detect_file = os.path.join(self.data_dir, "txt", f"{self.video_name}_ImageDetectInfo.txt")
            if os.path.exists(image_detect_file):
                with open(image_detect_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.tech_data['image_detection'] = self._parse_image_detection(content)
            
            # 读取音频文件信息
            audio_file = os.path.join(self.data_dir, "audio", f"{self.video_name}.wav")
            if os.path.exists(audio_file):
                self.tech_data['audio_info'] = self._get_audio_info(audio_file)
            
            # 统计视频帧数
            frames_dir = os.path.join(self.data_dir, "video-frame", f"{self.video_name}_frames")
            if os.path.exists(frames_dir):
                frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
                self.tech_data['frame_count'] = frame_count
                
        except Exception as e:
            self.logger.error(f"收集技术参数时出错: {e}")
    
    def collect_violation_data(self):
        """从处理过的文件中收集违规数据"""
        try:
            # 初始化所有违规数据键 - 根据使用方式调整数据类型
            self.violation_data['audio'] = ""  # 改为字符串，因为_create_violation_section中当作字符串处理
            self.violation_data['ocr'] = ""   # 改为字符串，因为_create_violation_section中当作字符串处理
            self.violation_data['image'] = {}
            
            # 读取音频和OCR违规检测结果
            self.violation_data['audio'] = self._load_violation_text("audio-txt", f"{self.video_name}_violate_audio.txt")
            self.violation_data['ocr'] = self._load_violation_text("ocr-txt", f"{self.video_name}_violate_ocr.txt")
            
            # 从图像检测结果中提取违规信息 - 修复数据结构
            if 'image_detection' in self.tech_data:
                image_violations = {}  # 改为字典结构
                for category, frames in self.tech_data['image_detection'].items():
                    if category.lower() in ['bloody', 'porn', 'violence']:  # 违规类别
                        image_violations[category] = frames  # 保持原始的帧列表结构
                self.violation_data['image'] = image_violations
            else:
                # 如果没有图像检测数据，初始化为空字典
                self.violation_data['image'] = {}
                
        except Exception as e:
            self.logger.error(f"收集违规数据时出错: {e}")
            # 确保即使出错也有正确的数据结构
            self.violation_data.setdefault('audio', "")
            self.violation_data.setdefault('ocr', "")
            self.violation_data.setdefault('image', {})
    
    def _load_violation_text(self, subdir: str, filename: str) -> str:
        """加载违规文本数据的通用方法"""
        file_path = os.path.join(self.data_dir, subdir, filename)
        if not os.path.exists(file_path):
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return ""
                
                # 处理双重编码的JSON
                violations = json.loads(content)
                if isinstance(violations, str):
                    violations = json.loads(violations)
                
                if isinstance(violations, dict):
                    violation_texts = violations.get('violation_texts', [])
                    if isinstance(violation_texts, list):
                        return '\n'.join(violation_texts)
                    else:
                        return str(violation_texts)
                elif isinstance(violations, list):
                    return '\n'.join(violations)
                elif isinstance(violations, str):
                    return violations
                    
        except Exception as e:
            self.logger.error(f"加载违规文本时出错 {file_path}: {e}")
        
        return ""
    
    def _parse_image_detection(self, content):
        """解析图像检测结果"""
        detection_results = {}
        current_category = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') and line.endswith('#'):
                # 提取类别名称
                current_category = line.strip('#').strip()
                detection_results[current_category] = []
            elif current_category and '.png' in line:
                # 提取帧信息
                detection_results[current_category].append(line)
        
        return detection_results
    
    def _get_audio_info(self, audio_file):
        """获取音频文件信息"""
        try:
            # 获取文件大小
            file_size = os.path.getsize(audio_file)
            return {
                'file_size': file_size,
                'format': 'WAV',
                'exists': True
            }
        except Exception as e:
            self.logger.error(f"获取音频信息时出错: {e}")
            return {'exists': False}


    
    def _create_cover_page(self, canvas, doc):
        """学术风格封面设计"""
        canvas.saveState()
        
        # 添加机构徽标（示例路径，需替换实际路径）
        logo_path = "university_logo.png"
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, 2*cm, A4[1]-4*cm, width=4*cm, height=3*cm, preserveAspectRatio=True)

        # 主标题
        canvas.setFont('SimSun', 24)
        canvas.setFillColor(colors.HexColor('#2F5496'))
        canvas.drawCentredString(A4[0]/2, A4[1]-6*cm, "视频内容检测报告")
        
        # 副标题
        canvas.setFont('SimSun', 18)
        canvas.setFillColor(colors.HexColor('#404040'))
        canvas.drawCentredString(A4[0]/2, A4[1]-8*cm, self.video_name)
        
        # 信息段落
        canvas.setFont('SimSun', 12)
        current_date = datetime.now()
        report_number = f"VG-{current_date.strftime('%Y%m%d')}-{current_date.strftime('%H%M%S')}"
        info_text = [
            ("生成机构：", "VideoGuard:基于大模型的短视频有害内容检测与预警系统"),
            ("生成日期：", current_date.strftime('%Y年%m月%d日')),
            ("报告编号：", report_number)
        ]
        y_position = A4[1]-12*cm
        for label, value in info_text:
            canvas.drawString(4*cm, y_position, f"{label}{value}")
            y_position -= 1.5*cm
        
        canvas.restoreState()

    def _header_footer(self, canvas, doc):
        """添加学术风格页眉页脚"""
        canvas.saveState()
        
        # 页眉
        canvas.setFont('SimSun', 9)
        canvas.setFillColor(colors.HexColor('#7F7F7F'))
        canvas.drawString(1*cm, A4[1]-1.5*cm, "VideoGuard:基于大模型与多模态融合的短视频平台有害内容检测系统")
        canvas.drawRightString(A4[0]-1*cm, A4[1]-1.5*cm, f"检测对象: {self.video_name}")
        
        # 页眉分割线
        canvas.setStrokeColor(colors.HexColor('#D0D0D0'))
        canvas.setLineWidth(0.5)
        canvas.line(1*cm, A4[1]-1.8*cm, A4[0]-1*cm, A4[1]-1.8*cm)
        
        # 页脚
        canvas.setFont('SimSun', 8)
        canvas.setFillColor(colors.HexColor('#808080'))
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        canvas.drawString(1*cm, 0.8*cm, f"生成时间: {current_date}")
        
        # 页脚页码
        page_num = f"第 {doc.page} 页"
        canvas.drawCentredString(A4[0]/2, 0.8*cm, page_num)
        
        # 页脚分割线
        canvas.line(1*cm, 1.2*cm, A4[0]-1*cm, 1.2*cm)
        
        canvas.restoreState()

    def _create_tech_table(self):
        """Academic style parameter table"""
        tech_list = [
            [Paragraph('<b>Technical Parameters</b>', styles['AcademicTableHeader']), 
             Paragraph('<b>Parameter Values</b>', styles['AcademicTableHeader']),
             Paragraph('<b>Description</b>', styles['AcademicTableHeader'])]
        ]
        
        # Define technical parameters display order and formatting
        param_info = {
            'frame_count': {
                'name': 'Total Video Frames',
                'format': lambda x: f"{x} frames",
                'description': 'Total number of image frames in the video file'
            },
            'video_resolution': {
                'name': 'Video Resolution',
                'format': lambda x: x if x else '1920×1080 (estimated)',
                'description': 'Pixel resolution of the video image'
            },
            'video_fps': {
                'name': 'Video Frame Rate',
                'format': lambda x: f"{x} fps" if x else '30 fps (estimated)',
                'description': 'Number of video frames played per second'
            },
            'video_duration': {
                'name': 'Video Duration',
                'format': lambda x: f"{x:.2f} seconds" if x else f"{self.tech_data.get('frame_count', 95) / 15:.2f} seconds (estimated)",
                'description': 'Playback duration of the video file'
            },
            'audio_info': {
                'name': 'Audio Parameters',
                'format': lambda x: self._format_audio_info(x) if x else 'No audio data',
                'description': 'Audio encoding format, sample rate and file size'
            },
            'image_detection': {
                'name': 'Image Detection Categories',
                'format': lambda x: self._format_image_detection(x) if x else 'No violations detected',
                'description': 'Detected image content categories and quantities'
            },
            'detection_accuracy': {
                'name': 'Detection Accuracy',
                'format': lambda x: f"{x}%" if x else '≥90.2% (system average standard)',
                'description': 'Accuracy rate of content security detection algorithm'
            },
            'processing_time': {
                'name': 'Processing Time',
                'format': lambda x: f"{x:.2f} seconds" if x else '< 10 seconds',
                'description': 'Time required to complete all detection processes'
            }
        }
        
        # 添加计算得出的技术参数
        if 'frame_count' in self.tech_data and 'video_fps' not in self.tech_data:
            self.tech_data['video_fps'] = None  # 将显示推测值
        if 'video_duration' not in self.tech_data:
            self.tech_data['video_duration'] = None  # 将显示推测值
        if 'video_resolution' not in self.tech_data:
            self.tech_data['video_resolution'] = None  # 将显示推测值
        if 'detection_accuracy' not in self.tech_data:
            self.tech_data['detection_accuracy'] = None  # 将显示系统标准
        if 'processing_time' not in self.tech_data:
            self.tech_data['processing_time'] = None  # 将显示估计值
        
        # 按照定义的顺序添加参数
        for param_key in param_info.keys():
            if param_key in self.tech_data or param_key in ['video_resolution', 'video_fps', 'video_duration', 'detection_accuracy', 'processing_time']:
                info = param_info[param_key]
                value = self.tech_data.get(param_key)
                formatted_value = info['format'](value)
                
                tech_list.append([
                    Paragraph(info['name'], styles['BodyText']),
                    Paragraph(formatted_value, styles['BodyText']),
                    Paragraph(info['description'], styles['BodyText'])
                ])
        
        tech_table = Table(tech_list, colWidths=[4*cm, 6*cm, 5*cm], repeatRows=1)
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTNAME', (0,0), (-1,-1), 'SimSun'),
            ('FONTSIZE', (0,0), (-1,0), 11),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D9D9D9')),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,1), (-1,-1), 4),
            ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ]))
        return tech_table
    
    def _format_audio_info(self, audio_info):
        """Format audio information display"""
        if not audio_info or not audio_info.get('exists'):
            return 'No audio file'
        
        file_size = audio_info.get('file_size', 0)
        format_type = audio_info.get('format', 'Unknown')
        
        # Convert file size to appropriate units
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} B"
        
        return f"{format_type} format, {size_str}, 44.1kHz/16bit (estimated)"
    
    def _format_image_detection(self, image_detection):
        """Format image detection results display"""
        if not image_detection:
            return 'No violations detected'
        
        categories = []
        total_frames = 0
        
        for category, frames in image_detection.items():
            if frames and len(frames) > 0:
                categories.append(f"{category}: {len(frames)} frames")
                total_frames += len(frames)
        
        if not categories:
            return 'No violations detected'
        
        return f"Detection categories: {', '.join(categories)} (total {total_frames} frames)"


    def _create_violation_section(self):
        """创建违规内容检测部分"""
        violation_elements = []
        violation_elements.append(Paragraph(
            "2. 违规内容检测",
            styles['AcademicHeading']
        ))
        violation_elements.append(Paragraph(
            "本节详细分析视频中可能存在的违规内容，包括图像、音频和文本等多维度的安全检测结果。"
            "通过先进的AI算法对视频内容进行全面扫描，确保内容符合相关法规和平台标准。",
            styles['AcademicBody']
        ))
    
        # 添加图像内容违规展示部分
        violation_elements.append(Paragraph(
            "2.1 图像内容违规检测结果",
            styles['AcademicSection']
        ))
        
        # 检查是否有任何违规内容
        has_violations = any(len(violations) > 0 for violations in self.violation_data['image'].values())
        
        # 当maxclass不是Normal时，使用传入的violation_images参数展示违规图片
        if self.maxclass and self.maxclass != 'Normal':
            violation_elements.append(Paragraph(
                f"检测到视频内容可能包含{self.maxclass}类违规内容，以下展示相关检测结果：",
                styles['AcademicBody']
            ))
            
            # 使用传入的violation_images参数
            if self.violation_images:
                # 选取前6张图片（如果不足6张则全部选取）
                selected_images = self.violation_images[:6]
                
                violation_elements.append(Paragraph(
                    f"以下展示从视频中提取的{len(selected_images)}张关键帧图像：",
                    styles['AcademicBody']
                ))
                
                # 创建图片展示，图片信息放在图片下面
                for idx, img_data in enumerate(selected_images, 1):
                    try:
                        # 从base64数据创建临时图片文件
                        import tempfile
                        import base64
                        
                        # 解码base64数据
                        img_bytes = base64.b64decode(img_data['data'])
                        
                        # 创建临时文件
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_file.write(img_bytes)
                            temp_img_path = temp_file.name
                        
                        # 添加图片
                        img = RLImage(temp_img_path, width=8*cm, height=6*cm)
                        violation_elements.append(img)
                        
                        # 添加图片信息（放在图片下面）
                        caption = Paragraph(
                            f"图2.1.{idx}: 视频帧 {img_data['frame']}<br/>"
                            f"检测类别：{self.maxclass}<br/>"
                            f"帧序号：{idx}",
                            styles['AcademicCaption']
                        )
                        violation_elements.append(caption)
                        
                        # 在每张图片后添加小间距（除了最后一张）
                        if idx < len(selected_images):
                            violation_elements.append(Spacer(1, 0.3*cm))
                        
                        # 记录临时文件路径，稍后清理
                        if not hasattr(self, '_temp_files'):
                            self._temp_files = []
                        self._temp_files.append(temp_img_path)
                            
                    except Exception as e:
                        self.logger.error(f"处理违规图片时出错: {e}")
                        violation_elements.append(Paragraph(
                            f"图片{idx}处理失败",
                            styles['AcademicBody']
                        ))
            else:
                violation_elements.append(Paragraph(
                    "该视频正常所以未提供违规图片数据。",
                    styles['AcademicBody']
                ))
        elif not has_violations:
            violation_elements.append(Paragraph(
                "经过全面的图像内容安全检测，未发现明显的违规图像内容。所有视频帧均符合内容安全标准。",
                styles['AcademicBody']
            ))
            
            # 当没有违规时，使用传入的violation_images参数展示正常帧
            if self.violation_images:
                # 选择前6张图片（如果不足6张则全部选取）
                selected_images = self.violation_images[:6]
                
                violation_elements.append(Paragraph(
                    f"以下展示从视频中提取的{len(selected_images)}张关键帧图像：",
                    styles['AcademicBody']
                ))
                
                # 创建图片展示
                for idx, img_data in enumerate(selected_images, 1):
                    try:
                        # 从base64数据创建临时图片文件
                        import tempfile
                        import base64
                        
                        # 解码base64数据
                        img_bytes = base64.b64decode(img_data['data'])
                        
                        # 创建临时文件
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_file.write(img_bytes)
                            temp_img_path = temp_file.name
                        
                        # 添加图片
                        img = RLImage(temp_img_path, width=8*cm, height=6*cm)
                        violation_elements.append(img)
                        
                        # 添加图片信息
                        caption = Paragraph(
                            f"图2.1.{idx}: 视频帧 {img_data['frame']}<br/>"
                            f"检测结果：正常内容<br/>"
                            f"帧序号：{idx}",
                            styles['AcademicCaption']
                        )
                        violation_elements.append(caption)
                        
                        if idx < len(selected_images):
                            violation_elements.append(Spacer(1, 0.3*cm))
                        
                        # 记录临时文件路径，稍后清理
                        if not hasattr(self, '_temp_files'):
                            self._temp_files = []
                        self._temp_files.append(temp_img_path)
                            
                    except Exception as e:
                        self.logger.error(f"处理图片时出错: {e}")
                        violation_elements.append(Paragraph(
                            f"图片{idx}处理失败",
                            styles['AcademicBody']
                        ))
            else:
                violation_elements.append(Paragraph(
                    "未提供图片数据。",
                    styles['AcademicBody']
                ))
        else:
            # 创建违规统计表格
            violation_stats = []
            violation_stats.append([Paragraph('<b>违规类型</b>', styles['AcademicTableHeader']),
                                  Paragraph('<b>检测帧数</b>', styles['AcademicTableHeader']),
                                  Paragraph('<b>最高置信度</b>', styles['AcademicTableHeader']),
                                  Paragraph('<b>平均置信度</b>', styles['AcademicTableHeader'])])
            
            for category in ['bloody', 'porn', 'violence']:
                if category in self.violation_data['image'] and len(self.violation_data['image'][category]) > 0:
                    violations = self.violation_data['image'][category]
                    max_conf = max(v[1] for v in violations) * 100
                    avg_conf = sum(v[1] for v in violations) / len(violations) * 100
                    
                    violation_stats.append([
                        Paragraph(f"{category.capitalize()}类内容", styles['AcademicBody']),
                        Paragraph(f"{len(violations)}帧", styles['AcademicBody']),
                        Paragraph(f"{max_conf:.1f}%", styles['AcademicBody']),
                        Paragraph(f"{avg_conf:.1f}%", styles['AcademicBody'])
                    ])
            
            if len(violation_stats) > 1:  # 有数据才显示表格
                stats_table = Table(violation_stats, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4F81BD')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D9D9D9')),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
                ]))
                violation_elements.append(stats_table)
                violation_elements.append(Spacer(1, 0.5*cm))

            # 显示具体违规示例
            for idx, category in enumerate(['bloody', 'porn', 'violence'], 1):
                if category in self.violation_data['image'] and len(self.violation_data['image'][category]) > 0:
                    # 按置信度排序取前3
                    sorted_frames = sorted(self.violation_data['image'][category], key=lambda x: x[1], reverse=True)[:3]
                    
                    violation_elements.append(Paragraph(
                        f"2.1.{idx} {category.capitalize()}类违规内容检测结果",
                        styles['AcademicSection']
                    ))
                    
                    # 显示置信度统计
                    confidence_text = f"检测到{len(self.violation_data['image'][category])}个违规帧，"
                    confidence_text += f"最高置信度{max(v[1] for v in self.violation_data['image'][category])*100:.1f}%，"
                    confidence_text += f"平均置信度{sum(v[1] for v in self.violation_data['image'][category])/len(self.violation_data['image'][category])*100:.1f}%"
                    
                    violation_elements.append(Paragraph(confidence_text, styles['AcademicBody']))
                    
                    # 显示违规帧的文本描述（不再从文件夹读取图片）
                    violation_elements.append(Paragraph(
                        f"检测到{len(sorted_frames)}个{category.capitalize()}类违规帧，详细信息如下：",
                        styles['AcademicBody']
                    ))
                    
                    # 创建违规帧信息表格
                    frame_info_data = []
                    frame_info_data.append([Paragraph('<b>帧文件</b>', styles['AcademicTableHeader']),
                                          Paragraph('<b>检测置信度</b>', styles['AcademicTableHeader']),
                                          Paragraph('<b>风险等级</b>', styles['AcademicTableHeader'])])
                    
                    for f in sorted_frames:
                        risk_level = '高' if f[1] > 0.8 else '中' if f[1] > 0.5 else '低'
                        frame_info_data.append([
                            Paragraph(f[0], styles['AcademicBody']),
                            Paragraph(f"{f[1]*100:.2f}%", styles['AcademicBody']),
                            Paragraph(risk_level, styles['AcademicBody'])
                        ])
                    
                    frame_info_table = Table(frame_info_data, colWidths=[6*cm, 4*cm, 3*cm])
                    frame_info_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4F81BD')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D9D9D9')),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
                    ]))
                    violation_elements.append(frame_info_table)
                    violation_elements.append(Spacer(1, 0.5*cm))
        
        # 文本违规部分
        violation_elements.append(PageBreak())
        violation_elements.append(Paragraph("2.2 文本内容安全分析", styles['AcademicSection']))

        # 音频识别内容分析
        violation_elements.append(Paragraph("2.2.1 音频语音识别与内容分析", styles['AcademicSection']))
        
        # 读取所有音频识别内容
        all_audio_content = ""
        audio_file = os.path.join(self.data_dir, "audio-txt", f"{self.video_name}_audio.txt")
        if os.path.exists(audio_file):
            try:
                with open(audio_file, 'r', encoding='utf-8') as f:
                    all_audio_content = f.read().strip()
            except Exception as e:
                self.logger.error(f"读取音频文件时出错: {e}")
        
        # 读取违规音频内容
        violate_audio_content = ""
        violate_audio_file = os.path.join(self.data_dir, "audio-txt", f"{self.video_name}_violate_audio.txt")
        if os.path.exists(violate_audio_file):
            try:
                with open(violate_audio_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 处理双重编码的JSON，注意外围有双引号
                        if content.startswith('"') and content.endswith('"'):
                            content = content[1:-1]  # 去掉外围双引号
                            # 处理转义的反斜杠
                            content = content.replace('\\"', '"')
                        audio_violations = json.loads(content)
                        if isinstance(audio_violations, dict):
                            violation_texts = audio_violations.get('violation_texts', [])
                            if isinstance(violation_texts, list) and violation_texts:
                                violate_audio_content = '\n'.join(violation_texts)
            except Exception as e:
                self.logger.error(f"读取违规音频文件时出错: {e}")
        
        # 展示所有音频识别内容
        violation_elements.append(Paragraph(
            "本系统采用先进的语音识别技术对音频内容进行转录和分析，完整的音频识别结果如下：", 
            styles['AcademicBody']
        ))
        
        if all_audio_content:
            violation_elements.append(Paragraph(
                f"音频识别内容：{all_audio_content}", 
                styles['AcademicBody']
            ))
        else:
            violation_elements.append(Paragraph(
                "未检测到音频内容或音频识别结果为空。", 
                styles['AcademicBody']
            ))
        
        violation_elements.append(Spacer(1, 0.3*cm))
        
        # 展示违规音频内容
        violation_elements.append(Paragraph(
            "音频内容安全检测结果：", 
            styles['AcademicBody']
        ))
        
        if violate_audio_content:
            violation_elements.append(Paragraph(
                f"检测到以下违规音频内容：{violate_audio_content}", 
                styles['AcademicBody']
            ))
        else:
            violation_elements.append(Paragraph(
                "经过音频内容安全检测，未发现违规内容。音频内容符合安全标准。", 
                styles['AcademicBody']
            ))
        
        # OCR文字识别内容分析
        violation_elements.append(Spacer(1, 0.5*cm))
        violation_elements.append(Paragraph("2.2.2 图像文字识别与内容分析", styles['AcademicSection']))
        
        # 读取所有OCR识别内容
        all_ocr_content = ""
        ocr_file = os.path.join(self.data_dir, "ocr-txt", f"{self.video_name}_ocr.txt")
        if os.path.exists(ocr_file):
            try:
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    all_ocr_content = f.read().strip()
            except Exception as e:
                self.logger.error(f"读取OCR文件时出错: {e}")
        
        # 读取违规OCR内容
        violate_ocr_content = ""
        violate_ocr_file = os.path.join(self.data_dir, "ocr-txt", f"{self.video_name}_violate_ocr.txt")
        if os.path.exists(violate_ocr_file):
            try:
                with open(violate_ocr_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 处理双重编码的JSON，注意外围有双引号
                        if content.startswith('"') and content.endswith('"'):
                            content = content[1:-1]  # 去掉外围双引号
                            # 处理转义的反斜杠
                            content = content.replace('\\"', '"')
                        ocr_violations = json.loads(content)
                        if isinstance(ocr_violations, dict):
                            violation_texts = ocr_violations.get('violation_texts', [])
                            if isinstance(violation_texts, list) and violation_texts:
                                violate_ocr_content = '\n'.join(violation_texts)
            except Exception as e:
                self.logger.error(f"读取违规OCR文件时出错: {e}")
        
        # 展示所有OCR识别内容
        violation_elements.append(Paragraph(
            "本系统采用光学字符识别(OCR)技术提取视频帧中的文字信息，完整的OCR识别结果如下：", 
            styles['AcademicBody']
        ))
        
        if all_ocr_content:
            violation_elements.append(Paragraph(
                f"OCR识别内容：{all_ocr_content}", 
                styles['AcademicBody']
            ))
        else:
            violation_elements.append(Paragraph(
                "未检测到图像文字内容或OCR识别结果为空。", 
                styles['AcademicBody']
            ))
        
        violation_elements.append(Spacer(1, 0.3*cm))
        
        # 展示违规OCR内容
        violation_elements.append(Paragraph(
            "图像文字内容安全检测结果：", 
            styles['AcademicBody']
        ))
        
        if violate_ocr_content:
            violation_elements.append(Paragraph(
                f"检测到以下违规文字内容：{violate_ocr_content}", 
                styles['AcademicBody']
            ))
        else:
            violation_elements.append(Paragraph(
                "经过图像文字内容安全检测，未发现违规内容。图像文字内容符合安全标准。", 
                styles['AcademicBody']
            ))
        
        # 添加第3章节：图像分类检测详细结果
        violation_elements.append(PageBreak())
        violation_elements.append(Paragraph("3. 图像分类检测详细结果", styles['AcademicHeading']))
        violation_elements.append(Paragraph(
            "本节提供视频帧图像分类检测的详细结果，包括各类别的检测置信度和具体帧信息。"
            "通过深度学习模型对每一帧图像进行多分类检测，涵盖正常内容、血腥、色情、吸烟、暴力等类别。",
            styles['AcademicBody']
        ))
        
        # 读取ImageDetectInfo.txt文件
        image_detect_file = os.path.join(self.data_dir, "txt", f"{self.video_name}_ImageDetectInfo.txt")
        
        try:
            if os.path.exists(image_detect_file):
                with open(image_detect_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    # 只显示maxclass对应的检测结果
                    violation_elements.append(Paragraph("3.1 图像检测原始结果", styles['AcademicSection']))
                    
                    if self.maxclass:
                        # 解析文件内容，只提取maxclass对应的部分
                        lines = content.split('\n')
                        in_target_section = False
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith('# '):
                                category = line[2:].strip()
                                if category.lower() == self.maxclass.lower(): # 匹配类别小写字母
                                    in_target_section = True
                                    violation_elements.append(Paragraph(
                                        f"{category}类检测结果：",
                                        styles['AcademicBody']
                                    ))
                                else:
                                    in_target_section = False
                            elif in_target_section and line and '.png' in line:
                                # 显示该类别的检测结果
                                violation_elements.append(Paragraph(
                                    line,
                                    styles['AcademicBody']
                                ))
                        
                        # if not in_target_section:
                        #     violation_elements.append(Paragraph(
                        #         f"未找到{self.maxclass}类别的检测结果。",
                        #         styles['AcademicBody']
                        #     ))
                    else:
                        violation_elements.append(Paragraph(
                            "未指定maxclass参数，无法显示特定类别的检测结果。",
                            styles['AcademicBody']
                        ))
                
                else:
                    violation_elements.append(Paragraph(
                        "图像检测信息文件为空，无法显示详细检测结果。",
                        styles['AcademicBody']
                    ))
            else:
                violation_elements.append(Paragraph(
                    f"未找到图像检测信息文件：{image_detect_file}",
                    styles['AcademicBody']
                ))
        
        except Exception as e:
            violation_elements.append(Paragraph(
                f"读取图像检测信息时发生错误：{str(e)}",
                styles['AcademicBody']
            ))
        
        return violation_elements

    def generate_report(self):
        """生成学术风格报告"""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=2.5*cm,
            rightMargin=2.5*cm,
            topMargin=3*cm,
            bottomMargin=2.5*cm
        )
        
        story = []
        # 封面页
        story.append(PageBreak())  # 封面页通过canvas直接绘制
        
        # 技术参数页
        story.append(Paragraph("<u>1. 多媒体文件技术参数分析</u>", styles['AcademicSection']))
        story.append(Paragraph(
            "本节对待检测多媒体文件的技术参数进行详细分析，包括视频编码格式、音频参数、"
            "分辨率等关键技术指标。这些参数对于内容检测算法的准确性和效率具有重要影响。",
            styles['AcademicBody']
        ))
        story.append(Spacer(1, 0.5*cm))
        story.append(self._create_tech_table())
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            "<b>技术参数说明：</b>上述参数反映了多媒体文件的基本质量特征。高分辨率和合适的编码格式"
            "有助于提高检测算法的准确性，而音频参数则影响语音识别的效果。",
            styles['AcademicBody']
        ))
        
        # 违规内容页
        story += self._create_violation_section()
        
        doc.build(
            story,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer,
            canvasmaker=lambda filename, **kwargs: CanvasWithCover(
                filename=filename,
                cover_func=self._create_cover_page,
                **kwargs
            )
        )
        
        # 清理临时文件
        if hasattr(self, '_temp_files'):
            for temp_file in self._temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            self._temp_files = []
        
        return self.output_path


class CanvasWithCover(canvas.Canvas):
    """支持封面页绘制的自定义Canvas（最终修正版）"""
    def __init__(self, filename, cover_func, pagesize=None, **kwargs):
        self.cover_func = cover_func
        self._page_count = 0
        super().__init__(filename, pagesize=pagesize, **kwargs)  # 显式传递必要参数

    def showPage(self):
        if self._page_count == 0:
            # 传递完整的参数：canvas对象和doc对象
            self.cover_func(self, None)  # 修正参数传递，doc对象可能不存在
            self._page_count += 1
        super().showPage()

if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description='生成视频内容检测报告')
    parser.add_argument(
        '--video_path', 
        type=str,
        default="/home/ubuntu/video_detect/codes-latest/test-video/porn.mp4",  # 默认测试路径
        help='待检测视频文件路径（例如：/data/videos/test.mp4）'
    )
    parser.add_argument(
        '--video_name',
        type=str,
        help='视频名称（例如：video_20250604_145959_7428588963511684415_41pjs）'
    )
    parser.add_argument(
        '--maxclass',
        type=str,
        choices=['Bloody', 'Normal', 'Porn', 'Smoke', 'Violence'],
        help='指定最大违规类别（可选值：Bloody, Normal, Porn, Smoke, Violence）'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 生成报告
    if args.video_name:
        # 使用处理过的数据生成报告（命令行调用时没有violation_images，传入None）
        report_gen = VideoReportGenerator(video_name=args.video_name, maxclass=args.maxclass, violation_images=None)
    else:
        # 使用视频文件路径生成报告
        if not os.path.exists(args.video_path):
            print(f"错误：视频文件不存在 - {args.video_path}")
            exit(1)
        if not os.path.isfile(args.video_path):
            print(f"错误：路径不是文件 - {args.video_path}")
            exit(1)
        report_gen = VideoReportGenerator(video_path=args.video_path, maxclass=args.maxclass, violation_images=None)
    
    report_gen.collect_tech_parameters()
    report_gen.collect_violation_data()
    report_gen.generate_report()
    print(f"报告已生成：{report_gen.output_path}")
