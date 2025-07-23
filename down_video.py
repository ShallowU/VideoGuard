import os
import re
import time
import random
import string
import argparse
import requests
from urllib.parse import urlparse
import yt_dlp

"""
函数功能：清理文件名，去除非法字符和多余空格
@param name: 文件名或视频标题
@return: 清理后的文件名，去除非法字符和多余空格，限制长度为100字符
"""
def sanitize_filename(name):
    cleaned = re.sub(r'[\\/*?:"<>|]', '', name)
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned[:100] if cleaned else 'video'

"""
函数功能：生成备用文件名，包含时间戳和随机字符串
@param url: 原始视频URL
@return: 生成的备用文件名，格式为 video_时间戳_路径片段_随机字符串.mp4
"""
def generate_fallback_name(url):
    parsed = urlparse(url)
    path_segment = parsed.path.split('/')[-1] if parsed.path else 'video'
    path_segment = re.sub(r'\W+', '_', path_segment)[:30]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"video_{timestamp}_{path_segment}_{rand_str}.mp4"

"""
函数功能：解析抖音视频链接，获取视频下载地址
@param url: 抖音视频链接
@return: 视频下载地址
"""
def parse_douyin_via_api(url):
    """主接口解析抖音"""
    api_url = "https://api.xinyew.cn/api/douyinjx"
    try:
        response = requests.get(api_url, params={"url": url}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 200:
            return data["data"]["video_url"]
        raise Exception(f"主接口失败: {data.get('msg', '未知错误')}")
    except Exception as e:
        print(f"⚠️ 抖音主接口失败: {str(e)}，尝试备用接口")
        return parse_via_yyy001_api(url)

"""
函数功能：通过备用接口解析视频链接，支持多个平台
@param url: 视频链接
@return: 视频下载地址
"""
def parse_via_yyy001_api(url):
    """聚合接口解析，支持多个平台"""
    api_url = "https://api.yyy001.com/api/videoparse2"
    try:
        response = requests.get(api_url, params={"url": url}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 200 and data.get("data", {}).get("video"):
            return data["data"]["video"]
        raise Exception(f"备用接口失败: {data.get('error') or '无视频地址'}")
    except Exception as e:
        raise Exception(f"备用接口解析失败: {str(e)}")

""""
函数功能：直接下载视频文件到指定目录
@param video_url: 视频下载地址
@param original_url: 原始视频链接，用于生成备用文件名
@param output_dir: 输出目录
@param referer: 请求头中的Referer字段
@return: 下载完成的视频文件路径
"""
def download_direct(video_url, original_url, output_dir, referer):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80 Mobile Safari/537.36',
        'Referer': referer
    }

    fallback_name = generate_fallback_name(original_url)
    output_path = os.path.join(output_dir, fallback_name)

    try:
        with requests.get(video_url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return output_path
    except Exception as e:
        raise Exception(f"视频下载失败: {str(e)}")

"""
函数功能：使用yt-dlp下载视频
@param url: 视频链接
@param output_dir: 输出目录
@return: 下载完成的视频文件路径
"""
def download_with_ytdlp(url, output_dir):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bilibili.com/'
    }

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,
        'quiet': False,
        'headers': headers,
        'merge_output_format': 'mp4',
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_path = ydl.prepare_filename(info)
            safe_title = sanitize_filename(info.get('title', 'video'))
            video_id = info.get('id') or os.path.splitext(os.path.basename(original_path))[0]
            new_name = f"{safe_title}_{video_id}.mp4"
            new_path = os.path.join(output_dir, new_name)
            if not os.path.exists(new_path):
                os.rename(original_path, new_path)
                print(f"文件已重命名为: {new_name}")
            return new_path
    except Exception as e:
        raise Exception(f"B站视频下载失败: {str(e)}")

"""
函数功能：下载视频，支持多平台
@param url: 视频链接
@param output_dir: 输出目录
@return: 下载完成的视频文件路径"""
def download_video(url, output_dir='./uploads'):
    os.makedirs(output_dir, exist_ok=True)

    if 'douyin' in url:
        video_url = parse_douyin_via_api(url)
        return download_direct(video_url, url, output_dir, referer='https://www.douyin.com/')

    elif any(domain in url for domain in ['kuaishou', 'gifshow', 'xiaohongshu', 'weibo', 'ixigua']):
        video_url = parse_via_yyy001_api(url)
        return download_direct(video_url, url, output_dir, referer=url)

    elif 'bilibili' in url or 'b23.tv' in url:
        return download_with_ytdlp(url, output_dir)

    else:
        print("⚠️ 未识别的平台，尝试使用通用备用接口")
        video_url = parse_via_yyy001_api(url)
        return download_direct(video_url, url, output_dir, referer=url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="视频下载工具 - 支持抖音、快手、小红书、微博、西瓜视频、B站")
    parser.add_argument('-u', '--url', type=str, required=True, help="视频URL")
    parser.add_argument('-o', '--output', type=str, default='./uploads', help="输出目录")
    args = parser.parse_args()

    print(f"🎬 正在下载: {args.url}")
    try:
        path = download_video(args.url, args.output)
        print(f"✅ 下载成功 → {os.path.abspath(path)}")
        print(f"📦 文件大小: {os.path.getsize(path) // 1024} KB")
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
