#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

class BilibiliReporter:
    """B站视频举报工具"""
    
    def __init__(self, config_file=".env"):
        # 加载环境变量
        load_dotenv(config_file)
        
        # 基础配置
        self.report_url = "https://api.bilibili.com/x/web-interface/archive/report"
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        # 从环境变量加载cookie
        self.cookies = self._load_cookies()
        
        # 会话对象
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Referer": "https://www.bilibili.com",
            "Origin": "https://www.bilibili.com"
        })
        
        # 设置cookies
        if self.cookies:
            self._set_cookies()
        
        # 日志文件
        self.log_file = "report.txt"
    
    def _load_cookies(self):
        """从环境变量加载cookies"""
        cookie_str = os.getenv("BILIBILI_COOKIE")
        return self._parse_cookie_string(cookie_str)
    
    def _parse_cookie_string(self, cookie_str):
        """解析cookie字符串为字典"""
        cookies = {}
        for item in cookie_str.split("; "):
            if "=" in item:
                key, value = item.split("=", 1)
                cookies[key] = value
        return cookies
    
    def _set_cookies(self):
        """设置cookies到会话"""
        if isinstance(self.cookies, dict):
            self.session.cookies.update(self.cookies)
        else:
            print("Cookie格式错误，请检查")
    
    def get_video_id(self, bv):
        """通过BV号获取AV号"""
        try:
            url = f'https://www.bilibili.com/video/{bv}'
            headers = {
                "User-Agent": self.user_agent,
                "Referer": "https://www.bilibili.com"
            }
            
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            content = response.text
            
            aid_regx = r'"aid":(\d+),"bvid":"' + re.escape(bv) + r'"'
            matches = re.findall(aid_regx, content)
            
            if matches:
                return int(matches[0])
            else:
                # 尝试另一种正则表达式
                aid_regx2 = r'"aid":(\d+)'
                matches2 = re.findall(aid_regx2, content)
                if matches2:
                    return int(matches2[0])
                else:
                    raise Exception("无法从页面中提取AV号")
                    
        except Exception as e:
            raise Exception(f"获取视频ID失败: {str(e)}")
    
    def extract_bv_from_url(self, url):
        """从B站URL中提取BV号"""
        try:
            # 匹配BV号的正则表达式
            bv_pattern = r'BV[1-9a-km-zA-HJ-NP-Z]{10}'
            matches = re.findall(bv_pattern, url)
            
            if matches:
                return matches[0]
            else:
                raise Exception("无法从URL中提取BV号")
                
        except Exception as e:
            raise Exception(f"提取BV号失败: {str(e)}")

    def _log_detailed(self, info_type, data):
        """记录详细的响应信息"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if isinstance(data, dict):
                data_str = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                data_str = str(data)
            
            log_entry = f"\n{timestamp} - {info_type}:\n{data_str}\n" + "="*50 + "\n"
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"写入详细日志失败: {str(e)}")

    def report_video(self, aid, reason_type=2, detail="擦边色情视频", max_retries=2):
        """
        举报视频
        
        Args:
            aid: 视频AV号
            reason_type: 举报原因类型ID (2表示色情低俗)
            detail: 详细说明
            max_retries: 最大重试次数
            
        Returns:
            是否举报成功
        """
        for attempt in range(max_retries + 1):
            try:
                csrf = self.cookies.get("bili_jct")
                if not csrf:
                    raise Exception("未找到CSRF令牌，无法举报")
                
                data = {
                    "aid": aid,
                    "reason": reason_type,
                    "detail": detail,
                    "csrf": csrf
                }
                
                response = self.session.post(self.report_url, data=data)
                # 记录响应状态码
                self._log_detailed("HTTP状态码", response.status_code)
                
                # 记录响应头信息
                self._log_detailed("响应头", dict(response.headers))
                
                # 记录原始响应文本
                self._log_detailed("原始响应", response.text)
                result = response.json()
                
                # 记录完整的JSON响应
                self._log_detailed("JSON响应", result)
                
                if result["code"] == 0:
                    self._log_report(aid, "成功", f"举报成功")
                    print(f"成功举报视频 av{aid}")
                    return True
                else:
                    error_msg = result.get("message", "未知错误")
                    if attempt < max_retries:
                        print(f"举报失败，正在重试 ({attempt + 1}/{max_retries}): {error_msg}")
                        time.sleep(2)  # 等待2秒后重试
                        continue
                    else:
                        self._log_report(aid, "失败", f"举报失败: {error_msg}")
                        print(f"举报失败: {error_msg}")
                        return False
                        
            except Exception as e:
                if attempt < max_retries:
                    print(f"举报出错，正在重试 ({attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    self._log_report(aid, "失败", f"举报出错: {str(e)}")
                    print(f"举报视频时出错: {str(e)}")
                    return False
        
        return False
    
    def _log_report(self, aid, status, message):
        """记录举报日志"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - AV{aid} - {status} - {message}\n"
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"写入日志失败: {str(e)}")
    
    def report_bilibili_video(self, video_url):
        """
        举报B站视频的主函数
        
        Args:
            video_url: B站视频URL
            
        Returns:
            举报结果字典
        """
        try:
            # 提取BV号
            bv = self.extract_bv_from_url(video_url)
            print(f"提取到BV号: {bv}")
            
            # 获取AV号
            aid = self.get_video_id(bv)
            print(f"获取到AV号: {aid}")
            
            # 举报视频
            success = self.report_video(aid)
            
            result = {
                "success": success,
                "bv": bv,
                "aid": aid,
                "message": "举报成功" if success else "举报失败"
            }
            
            self._log_report(aid, "成功" if success else "失败", 
                           f"BV{bv} - {'举报成功' if success else '举报失败'}")
            
            return result
            
        except Exception as e:
            error_msg = f"举报过程出错: {str(e)}"
            print(error_msg)
            self._log_report("未知", "失败", error_msg)
            
            return {
                "success": False,
                "bv": None,
                "aid": None,
                "message": error_msg
            }


def main():
    """测试函数"""
    reporter = BilibiliReporter()
    
    # 测试URL
    test_url = "https://www.bilibili.com/video/BV16Z8jzkERx"
    
    print(f"开始举报视频: {test_url}")
    result = reporter.report_bilibili_video(test_url)
    
    print(f"举报结果: {result}")


if __name__ == "__main__":
    main()