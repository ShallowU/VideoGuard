# 一些实用工具函数
# 该模块包含视频分类转换为JSON格式的函数，以及使用DeepSeek API检测文本违规的函数。
# 主要用于视频内容检测和文本违规检测。
import requests
import json
from openai import OpenAI
import re

def videoclass2json(maxclass, secondclass):
    '''
    将视频分类结果转换为JSON格式
    @param maxclass: 主分类
    @param secondclass: 次分类(如果为None则不添加)
    @return: JSON格式的字符串
    '''
    # 构建字典
    result_dict = {
        "maxclass": maxclass
    }
    # 如果 secondclass 不是 "None"，则添加到字典中
    if secondclass != "None":
        result_dict["secondclass"] = secondclass
    # 将字典转换为JSON格式
    result_json = json.dumps(result_dict, ensure_ascii=False, indent=4)
    return result_json

"""
函数功能：使用DeepSeek 检测文本违规
@param texts: 要检测的文本内容
@return: 返回违规类别和违规文本的JSON格式字符串
"""
def deepseek_text(texts):
    url = "https://aigptapi.com/v1/chat/completions"
    payload = json.dumps({
        "model": "gemini-2.5-flash-lite-preview-06-17",
        "messages": [
            {"role": "system", "content": "You are a text violation detection assistant. Return Chinese categories first, then the violating texts in a standard list format."},
            {"role": "user", "content": f"分类：色情、辱骂、赌博、政治、正常。输入检测文字后，如果全部正常则正常，第一行返回：正常，第二行返回空列表[]。如果违规返回上述违规类别（可能有多个违规类别均返回）以及对应的违规文字。示例检测：输入格式内容：['今天的新闻联播播送完了','感谢收看更多新闻资讯','请关注央视新闻客户端','你是傻逼','日你奶奶','你妈死了','草泥马']，返回：辱骂 色情，再另起一行返回['你是傻逼','日你奶奶','你妈死了','草泥马']  检测以下文本，返回格式：第一行是违规类别，第二行是标准的Python列表格式的违规文本。输入：{texts}"},
        ],
        "stream": False
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': 'sk-j7w0H4kx7Fqm7kJlPet4rCCGXGrO8Jhy7kGWKdz1kOjwRmaR',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        response_data = json.loads(response.text)
        
        # 检查响应结构
        if "choices" not in response_data or len(response_data["choices"]) == 0 or "message" not in response_data["choices"][0]:
            return json.dumps({"error": "无效的API响应格式"}), json.dumps({"error": "无效的API响应格式"})
        
        assistant_message = response_data["choices"][0]["message"].get("content", "")
        
    except requests.RequestException as e:
        return json.dumps({"error": f"API请求失败: {str(e)}"}), json.dumps({"error": f"API请求失败: {str(e)}"})
    except json.JSONDecodeError:
        return json.dumps({"error": "无效的JSON响应"}), json.dumps({"error": "无效的JSON响应"})
    except Exception as e:
        return json.dumps({"error": f"发生未知错误: {str(e)}"}), json.dumps({"error": f"发生未知错误: {str(e)}"})

    # 解析结果
    if "正常" in assistant_message :
        violation_categories = ["正常"]
        violation_texts = []
    else:
        # 分割结果行
        lines = [line.strip() for line in assistant_message .split('\n') if line.strip()]
        
        # 提取违规类别（第一行）
        violation_categories = []
        if lines:
            # 清理类别字符串
            categories_str = re.sub(r'[,\[\]]', ' ', lines[0])  # 去除多余标点
            violation_categories = [cat.strip() for cat in categories_str.split() if cat.strip()]
        
        # 处理违规文本（剩余行）
        violation_texts = []
        if len(lines) > 1:
            text_content = '\n'.join(lines[1:])
            
            # 尝试解析为标准列表
            try:
                # 先尝试直接eval
                parsed = eval(text_content)
                if isinstance(parsed, list):
                    violation_texts = [str(item).strip() for item in parsed if str(item).strip()]
                else:
                    violation_texts = [str(parsed).strip()]
            except:
                # 手动处理非常规格式
                text_content = text_content.strip("[]'\"")
                # 按常见分隔符分割
                split_texts = re.split(r'[,\n]', text_content)
                violation_texts = [text.strip(" '\"") for text in split_texts if text.strip()]
                
                # 如果还是空，则使用原始内容
                if not violation_texts:
                    violation_texts = [text_content]

    # 后处理：清理文本中的异常符号
    cleaned_texts = []
    for text in violation_texts:
        # 移除多余的引号和方括号
        text = re.sub(r'^[\'\"\[]+|[\'\"\]]+$', '', text)
        # 合并连续的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            cleaned_texts.append(text)
    
    violation_texts = cleaned_texts

    # 返回JSON格式
    categories_json = json.dumps({"violation_categories": violation_categories}, ensure_ascii=False)
    texts_json = json.dumps({"violation_texts": violation_texts}, ensure_ascii=False)
    return categories_json, texts_json
