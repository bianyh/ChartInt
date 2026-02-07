import re



def paser_html(model_output: str) -> str:
    """从模型返回的```html内容中提取 ECharts 代码部分。
    
    Args:
        model_output: 包含```html代码块的字符串
        
    Returns:
        提取的HTML/ECharts代码字符串，如果没有找到则返回空字符串
    """
    # 正则模式：匹配 ```html ... ``` 代码块
    pattern = r'```html\s*(.*?)\s*```'
    
    # 查找所有匹配的代码块，re.DOTALL使.能匹配换行符
    matches = re.findall(pattern, model_output, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return model_output  # 如果没有匹配，返回原始输出
    
    # 返回第一个匹配结果（通常只有一个ECharts HTML代码块）
    return matches[0].strip()


import json
import re
import logging

# 设置日志，科研中记录错误至关重要
logger = logging.getLogger(__name__)

def extract_and_parse_json(llm_output: str):
    """
    从 LLM 的输出字符串中鲁棒地提取并解析 JSON。
    
    Args:
        llm_output (str): LLM 返回的原始字符串
        
    Returns:
        dict/list: 解析后的 JSON 对象
        None: 解析失败
    """
    try:
        # 1. 预处理：去除首尾空白
        content = llm_output.strip()
        
        # 2. 尝试提取 Markdown 代码块中的内容 (支持 json, markdown, 或无标记)
        # 正则解释: 
        # ```             : 匹配开头的三个反引号
        # (?:json|markdown)? : 非捕获组，匹配可选的语言标记 json 或 markdown
        # \s* : 匹配可能的换行或空格
        # (.*?)           : 非贪婪匹配，捕获核心内容
        # \s* : 匹配可能的换行或空格
        # ```             : 匹配结尾的三个反引号
        pattern = r"```(?:json|markdown)?\s*(.*?)\s*```"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            content = match.group(1)
        else:
            # 3. 如果没有 Markdown 标记，尝试寻找最外层的 {} 或 []
            # 这能处理 LLM 在 JSON 前后说了废话的情况
            json_start = content.find('{')
            json_end = content.rfind('}')
            
            # 如果没找到 {}，可能是列表 []
            if json_start == -1:
                json_start = content.find('[')
                json_end = content.rfind(']')
            
            if json_start != -1 and json_end != -1:
                content = content[json_start : json_end + 1]
            else:
                # 既没有代码块，也没找到括号，说明输出格式完全崩了
                logger.warning(f"Failed to find JSON delimiters in output: {llm_output[:50]}...")
                return None

        # 4. 尝试解析
        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}")
        logger.debug(f"Content causing error: {content}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}")
        return None
