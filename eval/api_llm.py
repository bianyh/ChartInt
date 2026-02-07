import openai
from typing import Optional, Dict, Any, List
import time
import logging


import base64

def encode_image(image_path):
    """
    将本地图片路径转换为 Base64 字符串
    """
    with open(image_path, "rb") as image_file:
        # 读取二进制 -> 转 Base64 -> 解码为 utf-8 字符串
        return base64.b64encode(image_file.read()).decode('utf-8')

class LLMClient:
    """
    大语言模型API客户端类
    支持OpenAI兼容格式的API调用
    """
    
    def __init__(
        self,
        api_key: str = "sk-VwXfmWXpNw6YCJRZEf97F7Ac81F24457801aA9462d8b7979",
        base_url: Optional[str] = "https://api.chatweb.plus/v1/",
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        timeout: int = 60,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        初始化LLM客户端
        
        参数:
            api_key: API密钥
            base_url: 自定义API基础URL（用于兼容OpenAI的第三方服务）
            model: 模型名称
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            temperature: 生成文本的随机性（0-1）
            max_tokens: 生成文本的最大长度
            system_prompt: 系统提示词，用于设定AI角色和行为
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def ask(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        向大模型提问并获取回答
        
        参数:
            prompt: 用户提示词
            temperature: 覆盖默认的temperature值
            max_tokens: 覆盖默认的max_tokens值
            system_prompt: 覆盖默认的系统提示词
            **kwargs: 其他传递给API的参数
            
        返回:
            str: 模型的回答内容
            
        异常:
            Exception: 当所有重试都失败时抛出
        """
        
        # 使用传入的参数或默认值
        current_temp = temperature if temperature is not None else self.temperature
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_system = system_prompt if system_prompt is not None else self.system_prompt
        
        # 构建消息列表
        messages: List[Dict[str, str]] = []
        if current_system:
            messages.append({"role": "system", "content": current_system})
        messages.append({"role": "user", "content": prompt})
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=current_temp,
                    max_tokens=current_max_tokens,
                    timeout=600,
                    **kwargs
                )
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.warning(f"第 {attempt + 1} 次请求失败: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    self.logger.error("所有重试均失败")
                    raise Exception(f"API调用失败: {str(e)}")
                
                # 指数退避
                wait_time = 2  ** attempt
                self.logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
    def ask_msg(
        self,
        messages: List[Dict[str, str]] = [],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        向大模型提问并获取回答
        
        参数:
            prompt: 用户提示词
            temperature: 覆盖默认的temperature值
            max_tokens: 覆盖默认的max_tokens值
            system_prompt: 覆盖默认的系统提示词
            **kwargs: 其他传递给API的参数
            
        返回:
            str: 模型的回答内容
            
        异常:
            Exception: 当所有重试都失败时抛出
        """
        
        # 使用传入的参数或默认值
        current_temp = temperature if temperature is not None else self.temperature
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_system = system_prompt if system_prompt is not None else self.system_prompt
        
        # # 构建消息列表
        # messages: List[Dict[str, str]] = []
        # if current_system:
        #     messages.append({"role": "system", "content": current_system})
        # messages.append({"role": "user", "content": prompt})
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=current_temp,
                    max_tokens=current_max_tokens,
                    timeout=600,
                    **kwargs
                )
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.warning(f"第 {attempt + 1} 次请求失败: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    self.logger.error("所有重试均失败")
                    raise Exception(f"API调用失败: {str(e)}")
                
                # 指数退避
                wait_time = 2  ** attempt
                self.logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

    def ask_msg_out_json(
        self,
        messages: List[Dict[str, str]] = [],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        向大模型提问并获取回答
        
        参数:
            prompt: 用户提示词
            temperature: 覆盖默认的temperature值
            max_tokens: 覆盖默认的max_tokens值
            system_prompt: 覆盖默认的系统提示词
            **kwargs: 其他传递给API的参数
            
        返回:
            str: 模型的回答内容
            
        异常:
            Exception: 当所有重试都失败时抛出
        """
        
        # 使用传入的参数或默认值
        current_temp = temperature if temperature is not None else self.temperature
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_system = system_prompt if system_prompt is not None else self.system_prompt
        
        # # 构建消息列表
        # messages: List[Dict[str, str]] = []
        # if current_system:
        #     messages.append({"role": "system", "content": current_system})
        # messages.append({"role": "user", "content": prompt})
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=current_temp,
                    max_tokens=current_max_tokens,
                    timeout=600,
                    **kwargs
                )
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.warning(f"第 {attempt + 1} 次请求失败: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    self.logger.error("所有重试均失败")
                    raise Exception(f"API调用失败: {str(e)}")
                
                # 指数退避
                wait_time = 2  ** attempt
                self.logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    
    def ask_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        ** kwargs
    ) -> str:
        """
        带对话历史的多轮问答
        
        参数:
            prompt: 当前用户提问
            history: 对话历史，格式如 [{"role": "user", "content": "..."}, ...]
            **  kwargs: 其他传递给ask方法的参数
            
        返回:
            str: 模型的回答内容
        """
        # 构建包含历史的完整消息列表
        messages = history.copy()
        
        if self.system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # 临时修改消息列表并调用
        original_messages = messages
        try:
            # 通过临时修改client的调用方式来实现
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"带历史记录的请求失败: {str(e)}")
            raise

# 使用示例
if __name__ == "__main__":
    # 示例1：基础用法
    client = LLMClient(
        # api_key="your-api-key-here",
        model="gpt-4o",
        # system_prompt="你是一个乐于助人的助手。"
    )

    # 1. 准备图片路径
    ref_image_path = './chart2code/complex/images/0.jpg'
    gen_image_path = './chart2code/complex/images/0.jpg'

    # 2. 转为 Base64
    ref_base64 = encode_image(ref_image_path)
    gen_base64 = encode_image(gen_image_path)
    
    # messages=[
    #     {
    #     "role": "user",
    #     "content": [
    #         {"type": "text", "text": "比较这两张图的差距，并指出具体的颜色差异。"},
    #         {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": "https://haowallpaper.com/link/common/file/previewFileImg/15758358777205056", # 或 base64
    #         },
    #         },
    #         {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": "https://haowallpaper.com/link/common/file/previewFileImg/15889971878350208",
    #         },
    #         },
    #     ],
    #     }
    # ]

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "你是一个专业的图表评估专家。请对比这两张图（第一张是参考图，第二张是复现图），从布局、颜色和数据趋势三个方面评估复现的准确性，并只返回 0-10 的打分。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # 注意：这里必须拼接正确的前缀
                        "url": f"data:image/jpeg;base64,{ref_base64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{gen_base64}"
                    }
                }
            ]
        }
    ]
    
    print(client.ask_msg(messages=messages))

