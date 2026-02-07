from playwright.sync_api import sync_playwright
from complex.O_MeC import calculate_local_image_omec
import json
import pandas as pd
import ast
from skimage import color
import numpy as np
from complex.O_TextInkRatio import calculate_local_image_otir
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from api_llm import LLMClient, encode_image
from utils import extract_and_parse_json

import os
os.chdir('/home/bianyuhan/chart2code')


class OCREvaluator:
    def __init__(self, ocr_model):
        self.ocr = ocr_model
        # 设定文本匹配的相似度阈值（0.85表示允许少量OCR字符错误）
        self.text_match_threshold = 0.85
        # 设定OCR置信度阈值，过滤掉低质量的识别结果
        self.ocr_confidence_threshold = 0.5 

    def _string_similarity(self, s1, s2):
        """计算两个字符串的相似度 (0-1)"""
        return SequenceMatcher(None, s1, s2).ratio()

    def _preprocess_ocr_df(self, df):
        """
        处理 OCR DataFrame：
        1. 过滤低置信度结果
        2. 去除空文本
        3. 标准化文本（转小写、去除首尾空格）
        """
        if df is None or df.empty:
            return []

        # 1. 过滤掉置信度过低的结果 (利用 'Score' 列)
        # 假设 Score 是 0-1 之间的浮点数
        df_filtered = df[df['Score'] > self.ocr_confidence_threshold].copy()

        # 2. 提取文本列
        texts = df_filtered['OCR result'].astype(str).tolist()

        # 3. 文本清洗
        clean_texts = []
        for t in texts:
            t = t.strip().lower()
            # 过滤掉无意义的单字符（除非是数字）
            if len(t) > 0:
                clean_texts.append(t)
        
        return clean_texts

    def _compute_f1(self, list_ref, list_gen):
        """核心算法：使用匈牙利算法计算 F1-Score"""
        n_ref = len(list_ref)
        n_gen = len(list_gen)

        if n_ref == 0 and n_gen == 0: return 1.0
        if n_ref == 0 or n_gen == 0: return 0.0

        # 构建代价矩阵 (Cost Matrix)
        cost_matrix = np.zeros((n_ref, n_gen))
        sim_matrix = np.zeros((n_ref, n_gen))

        for i, ref_txt in enumerate(list_ref):
            for j, gen_txt in enumerate(list_gen):
                # 特殊处理：对于数字，可以加入更严格的逻辑（可选）
                score = self._string_similarity(ref_txt, gen_txt)
                sim_matrix[i, j] = score
                cost_matrix[i, j] = 1 - score

        # 匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 统计匹配成功的数量 (True Positives)
        tp = 0
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] >= self.text_match_threshold:
                tp += 1

        # 计算 Precision, Recall, F1
        precision = tp / n_gen if n_gen > 0 else 0
        recall = tp / n_ref if n_ref > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def OCR_score(self, image1_path: str, image2_path: str):
        """
        计算图表复现的 OCR 文本相似度 F1 分数
        """
        # 1. 调用您的外部函数获取 DataFrame
        # image1 通常作为 Reference (参考图), image2 作为 Generated (生成图)
        _, OCR_result1 = calculate_local_image_otir(image1_path, ocr_model=self.ocr)
        _, OCR_result2 = calculate_local_image_otir(image2_path, ocr_model=self.ocr)

        # 2. 从 DataFrame 中提取并清洗文本列表
        ref_texts = self._preprocess_ocr_df(OCR_result1)
        gen_texts = self._preprocess_ocr_df(OCR_result2)

        # 3. 计算 F1 分数
        final_score = self._compute_f1(ref_texts, gen_texts)

        return final_score



class Evaluator:
    def __init__(self, 
                 color_df_path : str = '/home/bianyuhan/chart2code/complex/HeerStone_colorNaming.xlsx',
                 color_data_path : str = '/home/bianyuhan/chart2code/complex/HeerStone_colorSimilarity.xlsx',
                 llm_prompt_path: str = "./prompts/eval.txt",
                 interaction_llm_prompt_path = "./prompts/interaction_eval.txt",
                 ):
        self.color_df_path = color_df_path
        self.color_data_path = color_data_path
        self.ocr = PaddleOCR(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,lang="ch")
        self.ocr_evaluator = OCREvaluator(ocr_model=self.ocr)
        with open(llm_prompt_path, 'r', encoding='utf-8') as f:
            self.llm_prompt = f.read()
        with open(interaction_llm_prompt_path, 'r', encoding='utf-8') as f:
            self.interaction_llm_prompt = f.read()
        self.llm_judger = LLMClient(model="gpt-4o")


    def init_color(self):
        self.color_df = pd.read_excel(self.color_df_path)
        self.color_data = pd.read_excel(self.color_data_path)
        # Converting string representation of list to actual list
        self.color_data['Similar_name'] = self.color_data['Similar_name'].apply(ast.literal_eval)
        # Create a dictionary to map each color to its similar colors
        self.similar_colors_dict = dict(zip(self.color_data['Color_name'], self.color_data['Similar_name']))

    def calculate_color(self, image_path):
        self.init_color()
        color_nums, color_list = calculate_local_image_omec(image_path, self.color_df, self.similar_colors_dict, threshold=14)
        return color_nums, color_list
    
    def compute_color_hard_set_match(self, image1_path: str, image2_path: str):
            """
            计算颜色列表的严格集合匹配 (IoU - Intersection over Union)
            分值范围: 0.0 ~ 1.0
            """
            # 1. 提取颜色列表
            # 注意：修正了你原代码中的拼写 'canculate' -> 'calculate'
            _, image1_list = self.calculate_color(image1_path)
            _, image2_list = self.calculate_color(image2_path)

            # 2. 转换为集合 (Set)
            # Set 会自动去重，符合 "集合匹配" 的定义
            set1 = set(c for c in image1_list)
            set2 = set(c for c in image2_list)

            # 3. 处理空集合情况防止除零错误
            if len(set1) == 0 and len(set2) == 0:
                return 1.0 # 两个都没颜色，认为一致
            if len(set1) == 0 or len(set2) == 0:
                return 0.0 # 其中一个没颜色，完全不匹配

            # 4. 计算交集和并集
            intersection = set1.intersection(set2) # A ∩ B
            union = set1.union(set2)               # A ∪ B

            # 5. 计算 Jaccard Index (IoU)
            iou_score = len(intersection) / len(union)

            return iou_score



    @staticmethod
    def execute_html_in_sandbox(html_content: str, timeout: int = 10):
        """
        在沙盒中执行HTML并捕获所有错误（修复版）
        """
        result = {
            "success": False,
            "console_logs": [],
            "js_errors": [],
            "page_errors": [],
            "content": None
        }
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                ]
            )
            
            context = browser.new_context(
                java_script_enabled=True,
                bypass_csp=True,
                service_workers="block",
            )
            
            page = context.new_page()
            
            # ===== 核心修复：三重错误捕获机制 =====
            
            # 第一重：监听控制台错误（最可靠）
            def handle_console(msg):
                log_entry = {
                    "type": msg.type,
                    "text": msg.text,
                    "location": f"{msg.location.get('url', 'unknown')}:{msg.location.get('lineNumber', 0)}"
                }
                result["console_logs"].append(log_entry)
                
                # 主动将console.error视为JS错误
                if msg.type == "error":
                    result["js_errors"].append({
                        "message": msg.text,
                        "source": "console.error"
                    })
                
            page.on("console", handle_console)
            
            # 第二重：监听未捕获异常
            def handle_page_error(error):
                # 修复：确保错误对象正确序列化
                error_info = {
                    "message": str(error),
                    "name": getattr(error, 'name', 'Error'),
                    "stack": getattr(error, 'stack', 'No stack trace')
                }
                result["js_errors"].append(error_info)
                
            page.on("pageerror", handle_page_error)
            
            # 第三重：在页面中注入全局错误处理器（最保险）
            page.add_init_script("""
                window.__sandbox_errors = [];
                
                // 捕获未处理的Promise错误
                window.addEventListener('unhandledrejection', event => {
                    window.__sandbox_errors.push({
                        type: 'unhandledrejection',
                        message: event.reason.toString(),
                        stack: event.reason.stack || ''
                    });
                    console.error('Caught unhandledrejection:', event.reason);
                });
                
                // 捕获常规JS错误
                window.addEventListener('error', event => {
                    window.__sandbox_errors.push({
                        type: 'error',
                        message: event.message,
                        filename: event.filename,
                        lineno: event.lineno,
                        colno: event.colno
                    });
                    console.error('Caught error:', event.message);
                });
            """)
            
            # 修复：使用更可靠的加载策略
            try:
                # 先设置基本HTML内容，再替换为完整内容
                page.set_content(html_content, timeout=timeout * 1000, wait_until='domcontentloaded')
                
                # 等待JS执行完成
                page.wait_for_timeout(1000)  # 比time.sleep更可靠
                
                # 提取注入的错误信息
                injected_errors = page.evaluate("() => window.__sandbox_errors")
                if injected_errors:
                    result["js_errors"].extend(injected_errors)
                
                result["content"] = page.content()
                result["success"] = True
                
            except Exception as e:
                result["page_errors"].append(f"Execution error: {str(e)}")
                
            finally:
                context.close()
                browser.close()
        
        return result

    # 测试函数
    @staticmethod
    def test_sandbox(test_html_with_errors: str):

        print("=== 执行HTML ===")
        result = Evaluator.execute_html_in_sandbox(test_html_with_errors)
        
        # print(f"执行成功: {result['success']}")
        # print(f"JS错误数量: {len(result['js_errors'])}")
        # print(f"控制台日志: {len(result['console_logs'])} 条")
        
        # # 详细错误信息
        # if result['js_errors']:
        #     print("\n--- 捕获的错误详情 ---")
        #     for i, error in enumerate(result['js_errors'], 1):
        #         print(f"{i}. {error.get('message', 'Unknown error')}")
        #         if 'stack' in error:
        #             print(f"   Stack: {error['stack'][:100]}...")
        
        # # 控制台日志
        # if result['console_logs']:
        #     print("\n--- 控制台日志 ---")
        #     for log in result['console_logs']:
        #         print(f"[{log['type']}] {log['text']}")

        return len(result['js_errors'])
    
    def OCR_score(self, image1_path: str, image2_path: str):
        return self.ocr_evaluator.OCR_score(image1_path=image1_path, image2_path=image2_path)
    
    def gpt_score(self, image1_path: str, image2_path: str, data_interaction: bool = False):
        # 转为 Base64
        ref_base64 = encode_image(image1_path)
        gen_base64 = encode_image(image2_path)

        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.llm_prompt if data_interaction is False else self.interaction_llm_prompt
                },
                {
                    "type": "text",
                    "text": "original image as follow:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # 注意：这里必须拼接正确的前缀
                        "url": f"data:image/jpeg;base64,{ref_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "generated image as follow:"
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
    
        result = self.llm_judger.ask_msg_out_json(messages=messages)

        # 从result中解析出json结构：（需要能支持带有```markdown的格式）
        
        parsed_data = extract_and_parse_json(result)

        return parsed_data

    def __call__(self, gen_code_path: str, image1_path: str, image2_path: str, data_update: bool = False, data_interation: bool = False):
        # 进行完整的评估，使用已经实现的全部这些类方法完成完整的评估并返回最终评估结果
        # 先执行判断代码错误量
        gen_code = open(gen_code_path, 'r', encoding='utf-8').read()
        if 'html>' not in gen_code or '</html>' not in gen_code:
            return {
                "ocr_f1": 0.0,
                "color_iou": 0.0,
                "gpt_evaluation": {
                    "overall_score": 0.0,
                    "reasoning": "Generated code has JavaScript errors, cannot evaluate further."
                }
            }
        error_count = self.test_sandbox(gen_code)
        # 代码无错误，继续进行后续评估
        ocr_f1 = self.OCR_score(image1_path=image1_path, image2_path=image2_path)
        color_iou = self.compute_color_hard_set_match(image1_path=image1_path,
                                                        image2_path=image2_path)
        gpt_evaluation = self.gpt_score(image1_path=image1_path, image2_path=image2_path, data_interaction=data_interation)
        return {
            "error_count": error_count if data_update is False else error_count-1,
            "ocr_f1": ocr_f1,
            "color_iou": color_iou,
            "gpt_evaluation": gpt_evaluation
        }






    


if __name__ == "__main__":
    # print(Evaluator.test_sandbox("asdi"))
    evaluator = Evaluator()


    # iou_score = evaluator.compute_color_hard_set_match(image1_path='./complex/images/0.jpg', image2_path='./complex/images/2.jpg')

    # print(iou_score)


    print(evaluator(gen_code_path='./eval_result/deepseek-7b/data_interaction/141/gen.html', 
                    image1_path='./eval_result/deepseek-7b/data_interaction/141/orig.png', 
                    image2_path='./eval_result/deepseek-7b/data_interaction/141/gen.png', 
                    data_interation=True))

    pass