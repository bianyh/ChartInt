from eval import Evaluator
import os
import json
from tqdm import tqdm

import time 

t = time.localtime()
print(t)


def batch_evaluate():
    # 初始化评估器
    evaluator = Evaluator()
    
    # 确保结果目录存在
    os.makedirs('./final_result', exist_ok=True)
    
    # 读取现有结果，支持增量评估
    # result_file = './final_result/result.json'
    result_file = './final_result/result2.json'
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}
    
    # 遍历 ./eval_result/ 下的模型文件夹
    eval_result_dir = './eval_result'
    if not os.path.exists(eval_result_dir):
        print("eval_result directory not found.")
        return
    
    for model in os.listdir(eval_result_dir):


        if model != 'Qwen3-VL-2B-1':
            continue

        
        model_path = os.path.join(eval_result_dir, model)
        if not os.path.isdir(model_path):
            continue
        
        if model not in results:
            results[model] = {}
        
        for task in os.listdir(model_path):
            if task == 'chart_mimic' or task == 'high_complex':
                print(f'跳过{task}')
                continue
            print(f'现在是模型{model} ---- 任务{task}')

            task_path = os.path.join(model_path, task)
            if not os.path.isdir(task_path):
                continue
            
            if task not in results[model]:
                results[model][task] = {}
            
            for sample in tqdm(os.listdir(task_path)):
                sample_path = os.path.join(task_path, sample)
                if not os.path.isdir(sample_path):
                    continue
                
                # # qwen-2b重新测试
                # if task != 'before_complex' and task != 'after_complex':
                #     continue
                # 检查是否已有结果
                if sample in results[model][task] and 'error_count' in results[model][task][sample]:
                    print(f"Skipping already evaluated: {model}/{task}/{sample}")
                    continue
                
                # 准备文件路径
                gen_code_path = os.path.join(sample_path, 'gen.html')
                orig_png = os.path.join(sample_path, 'orig.png')
                eval_png = os.path.join(sample_path, 'eval.png')
                gen_png = os.path.join(sample_path, 'gen.png')
                
                # 确定 image2_path
                if os.path.exists(eval_png):
                    image1_path = eval_png
                elif os.path.exists(gen_png):
                    image1_path = orig_png
                else:
                    print(f"Missing image files for {model}/{task}/{sample}, skipping.")
                    continue
                
                image2_path = gen_png
                if not os.path.exists(image1_path) or not os.path.exists(gen_code_path):
                    print(f"Missing required files for {model}/{task}/{sample}, skipping.")
                    continue
                
                # 评估
                data_update = (task == 'data_update')
                data_interaction = (task == 'data_interaction')
                try:
                    result = evaluator(gen_code_path, image1_path, image2_path, data_update, data_interaction)
                    results[model][task][sample] = result
                    print(f"Evaluated: {model}/{task}/{sample}")
                except Exception as e:
                    print(f"Error evaluating {model}/{task}/{sample}: {e}")
                    continue
    
                # 保存结果
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print("Batch evaluation completed.")

if __name__ == "__main__":
    batch_evaluate()


