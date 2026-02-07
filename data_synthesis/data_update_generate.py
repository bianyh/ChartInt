import argparse
import json
import os
from pathlib import Path

from echarts_to_png import EchartsToPNG
from echarts_and_data_to_png import EchartsToPNGWithData
from qwen3 import Qwen3Local
from api_llm import LLMClient
import tqdm


# 如果是类型提示需要 Union
from typing import Union
import subprocess
import sys

from utils import paser_html

os.chdir('/home/bianyuhan/chart2code')


def make_html_from_model_output(model_output: str) -> str:
    """如果模型直接返回完整 HTML 则原样使用，否则将其包进一个基本模板。"""
    if "<html" in model_output.lower():
        return model_output

    # 简单包装：提供 echarts 脚本依赖并在 #main 中插入模型返回的 JS
    template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>html,body{{margin:0;padding:0;overflow:hidden;height:100vh}}#main{{width:100%;height:100vh}}</style>
</head>
<body>
  <div id="main"></div>
  <script>
  try {{
    var chart = echarts.init(document.getElementById('main'));
    {model_output}
  }} catch(e) {{ console.error(e); }}
  </script>
</body>
</html>"""
    return template


def process_one(sample_dir: Path, echarts: EchartsToPNG, echarts_with_data: EchartsToPNGWithData, bot: Union[Qwen3Local, LLMClient], out_root: Path):
    sample_name = sample_dir.name
    out_dir = out_root / sample_name
    if os.path.exists(out_dir):
        print(f"{sample_name} 已存在，跳过")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = sample_dir / "chart.html"
    js_path = sample_dir / "code.js"
    orig_image = sample_dir / "chart.png"

    with open(html_path, 'r', encoding='utf-8') as file:
        line_count = len(file.readlines())
    if line_count > 200:
        print(f'文件的行数为：{line_count},跳过')
        return None

    if not html_path.exists():
        print(f"跳过 {sample_name}：找不到 chart.html")
        return None

    # 1) 使用 echarts_to_png 渲染原始 HTML（再次生成一份到输出目录）
    orig_render_path = out_dir / "orig_render.png"
    try:
        echarts.from_html_file(str(html_path), str(orig_render_path))
    except Exception as e:
        print(f"渲染原始 HTML 失败 {sample_name}: {e}")

    # 2) 使用文本模型读取原始 HTML 并生成复杂化的 ECharts HTML
    try:
        html_content = html_path.read_text(encoding="utf-8")
    except Exception:
        html_content = ""

    # 保存复杂化前的原始 code 和（如果存在的）原始图像
    (out_dir / "original_code.html").write_text(html_content, encoding="utf-8")

    js_content = js_path.read_text(encoding="utf-8")
    prompt = f"""
## 任务背景
我正在构建一个“图表代码复现与重构”的基准测试。输入是一段包含硬编码数据的 ECharts 配置代码。

## 你的任务
请基于我提供的 ECharts 代码，生成一道“将数据源替换为 CSV 文件”的评估题目。你需要完成以下四个步骤，并返回一个 JSON 对象：

1.  **提取并生成 CSV 格式**：
    * 分析原始代码中的数据格式。
    * 分析 CSV 应有的合理的表头（Header），根据原始代码的进行推断。

2.  **编写 CSV 文件生成脚本**：
    * 依据所需的数据结构，编写合适的Python代码，仅使用标准库 `random` 和 `csv`，生成符合格式的数据文件；
    * 生成的数据和原图要不一致；
    * 生成的数据量大一点；
    * 生成的数据存储到 data.csv 文件中。

3.  **生成参考代码 (Reference HTML)**：
    * 编写一个完整的 HTML 文件。
    * 在 `<script>` 中，使用 `fetch('data.csv')` API 异步读取数据。
    * 解析 CSV 文本（简单的 split 处理或类似逻辑），将其转换为 ECharts 需要的格式 `[[数字, 数字], ...]`。
    * **关键要求**：除了图表数据来源改变外，图表的所有其他样式必须与原始代码**完全保持一致**，不得修改任何视觉配置。

4.  **输出 JSON 格式**：
    请严格按照下方定义的 JSON 结构返回结果，不要包含 Markdown 代码块标记（```json），直接返回纯文本 JSON。

## 输入代码
{js_content}

## 目标 JSON 结构
{{
    "csv_content": "一段描述csv文件中的数据格式的文字，需指出具体的列名含义。",
    "csv_code": "用于生成csv文件的python代码",
    "reference_html": "完整的参考 HTML 字符串..."
}}
"""

    try:
        model_output = bot.ask(prompt=prompt)
    except Exception as e:
        print(f"模型生成代码失败 {sample_name}: {e}")
        model_output = ""
    model_output = json.loads(model_output)

    csv_content = model_output['csv_content']
    csv_code = model_output['csv_code']
    reference_html = model_output['reference_html']

    # 保存模型输出
    gen_code_path = out_dir / "generated_code.json"
    with open(gen_code_path, 'w') as f:
        json_output = json.dumps(model_output)
        f.write(json_output)

    html_output_path = out_dir / "generated.html"
    html_output_path.write_text(reference_html, encoding="utf-8")

    csv_code = f"import os\nos.chdir('/home/bianyuhan/chart2code/data_update/{sample_name}')\n" + csv_code
    csv_code_path = out_dir / "csv_gen.py"
    csv_code_path.write_text(csv_code, encoding="utf-8")

    csv_content_path = out_dir / "csv_content.txt"
    csv_content_path.write_text(csv_content, encoding="utf-8")



    # 执行csv_code脚本生成data.csv
    try:
        # 启动子进程运行脚本
        # sys.executable 确保使用当前运行主程序相同的 Python解释器
        result = subprocess.run(
            [sys.executable, csv_code_path], 
            capture_output=True, # 捕获输出
            text=True,           # 以文本形式而不是字节形式获取输出
            timeout=30           # 设置超时时间（秒），防止脚本卡死
        )

    # 检查执行结果
        if result.returncode == 0:
            print("执行成功！")
            print("输出内容:", result.stdout)
        else:
            print("执行出错！")
            print("错误信息:", result.stderr)

    except subprocess.TimeoutExpired:
        print("执行超时！")
    except Exception as e:
        print(f"发生未知错误: {e}")


    # 3) 将模型输出包装为 HTML 并渲染
    try:
        WORK_DIR = f'/home/bianyuhan/chart2code/data_update/{sample_name}/'
    
        # 你的文件名
        HTML_FILE = "generated.html"
        OUTPUT_PNG = "generated_render.png"
        # echarts.from_html_file(str(html_output_path), str(gen_render_path))
        echarts_with_data.process_chart(WORK_DIR, HTML_FILE, OUTPUT_PNG)
    except Exception as e:
        print(f"渲染生成 HTML 失败 {sample_name}: {e}")

    # # 尝试解析模型返回的 JSON
    # score_json = {"raw": score_text}
    # try:
    #     # 模型可能直接返回 JSON 字符串，也可能带文本，我们尝试提取第一个 json 对象
    #     import re

    #     m = re.search(r"\{.*\}", score_text, re.S)
    #     if m:
    #         score_json = json.loads(m.group(0))
    #     else:
    #         score_json = {"raw": score_text}
    # except Exception:
    #     score_json = {"raw": score_text}

    # 保存结果条目
    result = {
        "sample": sample_name,
        "csv_cotent": csv_content,
    }

    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./to_complex_data", help="数据目录，包含每条数据的子文件夹")
    parser.add_argument("--output-dir", default="./data_update", help="输出目录")
    # parser.add_argument("--model-path", default="/data/models/Qwen-8b-VL/Qwen3-VL-8B-Instruct", help="Qwen3-VL 模型路径或 model id")
    # parser.add_argument("--model-path", default="/data/models/Qwen-32b-VL/Qwen3-VL-32B-Instruct", help="Qwen3-VL 模型路径或 model id")
    parser.add_argument("--model-path", default="/data/models/Qwen3-32B", help="Qwen3-VL 模型路径或 model id")
    parser.add_argument("--device", default="cuda:2", help="设备，如 cpu 或 cuda")
    parser.add_argument("--max-samples", type=int, default=0, help="测试时只处理前 N 个样本，0 表示全部")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) 初始化渲染器和模型（模型加载可能很慢）
    echarts = EchartsToPNG(width=1200, height=800, output_dir='./')
    echarts_with_data = EchartsToPNGWithData(width=1200, height=800)

    model_path = args.model_path if args.model_path else "/data/models/Qwen-32b-VL/Qwen3-VL-32B-Instruct"
    print(f"加载模型: {model_path} (device={args.device})，如果是远程 id 会自动下载权重。")
    # bot = Qwen3Local(model_id_or_path=model_path, device=args.device)
    bot = LLMClient()

    # 2) 遍历样本目录
    all_samples = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]

    results = []
    for s in tqdm.tqdm(all_samples):
        try:
            print(f"处理: {s.name}")
            r = process_one(s, echarts, echarts_with_data, bot, out_root)
            if r:
                results.append(r)
        except:
            continue

    # 保存汇总
    (out_root / "results_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"处理完成，结果保存在 {out_root}")


if __name__ == "__main__":
    main()
