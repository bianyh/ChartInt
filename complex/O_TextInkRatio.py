import os
import numpy as np
from PIL import Image
import pandas as pd
from paddleocr import PaddleOCR



def ocr_single_img(image, ocr=None):
    if ocr==None:
        ocr = PaddleOCR(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,lang="ch")
    try:
        img_array = np.array(image)
        result = ocr.predict(input=img_array)

        result = result[0]
    except Exception as e:
        print(f"OCR 识别失败: {e}")
        result = {'rec_boxes':np.array([]),'rec_texts':[],'rec_scores':[]}

    image_size=image.size
    boxes=result['rec_boxes']

    if len(boxes)>0:
        txts=result['rec_texts']
        scores=result['rec_scores']

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        total_area = np.sum(areas)

     #   im_show = draw_ocr(image, boxes, txts=None, scores=None)
     #   im_show = Image.fromarray(im_show)
        df = pd.DataFrame({'OCR result':txts,'Score':scores,'Detect box':[box.tolist() for box in boxes]})


    else:
        boxes=[]
        txts=[]
        scores=[]
        total_area=0
        df = pd.DataFrame({'OCR result':txts,'Score':scores,'Detect box':boxes})
        # im_show=image
    return df,total_area,image_size

def calculate_local_image_otir(img_path, ocr_model=None):
    """
    计算本地图片的 O.TiR (Text-ink Ratio / 文本区域占比)。

    参数:
    - img_path: str, 本地图片的完整路径 (例如 'C:/images/test.jpg')
    - ocr_model: PaddleOCR 对象 (可选). 如果在外部已经初始化，传入此参数可大幅提高速度。
                 如果为 None，函数内部会初始化一个新的 OCR 对象。

    返回:
    - otir_value: float, 文本区域占全图面积的百分比 (例如 12.5 代表 12.5%)
    - debug_info: dict, 包含具体的识别文本、分数和框坐标 (用于调试)
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到文件: {img_path}")

    # 2. 如果没有传入 OCR 模型，则在内部初始化 (注意：这会比较慢)
    if ocr_model is None:
        print("正在初始化 PaddleOCR 模型...")
        ocr_model = PaddleOCR(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,lang="ch")

    # 3. 加载图片并获取图片总面积
    try:
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        image_area = width * height
    except Exception as e:
        print(f"图片读取失败: {e}")
        return 0.0, {}

    # 4. 执行 OCR 预测
    # result 结构通常为 [[[[x1,y1],[x2,y2]..], (text, score)], ...]
    # result = ocr_model.ocr(img_path)
    OCR_result,bounding_box_areas,image_size = ocr_single_img(image, ocr=ocr_model)
    

    O_TiR = round((bounding_box_areas / image_area), 2)

    return O_TiR, OCR_result

# 示例用法
if __name__ == "__main__":
    img_path = "C:\\Users\\11494\\Desktop\\data\\output\\output\\97\\chart.png"  # 替换为你的图片路径
    otir, info = calculate_local_image_otir(img_path)
    print(f"O.TiR: {otir}%")
    print("Debug Info:", info)