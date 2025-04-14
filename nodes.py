import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import os
import torch
import folder_paths


# pdf加载节点
class PDFLoader:
    """PDF读取节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_path": ("STRING", {"default": "", "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("PDF_DOC",)
    RETURN_NAMES = ("pdf_document",)
    FUNCTION = "load_pdf"
    CATEGORY = "document/pdf"
    
    def load_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF文件不存在: {pdf_path}")
            
        try:
            pdf_doc = fitz.open(pdf_path)
            return (pdf_doc,)
        except Exception as e:
            raise ValueError(f"无法加载PDF: {str(e)}")
        
    @classmethod
    def IS_CHANGED(cls, pdf_path):
        if os.path.exists(pdf_path):
            m = os.path.getmtime(pdf_path)  # 获取文件的最后修改时间
            return float(m)

# pdf处理节点
class SealMigration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "source_pdf": ("PDF_DOC", {"description": "包含印章的源PDF文档"}),
                "target_pdf": ("PDF_DOC", {"description": "需要添加印章的目标PDF文档"}),
            },
            "optional": {
                "red_param": ("INT", {"default": 10, "min": 0, "max": 10, "step": 1, "description": "红色参数", "display": "slider"}),
                "white_param": ("INT", {"default": 10, "min": 0, "max": 10, "step": 1, "description": "白色参数", "display": "slider"}),
                "black_param": ("INT", {"default": 10, "min": 0, "max": 10, "step": 1, "description": "黑色参数", "display": "slider"}),
                "denoise_param": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1, "description": "去噪参数", "display": "slider"}),
                "source_page_num": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "description": "源PDF页码", "display": "number"}),
                "search_boundary": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1, "description": "印章搜索边界", "display": "number"}),
                "text_recognition_param": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1, "description": "文字识别参数", "display": "number"}),
                "target_page_num": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "description": "目标PDF页码", "display": "number"}),
                "seal_size": ("INT", {"default": 16, "min": 0, "max": 100, "step": 1, "description": "印章大小", "display": "number"}),
                "seal_pos_x": ("INT", {"default": -150, "min": -1000, "max": 1000, "step": 1, "description": "印章X偏移", "display": "number"}),
                "seal_pos_y": ("INT", {"default": -150, "min": -1000, "max": 1000, "step": 1, "description": "印章Y偏移", "display": "number"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "PDF_DOC", "IMAGE")
    RETURN_NAMES = ("印章图片", "处理结果pdf", "调试图像")
    FUNCTION = "process_pdf"
    CATEGORY = "document/pdf"

    def process_pdf(self, source_pdf, target_pdf, **kwargs):
        
        # 获取源页面
        source_page_num = kwargs.get("source_page_num", 1)
        if source_page_num < 1 or source_page_num > len(source_pdf):
            raise ValueError(f"源PDF页数无效，当前PDF共{len(source_pdf)}页")
        # 获取目标页面
        target_page_num = kwargs.get("target_page_num", 1)
        if target_page_num < 1 or target_page_num > len(target_pdf):
            raise ValueError(f"目标PDF页数无效，当前PDF共{len(target_pdf)}页")
        
        # 验证输入PDF文档
        if not isinstance(source_pdf, fitz.Document):
            raise ValueError("source_pdf 必须是有效的PDF文档对象")
        
        try:
            source_page_doc = source_pdf[source_page_num-1]
        except IndexError:
            raise ValueError(f"源PDF没有第{source_page_num}页")

        # 获取页面图像
        source_pix = source_page_doc.get_pixmap()
        img_bytes = source_pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 转换到HSV颜色空间（红色印章）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 获取颜色参数值（带默认值）
        red_param = kwargs.get("red_param", 10)
        white_param = kwargs.get("white_param", 10)
        black_param = kwargs.get("black_param", 10)
        denoise_param = kwargs.get("denoise_param", 5)
        search_boundary = kwargs.get("search_boundary", 100)
        text_recognition_param = kwargs.get("text_recognition_param", 1)
        seal_size = kwargs.get("seal_size", 16)
        seal_pos_x = kwargs.get("seal_pos_x", -150)
        seal_pos_y = kwargs.get("seal_pos_y", -150)
    
        # 定义红色范围（可根据实际印章颜色调整）
        lower_red1 = np.array([0, max(50, 250-20*red_param), max(50, 250-20*red_param)], np.float32)
        upper_red1 = np.array([10, 255, 255], np.float32)
        lower_red2 = np.array([160, max(50, 250-20*red_param), max(50, 250-20*red_param)], np.float32)
        upper_red2 = np.array([180, 255, 255], np.float32)
    
        # 创建红色区域的掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    
        # 形态学操作去除噪点
        denoise_param = kwargs.get("denoise_param", 5)
        kernel = np.ones((max(1, denoise_param), max(1, denoise_param)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的轮廓（假设印章是最大的红色区域）
        if not contours:
            raise ValueError("未找到印章图案")
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        # 提取印章区域（增加一些边界）
        margin = search_boundary  # 增加边界值，确保印章完整
        seal = img[max(0, y-margin):min(y+h+margin, img.shape[0]), 
                max(0, x-margin):min(x+w+margin, img.shape[1])]
    
        # 去除白色背景
        seal_hsv = cv2.cvtColor(seal, cv2.COLOR_BGR2HSV)
        lower_white = np.array([180-18*white_param, 30-3*white_param, 250-5*white_param], dtype=np.float32)  # 白色的HSV范围
        upper_white = np.array([180, 30, 255], dtype=np.float32)
        white_mask = cv2.inRange(seal_hsv, lower_white, upper_white)
    
        # 转换为4通道（RGBA）
        seal = cv2.cvtColor(seal, cv2.COLOR_BGR2BGRA)
        seal[white_mask == 255] = [0, 0, 0, 0]  # 将白色背景设置为透明
        
        # 去除黑色文字
        lower_black = np.array([180-18*black_param, 255-25.5*black_param, 200-20*black_param], dtype=np.float32)  # 黑色的HSV范围
        upper_black = np.array([180, 255, 200], dtype=np.float32)  # 调整上限以覆盖更多黑色
        black_mask = cv2.inRange(seal_hsv, lower_black, upper_black)
    
        # 扩展黑色区域的掩码以确保完全覆盖
        kernel = np.ones((text_recognition_param, text_recognition_param), np.uint8)  # 增大核大小以扩展黑色区域
        black_mask = cv2.dilate(black_mask, kernel, iterations=3)  # 增加迭代次数以扩展更多区域
        
        # 将黑色文字设置为透明
        seal[black_mask == 255] = [0, 0, 0, 0]
        
        # 将印章区域从OpenCV图像转换为PIL图像
        seal_pil = Image.fromarray(cv2.cvtColor(seal, cv2.COLOR_BGRA2RGBA))

        # 印章图像处理部分
        seal_rgb = cv2.cvtColor(seal, cv2.COLOR_BGRA2RGB)
        seal_rgba = cv2.cvtColor(seal, cv2.COLOR_BGRA2RGBA)
        
        # 修正1: 确保数组形状正确
        if seal_rgb.ndim == 2:
            seal_rgb = cv2.cvtColor(seal_rgb, cv2.COLOR_GRAY2RGBA)

        # 确保是uint8类型
        seal_rgb = seal_rgb.astype(np.uint8)
        print(f"Seal numpy shape: {seal_rgb.shape}")  # 应为 [H,W,3]

        # 修正2: 转换为PIL图像
        seal_pil = Image.fromarray(
            seal_rgba if seal_rgb.shape[-1] == 4 else cv2.cvtColor(seal_rgba, cv2.COLOR_RGB2RGBA),
            mode='RGBA'
        )

        seal_tensor = torch.from_numpy(seal_rgb).float()
        print(f"原始印章tensor形状: {seal_tensor.shape}")  # 应该是 [H,W,C]
        seal_tensor = seal_tensor.permute(2, 0, 1) / 255.0
        print(f"调整后印章tensor形状: {seal_tensor.shape}")  # 应该是 [1,4,H,W]

        # 在处理函数中返回中间结果
        debug_img = cv2.merge([mask]*3)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        print(f"原始调试图像形状: {debug_img.shape}")  # 应该是 [H,W,3]
        debug_tensor = torch.from_numpy(debug_img).float()
        debug_tensor = debug_tensor.permute(2, 0, 1) / 255.0
        print(f"调整后调试tensor形状: {debug_tensor.shape}")  # 应该是 [1,3,H,W]

        # 印章合成部分
        # 获取需要添加印章的页面
        target_page_doc = target_pdf[target_page_num-1]

        # 获取页面尺寸
        page_rect = target_page_doc.rect
            
        # 调整印章大小为原来的2倍
        seal = seal_pil.resize((int(seal_pil.width * seal_size/10), int(seal_pil.height * seal_size/10)), Image.Resampling.LANCZOS)
        
        # 将PIL图像转换为字节流
        img_byte_arr = io.BytesIO()
        seal.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 使用OCR识别最后一页的文字区域
        target_pix = target_page_doc.get_pixmap()
        img_bytes = target_pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用阈值分割来检测文字区域
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 寻找轮廓作为文字区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 构造一个字典来模拟文字区域信息
        d = {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 过滤掉过小的区域（可能是噪声）
            if w > 10 and h > 10:
                d['text'].append("dummy")  # 模拟文字内容
                d['left'].append(x)
                d['top'].append(y)
                d['width'].append(w)
                d['height'].append(h)
        
        # 找到最后的文字位置
        if d['text']:
            max_bottom = 0
            max_right = 0
            for i in range(len(d['text'])):
                if d['text'][i].strip():
                    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                    max_bottom = max(max_bottom, y + h)
                    max_right = max(max_right, x + w)
            
            x0 = max_right + seal_pos_x  # 在文字右侧留出边距
            y0 = max_bottom + seal_pos_y  # 在文字下方留出边距
        else:
            # 如果没有文字，默认放置在右下角，留出200个单位的边距
            x0 = page_rect.width - 200
            y0 = page_rect.height - 200

        x1 = x0 + seal.width / 2  # 印章宽度
        y1 = y0 + seal.height / 2  # 印章高度
        
        # 创建印章位置矩形
        seal_rect = fitz.Rect(x0, y0, x1, y1)

        # 创建新PDF用于修改(不直接修改输入)
        modified_pdf = fitz.open()
        modified_pdf.insert_pdf(target_pdf)

        # 在PDF上添加印章图像
        modified_page = modified_pdf[target_page_num-1]
        modified_page.insert_image(seal_rect,stream=img_byte_arr,)

        # 确保返回的 modified_pdf 是有效的
        if not modified_pdf:
            raise ValueError("生成的PDF文档无效")
        
        print(f"Seal tensor shape: {seal_tensor.shape}")  # 应该如 [1,4,H,W]
        print(f"Debug tensor shape: {debug_tensor.shape}") # 应该如 [1,3,H,W]
        print(f"Seal tensor dtype: {seal_tensor.dtype}")   # 应该是torch.float32
        print(f"Seal tensor range: {seal_tensor.min()}-{seal_tensor.max()}") # 应该在0-1之间

        assert seal_rgb.dtype == np.uint8, f"印章图像类型错误: {seal_rgb.dtype}"
        assert debug_img.dtype == np.uint8, f"调试图像类型错误: {debug_img.dtype}"

        return seal_tensor, modified_pdf, debug_tensor

# pdf保存节点
class PDFSaver:
    """PDF保存节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "pdf_document": ("PDF_DOC",),
                "filename": ("STRING", {"default": "result.pdf", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_pdf"
    CATEGORY = "document/pdf"
    
    def save_pdf(self, pdf_document, filename, prompt=None, extra_pnginfo=None):
        # 确保文件名以.pdf结尾
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        output_dir = folder_paths.get_output_directory()
        full_path = os.path.join(output_dir, filename)
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存PDF
        pdf_document.save(full_path)
        pdf_document.close()
        
        return {"ui": {"pdf": [filename]}}
