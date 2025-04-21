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
                # "red_param": ("INT", {"default": 10, "min": 0, "max": 10, "step": 1, "description": "红色参数", "display": "slider"}),
                "white_param": ("INT", {"default": 10, "min": 0, "max": 10, "step": 1, "description": "白色参数", "display": "slider"}),
                "black_param": ("INT", {"default": 10, "min": 0, "max": 10, "step": 1, "description": "黑色参数", "display": "slider"}),
                "denoise_param": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1, "description": "去噪参数", "display": "slider"}),
                "enhance_param": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1, "description": "印章颜色增强强度", "display": "slider"}),
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

        # 2. 图像增强
        enhanced_img = self.enhance_image(img)

        # 转换到HSV颜色空间（红色印章）
        hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

        # 获取颜色参数值（带默认值）
        # red_param = kwargs.get("red_param", 10)
        white_param = kwargs.get("white_param", 10)
        black_param = kwargs.get("black_param", 10)
        denoise_param = kwargs.get("denoise_param", 5)
        enhance_param = kwargs.get("enhance_param", 3)
        search_boundary = kwargs.get("search_boundary", 100)
        text_recognition_param = kwargs.get("text_recognition_param", 1)
        seal_size = kwargs.get("seal_size", 16)
        seal_pos_x = kwargs.get("seal_pos_x", -150)
        seal_pos_y = kwargs.get("seal_pos_y", -150)
    
        # # 定义红色范围（可根据实际印章颜色调整）
        # lower_red1 = np.array([0, max(1, 255-25.5*red_param), max(1, 255-25.5*red_param)], np.float32)
        # upper_red1 = np.array([10, 255, 255], np.float32)
        # lower_red2 = np.array([160, max(1, 255-25.5*red_param), max(1, 255-25.5*red_param)], np.float32)
        # upper_red2 = np.array([180, 255, 255], np.float32)
        
        # # 创建红色区域的掩码
        # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # mask = cv2.bitwise_or(mask1, mask2)  # 合并两个掩码

        mask = self.adaptive_color_extraction(hsv)

        # 5. 形态学处理
        denoise_param = kwargs.get("denoise_param", 5)
        processed_mask = self.advanced_morphology(mask, denoise_param)
        
        # # 提高精度：增加形态学操作的迭代次数
        # kernel = np.ones((max(1, denoise_param), max(1, denoise_param)), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 增加迭代次数
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)   # 增加迭代次数

        # # 提高完整度：对掩码进行膨胀操作以填补缺失区域
        # mask = cv2.dilate(mask, kernel, iterations=1)
    
        # # 寻找轮廓
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. 提取印章轮廓
        contour = self.extract_seal_contours(processed_mask, img)
        
        # 找到最大的轮廓（假设印章是最大的红色区域）
        if contour is None:
            raise ValueError("未找到印章图案")
        
        # largest_contour = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(largest_contour)
        x, y, w, h = cv2.boundingRect(contour)
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
            seal_rgba if seal_rgba.shape[-1] == 4 else cv2.cvtColor(seal_rgba, cv2.COLOR_RGB2RGBA),
            mode='RGBA'
        )

        if enhance_param > 0:
            seal_pil = self.enhance_seal_color(seal_pil, enhance_param)

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
        
        # 使用OCR识别文字区域
        target_pix = target_page_doc.get_pixmap()
        img_bytes = target_pix.tobytes("png")
        ocr_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
        
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
            y0 = max_bottom - seal_pos_y  # 在文字下方留出边距
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
    
    # 替换原有的固定阈值方法
    def adaptive_color_extraction(self, hsv_img):
        # 分离HSV通道
        h, s, v = cv2.split(hsv_img)
        
        # 自适应计算红色区域阈值
        red_mask1 = cv2.inRange(h, 0, 10)
        red_mask2 = cv2.inRange(h, 160, 180)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 自适应饱和度阈值
        s_mean = np.mean(s[red_mask > 0]) if np.count_nonzero(red_mask) > 0 else 0
        s_thresh = max(30, min(s_mean * 0.7, 100))
        saturation_mask = cv2.inRange(s, s_thresh, 255)
        
        # 自适应亮度阈值
        v_mean = np.mean(v[red_mask > 0]) if np.count_nonzero(red_mask) > 0 else 0
        v_thresh = max(50, min(v_mean * 0.7, 200))
        value_mask = cv2.inRange(v, v_thresh, 255)
        
        # 组合所有条件
        final_mask = cv2.bitwise_and(red_mask, saturation_mask)
        final_mask = cv2.bitwise_and(final_mask, value_mask)
        
        return final_mask

    # 在提取印章前增加图像增强步骤
    def enhance_image(self, img):
        # 直方图均衡化
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 锐化图像
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        l = cv2.filter2D(l, -1, kernel)
        
        lab = cv2.merge((l,a,b))
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_img
    
    # 改进的形态学处理方法
    def advanced_morphology(self, mask, denoise_param):
        # 自适应结构元素大小
        kernel_size = max(3, min(denoise_param, 11))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 分步处理
        # 1. 闭运算填充小孔
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # 2. 开运算去除小噪点
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        # 3. 选择性膨胀
        dilated = cv2.dilate(opened, kernel, iterations=1)
        
        return dilated
    
    # 改进的轮廓提取方法
    def extract_seal_contours(self, mask, original_img):
        # 1. 从mask中寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 2. 基于多维度特征过滤轮廓
        valid_contours = []
        
        # 转换原始图像为LAB色彩空间，用于颜色一致性检查
        lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 忽略小区域
                continue
                
            # 计算轮廓的圆形度
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 计算矩形度
            _, _, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # 获取轮廓区域的平均颜色
            mask_roi = np.zeros_like(mask)
            cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
            mean_color = cv2.mean(original_img, mask=mask_roi)[:3]
            
            # 转换平均颜色到LAB空间
            mean_lab = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2LAB)[0][0]
            
            # 印章通常有较高的a通道值(红色)
            is_red_seal = mean_lab[1] > 140  
            
            # 综合过滤条件
            if (circularity > 0.4 or rectangularity > 0.6) and is_red_seal:
                # 额外检查区域内颜色的方差（印章通常颜色均匀）
                stddev = cv2.meanStdDev(lab_img, mask=mask_roi)[1]
                color_uniformity = np.mean(stddev[:2])  # 不考虑亮度通道
                
                if color_uniformity < 30:  # 颜色变化不大的区域更可能是印章
                    valid_contours.append(cnt)
        
        if not valid_contours:
            return None
            
        # 3. 按面积合并轮廓（处理可能分离的印章部分）
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # 如果最大的轮廓已经足够大，直接用这个
        if cv2.contourArea(valid_contours[0]) > 500:
            return valid_contours[0]
        
        # 否则尝试合并相近的轮廓
        main_contour = valid_contours[0]
        main_center = np.mean(main_contour, axis=0)[0]
        
        for cnt in valid_contours[1:]:
            cnt_center = np.mean(cnt, axis=0)[0]
            distance = np.linalg.norm(main_center - cnt_center)
            
            # 如果轮廓中心距离小于50像素，且面积相近(1:3范围内)，则合并
            if distance < 50 and (0.33 < cv2.contourArea(cnt)/cv2.contourArea(main_contour) < 3):
                main_contour = np.vstack((main_contour, cnt))
        
        return main_contour
    
    def enhance_seal_color(self, seal_pil, enhance_param):
        """
        加深印章颜色
        :param seal_pil: PIL图像格式的印章
        :param enhance_param: 强度参数(0-10)
        :return: 处理后的PIL图像
        """
        # 确保输入是RGBA格式
        if seal_pil.mode != 'RGBA':
            seal_pil = seal_pil.convert('RGBA')
        
        # 将PIL图像转为NumPy数组
        seal_np = np.array(seal_pil)
        
        # 计算增强系数 (1.0-3.0之间，基于enhance_param)
        enhance_factor = 1.0 + (enhance_param / 5.0)  # 参数0时1.0，参数10时3.0
        
        # 分离RGB和Alpha通道
        rgb = seal_np[..., :3].astype(np.float32)
        alpha = seal_np[..., 3] / 255.0  # 归一化到0-1
        
        # 转换到HSV色彩空间进行颜色操作
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 增强饱和度(S通道)
        hsv[..., 1] = np.clip(hsv[..., 1] * enhance_factor, 0, 255)
        
        # 对于红色区域(H通道在0-10或160-180)，增强值(V通道)
        red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160))
        hsv[..., 2] = np.where(red_mask, 
                            np.clip(hsv[..., 2] * (enhance_factor*0.9), 0, 255),  # 红色稍微保守一点
                            np.clip(hsv[..., 2] * (enhance_factor*0.7), 0, 255))
        
        # 转换回RGB
        enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 恢复透明度
        enhanced_rgba = np.dstack((enhanced_rgb, (alpha * 255).astype(np.uint8)))
        
        # 转换回PIL图像
        enhanced_pil = Image.fromarray(enhanced_rgba, 'RGBA')
        
        return enhanced_pil


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


    
