import os
import json
import argparse
import openai
import faiss
import numpy as np
from typing import List, Dict, Any, Union
from datetime import datetime
import tiktoken
import glob
import re
import torch
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ----------------------------
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ColPali imports
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
from typing import cast

# ----------------------------
# AI_2025S_Team 8_Colpali Version
# System Main Code & Extension implementation by D13944024, 蔡宜淀
# Modifying by D13949002, 邱翊

# 配置 OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 嵌入模型與維度
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536
MAX_TOKENS = 7000
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500

# ColPali 設定
COLPALI_MODEL_NAME = "vidore/colpali-v1.2"

# ========= 1. ColPali 圖片處理類別 =========

class ColPaliImageProcessor:
    def __init__(self, model_name: str = COLPALI_MODEL_NAME):
        """初始化 ColPali 圖片處理器"""
        self.device = get_torch_device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.ocr_engine = None
        self.ocr_type = None
        self._load_model()
        self._init_ocr()
        
    def _load_model(self):
        """載入 ColPali 模型"""
        print(f"載入 ColPali 模型: {self.model_name}")
        try:
            self.model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.model_name))
            print("✅ ColPali 模型載入完成")
        except Exception as e:
            print(f"❌ 載入 ColPali 模型失敗: {e}")
            raise
    
    def _init_ocr(self):
        """初始化 OCR 引擎（按優先順序嘗試）"""
        # 嘗試載入不同的 OCR 引擎，按優先順序
        ocr_engines = [
            ("PaddleOCR", self._init_paddleocr),
            ("EasyOCR", self._init_easyocr),
            ("Tesseract", self._init_tesseract)
        ]
        
        for engine_name, init_func in ocr_engines:
            try:
                init_func()
                print(f"✅ {engine_name} OCR 引擎初始化成功")
                # 確保 ocr_type 被設定
                if self.ocr_type is None:
                    self.ocr_type = engine_name
                return
            except Exception as e:
                print(f"⚠️  {engine_name} 初始化失敗: {e}")
                continue
        
        print("❌ 所有 OCR 引擎初始化失敗")
        self.ocr_engine = None
        self.ocr_type = None

    

    def _init_paddleocr(self):
        """初始化 PaddleOCR"""
        from paddleocr import PaddleOCR
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='ch',  # 支援中文
            show_log=False,
            use_gpu=False
        )
        self.ocr_type = "PaddleOCR"
    
    def _init_easyocr(self):
        """初始化 EasyOCR（改善版）"""
        import easyocr
        try:
            # 使用更好的語言組合和設定
            self.ocr_engine = easyocr.Reader(
                ['ch_tra', 'en'],  # 繁體中文 + 英文
                gpu=False,
                verbose=False,
                model_storage_directory=None,  # 使用預設模型路徑
                download_enabled=True
            )
            self.ocr_type = "EasyOCR"
            print("✅ EasyOCR 初始化成功 (繁體中文+英文)")
        except Exception as e:
            print(f"⚠️  繁體中文模式失敗: {e}")
            try:
                # 降級到簡體中文
                self.ocr_engine = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
                self.ocr_type = "EasyOCR"
                print("✅ EasyOCR 初始化成功 (簡體中文+英文)")
            except Exception as e2:
                # 最後降級到純英文
                self.ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.ocr_type = "EasyOCR"
                print("✅ EasyOCR 初始化成功 (純英文)")

    
    def _init_tesseract(self):
        """初始化 Tesseract"""
        import pytesseract
        # 測試是否可用
        pytesseract.get_tesseract_version()
        self.ocr_engine = pytesseract
        self.ocr_type = "Tesseract"
    
    def extract_text_from_image(self, image_path: str, queries: List[str] = None) -> str:
        """從圖片中提取文字內容"""
        try:
            print(f"🔍 開始處理圖片: {image_path}")
            
            # 載入圖片
            if isinstance(image_path, str):
                image = Image.open(image_path)
                print(f"📷 圖片尺寸: {image.size}")
            else:
                image = image_path
                image_path = "temp_image"  # 臨時名稱
            
            # 1. 使用 ColPali 進行文件理解和檢索
            print("🧠 使用 ColPali 進行文件理解...")
            colpali_confidence = self._analyze_with_colpali(image, queries)
            
            # 2. 使用 OCR 進行文字提取
            if self.ocr_engine:
                print(f"🔤 使用 {self.ocr_type} 進行文字提取...")
                extracted_text = self._extract_with_ocr(image_path, image)
            else:
                print("⚠️  沒有可用的 OCR 引擎，嘗試使用備用方法...")
                extracted_text = self._fallback_text_extraction(image_path)
            
            if extracted_text and len(extracted_text.strip()) > 0:
                print(f"✅ 成功提取文字內容 ({len(extracted_text)} 字元)")
                print("📝 提取的文字內容:")
                print("-" * 50)
                print(extracted_text)
                print("-" * 50)
                
                # 3. 結合 ColPali 的理解和 OCR 的文字提取
                enhanced_text = self._enhance_text_with_colpali(extracted_text, colpali_confidence)
                
                return enhanced_text
            else:
                raise ValueError("無法從圖片中提取文字內容")
                
        except Exception as e:
            print(f"❌ 圖片文字提取失敗: {e}")
            raise ValueError(f"圖片處理失敗: {e}")
    
    def _analyze_with_colpali(self, image: Image.Image, queries: List[str] = None) -> Dict:
        """使用 ColPali 分析圖片"""
        try:
            if queries is None:
                queries = [
                    "這是什麼類型的廣告？",
                    "圖片中有什麼產品信息？",
                    "有什麼健康或美容聲明？",
                    "圖片的主要內容是什麼？",
                    "這個廣告在宣傳什麼功效？"
                ]
            
            print(f"🔤 使用 {len(queries)} 個查詢來分析文件...")
            
            # 處理查詢
            query_loader = DataLoader(
                dataset=ListDataset[str](queries),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_queries(x),
            )
            
            # 生成查詢嵌入
            query_embeddings = []
            for batch_query in query_loader:
                with torch.no_grad():
                    batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                    embeddings_query = self.model(**batch_query)
                query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))
            
            # 處理圖片
            image_loader = DataLoader(
                dataset=ListDataset[Image.Image]([image]),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_images(x),
            )
            
            # 生成圖片嵌入
            image_embeddings = []
            for batch_doc in image_loader:
                with torch.no_grad():
                    batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                    embeddings_doc = self.model(**batch_doc)
                image_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
            
            # 計算相似度
            scores = []
            best_score = 0
            for query_emb in query_embeddings:
                for img_emb in image_embeddings:
                    score = torch.einsum("bd,cd->bc", query_emb, img_emb).max()
                    score_value = score.item()
                    scores.append(score_value)
                    best_score = max(best_score, score_value)
            
            print(f"📊 ColPali 分析完成，最高相似度分數: {best_score:.4f}")
            
            return {
                "confidence": best_score,
                "document_type": "advertisement",
                "has_text": True,
                "scores": scores
            }
            
        except Exception as e:
            print(f"⚠️  ColPali 分析失敗: {e}")
            return {"confidence": 0.5, "document_type": "unknown", "has_text": True}
    
    def _preprocess_image(self, image_path: str) -> str:
        """圖片預處理以改善 OCR 效果"""
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import cv2
            import numpy as np
            
            # 載入圖片
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
            
            # 轉換為 OpenCV 格式
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 1. 調整大小（如果圖片太大）
            height, width = img_cv.shape[:2]
            if width > 1500 or height > 1500:
                scale = min(1500/width, 1500/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 2. 轉換為灰度
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 3. 降噪
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 4. 增強對比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 5. 二值化
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 儲存預處理後的圖片
            processed_path = "temp_processed_image.png"
            cv2.imwrite(processed_path, binary)
            
            return processed_path
            
        except Exception as e:
            print(f"⚠️  圖片預處理失敗: {e}")
            return image_path  # 返回原始路徑

    
    def _extract_with_ocr(self, image_path: str, image: Image.Image = None) -> str:
        """使用 OCR 提取文字（改善版）"""
        if not self.ocr_engine:
            raise ValueError("沒有可用的 OCR 引擎")
        
        try:
            # 圖片預處理
            processed_path = self._preprocess_image(image_path)
            
            if self.ocr_type == "EasyOCR":
                # 使用更好的參數設定
                results = self.ocr_engine.readtext(
                    processed_path,
                    detail=1,  # 返回詳細信息
                    paragraph=False,  # 不合併段落
                    width_ths=0.7,  # 文字寬度閾值
                    height_ths=0.7,  # 文字高度閾值
                    text_threshold=0.7,  # 文字檢測閾值
                    link_threshold=0.4,  # 文字連接閾值
                    low_text=0.4  # 低文字分數閾值
                )
                
                # 過濾低信心度的結果
                filtered_results = [item for item in results if item[2] > 0.5]
                text_parts = [item[1] for item in filtered_results]
                final_text = '\n'.join(text_parts)
                
            elif self.ocr_type == "PaddleOCR":
                result = self.ocr_engine.ocr(processed_path, cls=True)
                if result[0] is None:
                    return "未檢測到文字"
                
                # 過濾低信心度的結果
                filtered_results = [line for line in result[0] if line[1][1] > 0.5]
                text_parts = [line[1][0] for line in filtered_results]
                final_text = '\n'.join(text_parts)
                
            elif self.ocr_type == "Tesseract":
                from PIL import Image
                img = Image.open(processed_path)
                final_text = pytesseract.image_to_string(
                    img, 
                    lang='chi_tra+eng',
                    config='--psm 6'  # 統一文字塊模式
                )
            
            # 清理臨時檔案
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)
            
            return final_text.strip()
            
        except Exception as e:
            print(f"❌ {self.ocr_type} 文字提取失敗: {e}")
            raise

    
    def _enhance_text_with_colpali(self, ocr_text: str, colpali_info: Dict) -> str:
        """結合 ColPali 理解增強 OCR 文字"""
        # 根據 ColPali 的理解來改善 OCR 結果
        enhanced_text = ocr_text
        
        # 簡單的文字清理
        lines = enhanced_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 1:  # 過濾太短的行
                cleaned_lines.append(line)
        
        enhanced_text = "\n".join(cleaned_lines)
        
        # 根據 ColPali 的信心分數提供額外信息
        confidence = colpali_info.get('confidence', 0)
        if confidence > 0.8:
            print(f"🎯 ColPali 高信心分數 ({confidence:.3f})，文字提取品質通過")
        elif confidence > 0.5:
            print(f"⚠️  ColPali 中等信心分數 ({confidence:.3f})，建議檢查文字提取結果")
        else:
            print(f"❌ ColPali 低信心分數 ({confidence:.3f})，文字提取可能不完整")
        
        return enhanced_text
    
    def _fallback_text_extraction(self, image_path: str) -> str:
        """備用文字提取方法"""
        try:
            print("🔄 嘗試使用備用文字提取方法...")
            
            # 嘗試使用 pytesseract OCR
            try:
                import pytesseract
                if isinstance(image_path, str) and os.path.exists(image_path):
                    image = Image.open(image_path)
                else:
                    raise ValueError("無效的圖片路徑")
                
                text = pytesseract.image_to_string(image, lang='chi_tra+eng')
                if text and len(text.strip()) > 0:
                    print(f"✅ 備用 OCR 提取成功: {len(text)} 字元")
                    print("📝 備用 OCR 提取的文字內容:")
                    print("-" * 50)
                    print(text)
                    print("-" * 50)
                    return text
            except ImportError:
                print("⚠️  pytesseract 未安裝，無法使用備用 OCR 方案")
            except Exception as e:
                print(f"⚠️  備用 OCR 提取失敗: {e}")
            
            # 如果所有方法都失敗，返回錯誤信息
            error_msg = f"""
❌ 無法從圖片 {image_path} 中提取文字內容。

📋 建議解決方案：
1. 安裝 PaddleOCR（推薦）: pip install paddlepaddle paddleocr
2. 安裝 EasyOCR: pip install easyocr  
3. 安裝 Tesseract: pip install pytesseract
4. 確保圖片質量清晰，文字可讀
5. 手動輸入文字內容進行檢測

💡 安裝任一 OCR 套件後重新運行即可。
"""
            print(error_msg)
            return ""
            
        except Exception as e:
            print(f"❌ 備用文字提取也失敗: {e}")
            return ""

# ========= 安裝指南函數 =========

def print_ocr_installation_guide():
    """打印 OCR 安裝指南"""
    guide = """
📦 OCR 套件安裝指南
==================

🥇 方案 1：PaddleOCR（推薦，中文效果最好）
pip install paddlepaddle paddleocr

🥈 方案 2：EasyOCR（簡單易用，支援多語言）
pip install easyocr

🥉 方案 3：Tesseract（需要額外安裝系統套件）

Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-chi-sim
pip install pytesseract

macOS:
brew install tesseract tesseract-lang
pip install pytesseract

Windows:
1. 下載並安裝 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
2. pip install pytesseract

✅ 安裝任一套件後，系統將自動使用該 OCR 引擎進行文字提取。
"""
    print(guide)

# ========= 測試函數 =========

def test_colpali_with_sample():
    """測試 ColPali 功能"""
    try:
        print("🧪 測試 ColPali 圖片處理功能...")
        processor = ColPaliImageProcessor()
        
        # 創建一個測試圖片（如果沒有真實圖片）
        from PIL import Image, ImageDraw, ImageFont
        
        # 創建測試圖片
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 添加測試文字
        test_text = "本產品能有效改善體質\n促進新陳代謝\n健康維持"
        try:
            # 嘗試使用系統字體
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # 如果沒有字體，使用預設字體
            font = ImageFont.load_default()
        
        draw.text((50, 50), test_text, fill='black', font=font)
        
        # 儲存測試圖片
        test_image_path = "test_advertisement.png"
        img.save(test_image_path)
        print(f"✅ 創建測試圖片: {test_image_path}")
        
        # 測試文字提取
        extracted_text = processor.extract_text_from_image(test_image_path)
        print(f"📝 提取結果: {extracted_text}")
        
        # 清理測試檔案
        os.remove(test_image_path)
        print("🧹 清理測試檔案完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

# ========= 2. 通用工具函數 =========
def _extract_with_multiple_ocr(self, image_path: str, image: Image.Image = None) -> str:
    """使用多個 OCR 引擎並選擇最佳結果"""
    results = []
    
    # 1. 預處理圖片
    processed_path = self._preprocess_image(image_path)
    
    # 2. 嘗試不同的 OCR 引擎
    ocr_engines = []
    
    # EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(['ch_tra', 'en'], gpu=False, verbose=False)
        result = reader.readtext(processed_path)
        text = '\n'.join([item[1] for item in result if item[2] > 0.3])  # 信心度過濾
        results.append(('EasyOCR', text, len(text)))
        print(f"📝 EasyOCR 結果: {len(text)} 字元")
    except Exception as e:
        print(f"⚠️  EasyOCR 失敗: {e}")
    
    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        result = paddle_ocr.ocr(processed_path, cls=True)
        if result[0]:
            text = '\n'.join([line[1][0] for line in result[0] if line[1][1] > 0.3])
            results.append(('PaddleOCR', text, len(text)))
            print(f"📝 PaddleOCR 結果: {len(text)} 字元")
    except Exception as e:
        print(f"⚠️  PaddleOCR 失敗: {e}")
    
    # Tesseract
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(processed_path)
        text = pytesseract.image_to_string(img, lang='chi_tra+eng')
        results.append(('Tesseract', text, len(text.strip())))
        print(f"📝 Tesseract 結果: {len(text.strip())} 字元")
    except Exception as e:
        print(f"⚠️  Tesseract 失敗: {e}")
    
    # 清理臨時檔案
    if processed_path != image_path and os.path.exists(processed_path):
        os.remove(processed_path)
    
    # 3. 選擇最佳結果（基於文字長度和品質）
    if results:
        # 按文字長度排序，選擇最長的結果
        best_result = max(results, key=lambda x: x[2])
        print(f"✅ 選擇 {best_result[0]} 的結果 ({best_result[2]} 字元)")
        return best_result[1]
    else:
        return "無法提取文字內容"


def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """計算文本的 token 數量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 3

def smart_text_chunker(text: str, max_tokens: int = CHUNK_SIZE, overlap_tokens: int = CHUNK_OVERLAP) -> List[str]:
    """智能文本切割器"""
    if count_tokens(text) <= max_tokens:
        return [text]

    chunks, current_chunk, current_tokens = [], "", 0
    sentences = re.split(r"[。！？；\n]", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        tok = count_tokens(sentence)
        
        # 如果單句過長，強制切割
        if tok > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk, current_tokens = "", 0
            
            # 按 token 強制切割
            try:
                encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
                enc = encoding.encode(sentence)
                for i in range(0, len(enc), max_tokens - overlap_tokens):
                    chunk_tokens = enc[i:i + max_tokens]
                    chunks.append(encoding.decode(chunk_tokens))
            except Exception:
                # 如果編碼失敗，按字符切割
                chunk_size = max_tokens * 3  # 估算字符數
                for i in range(0, len(sentence), chunk_size):
                    chunks.append(sentence[i:i + chunk_size])
            continue
        
        # 一般累積邏輯
        if current_tokens + tok > max_tokens:
            chunks.append(current_chunk.strip())
            
            # 處理重疊
            if overlap_tokens > 0:
                overlap_text = current_chunk[-overlap_tokens*3:]
                current_chunk = overlap_text + sentence
                current_tokens = count_tokens(current_chunk)
            else:
                current_chunk, current_tokens = sentence, tok
        else:
            current_chunk += sentence + "。"
            current_tokens += tok
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def translate_to_cht(text: str) -> str:
    """文字翻譯（非中文 → 繁中）"""
    try:
        # 簡單的語言檢測
        if any(char in text for char in '的是在有一個'):
            return text  # 已經是中文
    except Exception:
        pass

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a translator; translate to Traditional Chinese without adding or removing content."},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"翻譯失敗: {e}")
        return text

# ========= 3. JSON 萃取處理器 =========

class UniversalJSONProcessor:
    """通用 JSON 處理器"""
    
    @staticmethod
    def extract_text_from_any_structure(data: Any, path: str = "") -> List[str]:
        """從任意 JSON 結構中提取文本"""
        texts = []
        
        if isinstance(data, str):
            if len(data.strip()) > 5:
                texts.append(data.strip())
        elif isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in ['id', 'timestamp', 'created_at', 'updated_at']:
                    continue
                texts.extend(UniversalJSONProcessor.extract_text_from_any_structure(value, f"{path}.{key}"))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                texts.extend(UniversalJSONProcessor.extract_text_from_any_structure(item, f"{path}[{i}]"))
        
        return texts
    
    @staticmethod
    def identify_content_type(data: Dict, filename: str) -> str:
        """識別內容類型"""
        filename_lower = filename.lower()
        
        # 基於檔名判斷
        if '違規' in filename or 'violation' in filename_lower:
            return 'violation'
        elif '可用' in filename or 'allowed' in filename_lower or '得使用' in filename:
            return 'allowed'
        elif '法規' in filename or 'law' in filename_lower or '管理法' in filename:
            return 'regulation'
        elif '保健功效' in filename or 'health' in filename_lower:
            return 'health_function'
        elif '化妝品' in filename or 'cosmetic' in filename_lower:
            return 'cosmetic'
        elif '中藥' in filename or 'tcm' in filename_lower:
            return 'tcm'
        
        # 基於內容結構判斷
        data_str = str(data).lower()
        if 'inappropriate' in data_str or '不適當' in str(data):
            return 'violation'
        elif 'allowed' in data_str or '允許' in str(data) or '可用' in str(data):
            return 'allowed'
        else:
            return 'general'

# ========= 4. 增強版向量資料庫 =========

class EnhancedVectorStore:
    def __init__(self, index_path: str = "legal_docs.idx", data_path: str = "legal_docs.json"):
        self.index_path = index_path
        self.data_path = data_path
        self.index = None
        self.metadatas = []
        # -------------------------------------------
        self.sent_transformer = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # -------------------------------------------
        
        # 分類儲存不同類型的內容
        self.violation_patterns = set()
        self.allowed_phrases = set()
        self.regulation_texts = []
        self.health_functions = []
        
        # 基礎允許用詞
        self.base_allowed_claims = {
            "完整補充營養", "調整體質", "促進新陳代謝", "幫助入睡",
            "保護消化道全機能", "改變細菌叢生態", "排便有感",
            "維持正常的排便習慣", "排便順暢", "提升吸收滋養消化機能",
            "青春美麗", "營養補充", "膳食補充", "健康維持",
            "能完整補充人體營養", "提升生理機能", "調節生理機能",
            "排便超有感", "給你排便順暢新體驗",
            "在嚴謹的營養均衡與熱量控制，以及適當的運動條件下，適量攝取本產品有助於不易形成體脂肪"
        }
        
        # 嚴格違規關鍵詞
        self.strict_violation_keywords = {
            "治療", "治癒", "醫治", "根治", "療效", "藥效", "消除疾病",
            "治好", "痊癒", "根除", "醫療", "診斷", "預防疾病", "抗癌",
            "降血糖", "降血壓", "治糖尿病", "治高血壓",
            "替代方案", "最佳替代", "治療替代", "醫療替代",
            "替代療法", "替代治療", "取代藥物", "不用吃藥",
            "胃食道逆流", "糖尿病", "高血壓", "心臟病", "癌症",
            "肝病", "腎病", "關節炎", "憂鬱症", "失眠症",
            "100%有效", "完全治癒", "永久根治", "立即見效",
            "保證有效", "絕對有效", "神奇療效",
            "活化毛囊", "刺激毛囊細胞", "增加毛囊角質細胞增生",
            "刺激毛囊讓髮絲再次生長不易落脫", "刺激毛囊不萎縮",
            "堅固毛囊刺激新生秀髮", "頭頂不再光禿禿", "頭頂不再光溜溜",
            "避免稀疏", "避免髮量稀少問題", "有效預防落髮/抑制落髮/減少落髮",
            "有效預防掉髮/抑制掉髮/減少掉髮",
            "增強(增加)自體免疫力", "增強淋巴引流", "促進細胞活動", "深入細胞膜作用",
            "減弱角化細胞", "刺激細胞呼吸作用", "提高肌膚細胞帶氧率",
            "進入甲母細胞和甲床深度滋潤", "刺激增長新的健康細胞", "增加細胞新陳代謝",
            "促進肌膚神經醯胺合成", "維持上皮組織機能的運作", "重建皮脂膜", "重建角質層",
            "促進(刺激)膠原蛋白合成", "促進(刺激)膠原蛋白增生", "瘦身", "減肥", "去脂",
            "減脂", "消脂", "燃燒脂肪", "減緩臀部肥油囤積", "預防脂肪細胞堆積",
            "刺激脂肪分解酵素", "纖(孅)體", "塑身", "雕塑曲線", "消除掰掰肉", "消除蝴蝶袖",
            "告別小腹婆", "減少橘皮組織", "豐胸", "隆乳", "使胸部堅挺不下垂",
            "感受托高集中的驚人效果", "漂白", "使乳暈漂成粉紅色", "不過敏", "零過敏",
            "減過敏", "抗過敏", "舒緩過敏", "修護過敏", "過敏測試", "醫藥級", "鎮靜劑",
            "鎮定劑", "消除浮腫", "改善微血管循環", "功能強化微血管", "增加血管含氧量提高肌膚帶氧率",
            "預防(防止)肥胖紋", "預防(防止)妊娠紋", "緩減妊娠紋生產"
        }
        
        # 上下文敏感模式
        self.context_sensitive_patterns = {
            "專家推薦": ["治療", "疾病", "替代", "醫療"],
            "科學實證": ["治療", "療效", "醫療"],
            "國外研究": ["治療", "療效", "醫療"],
            "臨床證實": ["治療", "療效", "醫療"]
        }
        
        # 產品類別關鍵字
        self.product_type_map = {
            "food": ["食品", "supplement", "膳食", "代餐"],
            "cosmetic": ["化妝品", "美白", "skin", "cream", "toner"],
            "drug": ["藥", "治療", "藥品", "藥物"]
        }

    def _cluster_representative_texts(self, texts: List[str], max_clusters: int = 5) -> List[str]:
        if len(texts) <= max_clusters:
            return texts
    
        embeddings = self.sent_transformer.encode(texts)
        n_clusters = min(max_clusters, len(texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    
        representatives = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            centroid = centroids[cluster_id]
            distances = [np.linalg.norm(e - centroid) for e in cluster_embeddings]
            rep_idx = cluster_indices[np.argmin(distances)]
            group_text = "。".join([texts[i] for i in cluster_indices])
            representatives.append(group_text)
    
        return representatives
    
    def identify_product_type(self, text: str) -> str:
        """識別產品類型"""
        text_lower = text.lower()
        for product_type, keywords in self.product_type_map.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return product_type
        return "food"
    
    def load_json_files(self, json_folder: str) -> List[Dict[str, Any]]:
        """載入 JSON 文件"""
        all_data = []
        json_files = glob.glob(os.path.join(json_folder, "*.json"))
        
        if not json_files:
            raise ValueError(f"資料夾中沒有 JSON 文件: {json_folder}")
        
        print(f"發現 {len(json_files)} 個 JSON 文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                processed_data = self.process_universal_json(data, os.path.basename(json_file))
                all_data.extend(processed_data)
                print(f"已處理: {os.path.basename(json_file)} - {len(processed_data)} 筆資料")
                
            except Exception as e:
                print(f"處理 {json_file} 時發生錯誤: {e}")
                continue
        
        return all_data
    
    def process_universal_json(self, data: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
        """處理通用 JSON 數據"""
        processed_items = []
        content_type = UniversalJSONProcessor.identify_content_type(data, filename)
        
        # 提取所有文本內容
        all_texts = UniversalJSONProcessor.extract_text_from_any_structure(data)
        
        for i, text in enumerate(all_texts):
            if len(text.strip()) < 10:
                continue
            
            token_count = count_tokens(text)
            
            if token_count > MAX_TOKENS:
                print(f"文本過長 ({token_count} tokens)，進行切割...")
                chunks = smart_text_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
                print(f"切割為 {len(chunks)} 個片段")
                
                for chunk_idx, chunk in enumerate(chunks):
                    item_id = f"{filename}_item_{i}_chunk_{chunk_idx}"
                    
                    # 分類處理
                    self._categorize_content(chunk, content_type)
                    
                    processed_items.append({
                        'id': item_id,
                        'text': chunk,
                        'source_file': filename,
                        'content_type': content_type,
                        'type': f'{content_type}_content',
                        'chunk_info': {
                            'original_item': i,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'original_length': token_count
                        }
                    })
            else:
                item_id = f"{filename}_item_{i}"
                
                # 分類處理
                self._categorize_content(text, content_type)
                
                processed_items.append({
                    'id': item_id,
                    'text': text,
                    'source_file': filename,
                    'content_type': content_type,
                    'type': f'{content_type}_content'
                })
        
        # 處理結構化數據
        if isinstance(data, dict):
            processed_items.extend(self._process_structured_data(data, filename, content_type))
        
        return processed_items
    
    def _categorize_content(self, text: str, content_type: str):
        """分類內容到對應的集合"""
        text_stripped = text.strip()
        
        if content_type == 'violation':
            if len(text_stripped) < 100:
                self.violation_patterns.add(text_stripped)
        elif content_type == 'allowed':
            if len(text_stripped) < 50:
                self.allowed_phrases.add(text_stripped)
        elif content_type == 'regulation':
            self.regulation_texts.append(text_stripped)
        elif content_type == 'health_function':
            self.health_functions.append(text_stripped)
    
    def _process_structured_data(self, data: Dict, filename: str, content_type: str) -> List[Dict]:
        """處理結構化數據"""
        items = []
        
        patterns = [
            ('cases', 'case'),
            ('items', 'item'), 
            ('examples', 'example'),
            ('categories', 'category'),
            ('violations', 'violation'),
            ('regulations', 'regulation')
        ]
        
        for pattern_key, item_type in patterns:
            if pattern_key in data and isinstance(data[pattern_key], list):
                for i, item in enumerate(data[pattern_key]):
                    if isinstance(item, dict):
                        content_fields = ['content', 'text', 'description', '內容', '廣告內容', 'ad_content']
                        main_content = ""
                        
                        for field in content_fields:
                            if field in item and item[field]:
                                main_content = str(item[field])
                                break
                        
                        if not main_content:
                            main_content = json.dumps(item, ensure_ascii=False)
                        
                        token_count = count_tokens(main_content)
                        
                        if token_count > MAX_TOKENS:
                            chunks = smart_text_chunker(main_content, CHUNK_SIZE, CHUNK_OVERLAP)
                            
                            for chunk_idx, chunk in enumerate(chunks):
                                self._categorize_content(chunk, content_type)
                                
                                items.append({
                                    'id': f"{filename}_{item_type}_{i}_chunk_{chunk_idx}",
                                    'text': chunk,
                                    'source_file': filename,
                                    'content_type': content_type,
                                    'item_type': item_type,
                                    'type': f'structured_{item_type}',
                                    'chunk_info': {
                                        'original_item': i,
                                        'chunk_index': chunk_idx,
                                        'total_chunks': len(chunks),
                                        'original_length': token_count
                                    }
                                })
                        else:
                            self._categorize_content(main_content, content_type)
                            
                            items.append({
                                'id': f"{filename}_{item_type}_{i}",
                                'text': main_content,
                                'source_file': filename,
                                'content_type': content_type,
                                'item_type': item_type,
                                'type': f'structured_{item_type}'
                            })
        
        return items
    
    def build(self, json_folder: str):
        """建立向量索引"""
        if not os.path.isdir(json_folder):
            raise FileNotFoundError(f"找不到資料夾: {json_folder}")
        
        # 載入並處理 JSON 文件
        all_data = self.load_json_files(json_folder)
        
        if not all_data:
            raise ValueError("沒有成功處理任何 JSON 文件")
        
        # 合併基礎允許用詞
        self.allowed_phrases.update(self.base_allowed_claims)
        
        print(f"建立索引：共處理 {len(all_data)} 個資料項目")
        print(f"違規模式: {len(self.violation_patterns)} 個")
        print(f"允許用詞: {len(self.allowed_phrases)} 個")
        print(f"法規條文: {len(self.regulation_texts)} 個")
        print(f"保健功效: {len(self.health_functions)} 個")
        
        self.metadatas = []
        embeddings = []
        
        for item in all_data:
            try:
                text = item['text']
                token_count = count_tokens(text)
                
                if token_count > MAX_TOKENS:
                    print(f"警告：項目 {item['id']} 仍然過長 ({token_count} tokens)，跳過")
                    continue
                
                # 生成嵌入向量
                emb = openai.Embedding.create(
                    model=EMBEDDING_MODEL,
                    input=text
                )['data'][0]['embedding']
                
                embeddings.append(emb)
                self.metadatas.append(item)
                
                if len(embeddings) % 50 == 0:
                    print(f"已處理 {len(embeddings)} 個項目...")
                
            except Exception as e:
                print(f"處理項目 {item.get('id', 'unknown')} 時發生錯誤: {e}")
                continue
        
        if not embeddings:
            raise ValueError("沒有成功處理任何項目")
        
        # 建立 FAISS 索引
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(np.array(embeddings, dtype='float32'))
        self.index = index
        
        # 儲存索引和元數據
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        
        print(f"索引與元資料已儲存。總共處理了 {len(embeddings)} 個項目。")
        
        chunked_items = [item for item in self.metadatas if 'chunk_info' in item]
        if chunked_items:
            print(f"其中 {len(chunked_items)} 個項目來自文本切割")
    
    def load(self):
        """載入現有的索引和元數據"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.data_path):
            raise FileNotFoundError("索引或元資料檔案不存在，請先建立索引（使用 --rebuild）。")
        
        self.index = faiss.read_index(self.index_path)
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.metadatas = json.load(f)
        
        # 重建分類數據
        self.allowed_phrases.update(self.base_allowed_claims)
        
        for item in self.metadatas:
            content_type = item.get('content_type', 'general')
            text = item.get('text', '').strip()
            
            if content_type == 'violation' and text and len(text) < 100:
                self.violation_patterns.add(text)
            elif content_type == 'allowed' and text and len(text) < 50:
                self.allowed_phrases.add(text)
            elif content_type == 'regulation' and text:
                self.regulation_texts.append(text)
            elif content_type == 'health_function' and text:
                self.health_functions.append(text)
        
        print(f"已載入 {len(self.metadatas)} 筆法規元資料。")
        print(f"違規模式: {len(self.violation_patterns)} 個")
        print(f"允許用詞: {len(self.allowed_phrases)} 個")
        
        chunked_items = [item for item in self.metadatas if 'chunk_info' in item]
        if chunked_items:
            print(f"其中 {len(chunked_items)} 個項目來自文本切割")
    
    def query(self, query_text: str, top_k: int = 5, product_type: str = "food") -> List[dict]:
        """查詢最相關的法規文件"""
        if self.index is None:
            raise ValueError("索引未載入，請先建立或載入索引")
        
        # 生成查詢嵌入向量
        q_emb = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=query_text
        )['data'][0]['embedding']
        
        # 搜索最相似的文件
        D, I = self.index.search(np.array([q_emb], dtype='float32'), top_k * 2)
        
        # 根據產品類型過濾結果
        results = []
        for i in I[0]:
            if i < len(self.metadatas):
                meta = self.metadatas[i]
                # 簡化過濾邏輯，主要基於內容相關性
                results.append(meta)
                if len(results) >= top_k:
                    break
        
        return results[:top_k]
    
    def enhanced_compliance_check(self, ad_text: str) -> Dict[str, Any]:
        """增強的合規檢查"""
        ad_text_lower = ad_text.lower()
        
        # 檢查嚴格違規關鍵詞
        strict_violations = [kw for kw in self.strict_violation_keywords if kw in ad_text]
        
        # 檢查上下文敏感組合
        context_violations = []
        for trigger, sensitive_words in self.context_sensitive_patterns.items():
            if trigger in ad_text:
                for word in sensitive_words:
                    if word in ad_text:
                        context_violations.append(f"{trigger}+{word}")
        
        # 檢查醫療替代方案
        medical_substitute_patterns = [
            "替代方案.*疾病", "替代方案.*症", "最佳替代.*疾病", 
            "專家推薦.*治療", "專家推薦.*疾病", "科學實證.*療效"
        ]
        
        substitute_violations = []
        for pattern in medical_substitute_patterns:
            if re.search(pattern, ad_text):
                substitute_violations.append(pattern)
        
        # 檢查允許用詞
        found_allowed = [phrase for phrase in self.allowed_phrases if phrase in ad_text]
        
        # 檢查官方認證
        has_official_approval = bool(re.search(r'衛福部.*字號|衛福部認證|衛署.*字第.*號', ad_text))
        
        # 計算風險分數
        risk_score = 0
        
        if strict_violations:
            risk_score += len(strict_violations) * 50
        
        if context_violations:
            risk_score += len(context_violations) * 40
        
        if substitute_violations:
            risk_score += len(substitute_violations) * 60
        
        if (strict_violations or context_violations or substitute_violations) and not has_official_approval:
            risk_score += 20
        
        if found_allowed and risk_score < 70:
            risk_score = max(0, risk_score - len(found_allowed) * 5)
        
        risk_score = min(95, max(0, risk_score))
        
        return {
            'strict_violations': strict_violations,
            'context_violations': context_violations,
            'substitute_violations': substitute_violations,
            'allowed_phrases_found': found_allowed,
            'has_official_approval': has_official_approval,
            'risk_score': risk_score,
            'risk_level': 'high' if risk_score >= 80 else 'medium' if risk_score >= 40 else 'low',
            'product_type': self.identify_product_type(ad_text)
        }

# ========= 5. 增強版廣告檢測系統（支援圖片輸入）=========

class EnhancedAdvertisementDetector:
    def __init__(self, vector_store: EnhancedVectorStore):
        """初始化增強版廣告檢測系統"""
        self.vector_store = vector_store
        self.colpali_processor = None
        self._init_colpali()
    
    def _init_colpali(self):
        """初始化 ColPali 處理器"""
        try:
            self.colpali_processor = ColPaliImageProcessor()
            print("✅ ColPali 圖片處理器初始化完成")
        except Exception as e:
            print(f"❌ ColPali 初始化失敗: {e}")
            print("⚠️  將僅支援文字輸入模式")
            self.colpali_processor = None
    
    def process_input(self, input_data: Union[str, Path], input_type: str = "auto") -> str:
        """處理輸入數據（文字或圖片）"""
        if input_type == "auto":
            # 自動檢測輸入類型
            if isinstance(input_data, str):
                if os.path.isfile(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_type = "image"
                else:
                    input_type = "text"
            else:
                input_type = "text"
        
        if input_type == "image":
            return self._process_image_input(input_data)
        else:
            return self._process_text_input(input_data)
    
    def _process_text_input(self, text: str) -> str:
        """處理文字輸入"""
        print(f"📝 處理文字輸入: {text[:100]}...")
        return translate_to_cht(text)
    
    def _process_image_input(self, image_path: str) -> str:
        """處理圖片輸入"""
        if not self.colpali_processor:
            raise ValueError("❌ ColPali 處理器未初始化，無法處理圖片。請檢查 ColPali 安裝或使用文字輸入模式。")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 找不到圖片檔案: {image_path}")
        
        print(f"🖼️ 正在處理圖片: {image_path}")
        
        try:
            extracted_text = self.colpali_processor.extract_text_from_image(image_path)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise ValueError("❌ ColPali 無法從圖片中提取足夠的文字內容。建議：\n1. 確保圖片清晰可讀\n2. 使用專門的 OCR 工具\n3. 手動輸入文字內容")
            
            print(f"✅ 成功從圖片提取文字 ({len(extracted_text)} 字元)")
            translated_text = translate_to_cht(extracted_text)
            
            # 顯示最終處理結果
            print("🔄 翻譯後的文字內容:")
            print("=" * 60)
            print(translated_text)
            print("=" * 60)
            
            return translated_text
            
        except Exception as e:
            print(f"❌ 圖片處理失敗: {e}")
            raise
    
    def detect_advertisement_compliance(self, input_data: Union[str, Path], input_type: str = "auto") -> Dict[str, Any]:
        """檢測廣告合規性"""
        try:
            print("\n" + "="*60)
            print("🔍 開始廣告合規性檢測")
            print("="*60)
            
            # 處理輸入
            processed_text = self.process_input(input_data, input_type)
            
            if not processed_text or len(processed_text.strip()) < 5:
                return {
                    'error': '處理後的文字內容過短，無法進行有效檢測',
                    'input_type': input_type,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            print(f"📋 最終檢測文字 ({len(processed_text)} 字元):")
            print("-" * 40)
            print(processed_text)
            print("-" * 40)
            
            # 使用原有的檢測邏輯
            print("🔬 執行合規性分析...")
            result = strict_determine_legality(processed_text, self.vector_store)
            
            # 添加輸入類型信息
            result['input_type'] = input_type
            result['original_input'] = str(input_data)
            
            if input_type == "image":
                result['image_path'] = str(input_data)
                result['extracted_text'] = processed_text
            
            print("✅ 合規性檢測完成")
            return result
            
        except Exception as e:
            error_msg = f"檢測過程發生錯誤: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'error': error_msg,
                'input_type': input_type,
                'original_input': str(input_data),
                'timestamp': datetime.utcnow().isoformat()
            }

# ========= 6. 主判斷函數 =========

def strict_determine_legality(ad_text: str, vs: EnhancedVectorStore) -> dict:
    """嚴格合法性判斷函數"""
    # 翻譯為繁體中文
    ad_text = translate_to_cht(ad_text)
    
    # 增強合規檢查
    compliance_check = vs.enhanced_compliance_check(ad_text)
    
    # 檢索相關法規
    contexts = vs.query(ad_text, top_k=5, product_type=compliance_check["product_type"])
    
    # 準備 vector store 文件 ID 列表
    vector_store_ids = []
    for c in contexts:
        doc_id = c.get('id', c.get('source_file', 'unknown'))
        if 'chunk_info' in c:
            chunk_info = c['chunk_info']
            doc_id += f"_chunk_{chunk_info['chunk_index']+1}/{chunk_info['total_chunks']}"
        vector_store_ids.append(doc_id)
    
    # 根據智能檢測結果調整判斷邏輯
    risk_level = compliance_check['risk_level']
    risk_score = compliance_check['risk_score']
    allowed_phrases = compliance_check['allowed_phrases_found']
    has_approval = compliance_check['has_official_approval']
    
    # 設定系統提示
    system_prompt = f"""
你是一位專業的政府法規審核員，任務是分析廣告文字是否違反《食品安全衛生管理法》第28條規定，是否違反《健康食品管理法》第14條規定，是否違反《化粧品衛生管理法》第10條規定，是否違反《藥事法》第66、68、69條規定、是否違反《醫療器材管理法》第6、40、41、46條規定。
尤其涉及「誇大療效」、「虛假宣稱」、「未經許可醫療用語」、「涉及醫療效能」、「誇張藥物安全性」。
你將根據提供的廣告內容與法規資料，進行逐步風險評估與違法機率判斷，**回應需符合結構與步驟指引，並僅分析實際存在的文字內容**。

---

## 評估任務目標

1. 理解廣告實際上下文是否為描述商品。
2. 如確定廣告文字為描述商品，從中抽取實際存在的描述或用語。
3. 依據資料庫提供的法規／案例文本比對是否構成違規。
4. 依照智能檢測輸出的風險資訊以及你的判斷，綜合推估違法風險百分比（僅允許：10%、30%、50%、70%、90%）。
5. 回應應包含每個分析步驟的依據與結果。

---

## 檢測步驟（請依序逐步完成）

1. **判斷廣告整段內容是否在描述商品**
   - 在頁面如果出現很多不同種類的商品或分類，就不是在描述商品 (如：連身衣 包巾／包被 童襪/手腳套 童鞋 褲褲/帽子 穿搭配件 禮盒專區 嬰幼兒清潔 濕紙巾 紗布/棉巾 濕巾加熱器 奶瓶蔬果洗潔液/洗碗液)。
   - 就算只有包含成分，但無包含功效、使用方法，只要成分不違法，這個廣告內容就不算描述商品，並將最終判定的違法機率降至最低。
   - 若不是描述商品，則回應資料不足以判斷是否為違法，並將智能判斷風險標註為: low。
   - 若是描述商品，則開始以下步驟。
2. **檢查是否出現允許用詞？**
   - 例如：「完整補充營養」「調整體質」「促進新陳代謝」「幫助入睡」「保護消化道全機能」「改變細菌叢生態」「排便有感」「維持正常的排便習慣」「排便順暢」
    「提升吸收滋養消化機能」「青春美麗」「營養補充」「膳食補充」「健康維持」「能完整補充人體營養」「提升生理機能」「調節生理機能」「排便超有感」
    「給你排便順暢新體驗」「在嚴謹的營養均衡與熱量控制，以及適當的運動條件下，適量攝取本產品有助於不易形成體脂肪」等。
   - 若出現，請列舉並說明其合法性參考。
3. **檢查是否出現違規用詞？**
   - 例如：「全方位」「全面」「超能」「治療」「治癒」「醫治」「根治」「療效」「藥效」「消除疾病」
    「治好」「痊癒」「根除」「醫療」「診斷」「預防疾病」「抗癌」
    「降血糖」「降血壓」「治糖尿病」「治高血壓」「邊睡邊瘦」
    「替代方案」「最佳替代」「治療替代」「醫療替代」
    「替代療法」「替代治療」「取代藥物」「不用吃藥」
    「胃食道逆流」「糖尿病」「高血壓」「心臟病」「癌症」
    「肝病」「腎病」「關節炎」「憂鬱症」「失眠症」
    「100%有效」「完全治癒」「永久根治」「立即見效」
    「保證有效」「絕對有效」「神奇療效」等。
   - 若有，請標明實際句子與違規詞。
   - 但若出現但與產品無關（如頁面分類名稱），請說明並去除智能判斷的風險。
   - 若出現且與描述產品功效有關，儘管有出現允許用詞，仍須提高最終判斷的違法機率。
4. **檢查是否有上下文違規組合？**
   - 如：「替代方案」+「X疾病」、「替代方案」+「X症」、「最佳替代」+「X疾病」、「專家推薦」+「X治療」、「臨床證實」+「療效顯著」、「專業XX」+「推薦產品名」、「通過」+「X測試」、「專業藥師X」+「真心推薦X」、      「阻隔」+「成分」等。
   - 如果上下文的違規組合涉及醫療、治療效果，或是以某人推薦、某實驗結果證明來宣傳商品，都要提高最終判斷的違法機率。
   - 請說明違規語句內容。
5. **確認是否有官方認證（如：衛署字號）？**
   - 有則調降風險等級。
6. **整合智能檢測資訊**:
   - 智能風險等級：{risk_level}
   - 智能風險分數：{risk_score}%
   - 有允許用詞：{allowed_phrases}
   - 有官方認證 (如衛署字號)：{has_approval}
   - 違規詞是否出現在商品內容描述中？還是頁面分類／標籤？
      - 若為頁面分類、導航標題，請合理調降風險
      - 若為商品功能說明、標題宣稱，則需保留或提升風險
   - 允許用詞是否真正出現在產品描述中？還是僅為營養成分敘述？
      - 若僅作為配料或成分出現，而無宣稱作用，則風險不應過度下調
7. **根據以上結果，判斷違法機率（從以下數值中選擇）**
   - 違法機率應以第 1~6 步的整體語境、違法詞是否出現、允許用詞是否出現、智能檢測分數與法規比對結果為準。
   - 智能檢測風險分數僅作為參考，若經語境分析後發現判斷需調整，請合理調高或降低違法機率。
   - 若廣告詞過於誇大，儘管出現允許用詞，請合理調高違法機率。
   - 請確保最終違法機率與整體語義判斷一致，避免不合邏輯的矛盾情況（例如：沒有違規詞卻判為高風險）。
   - 僅能選: 10%、30%、50%、70%、90%

---

## 回應格式（回應必須符合下列三選一結構）

### 低風險 (違法機率 10% 或 30%)
違法機率: 30%
✔ 出現允許用詞: 是（例如：幫助入睡、健康維持）
✔ 出現違規用詞: 否
✔ 出現上下文違規組合: 否
違法內容分析: 廣告中使用合規詞句，且無明顯療效宣稱。根據智能檢測顯示低風險，綜合判定違法可能性低。
罰款額度: 無
參考依據: [vector store 文件 ID]

### 中風險（違法機率 50%）
違法機率: 50%
✔ 出現允許用詞: 否
✔ 出現違規用詞: 是（例如：根治）
✔ 出現上下文違規組合: 否
違法內容分析: 廣告中「[實際存在的具體文字]」可能涉及療效暗示，建議謹慎使用。
違反條款: [適用法規]
罰款額度: [依據法規]
參考依據: [vector store 文件 ID]

### 高風險（違法機率 70% 或 90%）
違法機率: 90%
✔ 出現允許用詞: 否
✔ 出現違規用詞: 是（治療、癌症）
✔ 出現上下文違規組合: 是（臨床證實+療效）
違法內容分析: 廣告中「[實際存在的具體文字]」明確宣稱療效，違反相關規定。
違反條款: [適用法規]
罰款額度: [依據法規]
參考依據: [vector store 文件 ID]
"""
    
    # 組合相關法規文本和vector store信息
    docs_parts = []
    for i, c in enumerate(contexts):
        source_info = f"[文件ID: {vector_store_ids[i]}]"
        docs_parts.append(f"{source_info}\n{c['text'][:500]}")
    
    docs = "\n---\n".join(docs_parts)
    
    # 設定用戶提示
    user_prompt = f"""## Vector Store 法規資料：
{docs}

## Vector Store 文件 ID 列表: 
{', '.join(vector_store_ids)}

## 待檢測廣告文本：
```
{ad_text}
```

## 智能檢測詳細結果：
- 嚴格違規關鍵詞: {compliance_check['strict_violations']}
- 上下文違規: {compliance_check['context_violations']}
- 醫療替代違規: {compliance_check['substitute_violations']}
- 發現允許用詞: {compliance_check['allowed_phrases_found']}
- 有官方認證: {compliance_check['has_official_approval']}
- 風險分數: {compliance_check['risk_score']}%
- 風險等級: {compliance_check['risk_level']}

## 分析要求：
1. **請優先根據廣告文本的實際語義與上下文進行判斷**
2. **只能分析廣告中實際出現的內容，不得推測或虛構**
3. **當發現允許用詞，應從寬認定合法性；但也需確認是否僅為成分列舉或無宣稱功能**
4. **若違規用詞僅出現在商品分類、頁面標題等位置，應視情境合理降低風險判定**
5. **智能檢測結果可作為輔助參考，但不應取代語境判斷**
6. **最終違法機率應基於綜合語境、文字內容、合規用詞、違規語彙與官方認證來合理評估**

請根據以上資料與語境分析結果進行合理判斷，智能檢測可作為參考輔助，但違法機率須與實際文字與上下文相符，不應僅依據風險分數機械化決定。
"""
    
    # 調用 GPT 模型
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        analysis_result = resp.choices[0].message.content.strip()
        
        # 解析結果並結構化返回
        result = {
            'analysis': analysis_result,
            'compliance_check': compliance_check,
            'vector_store_contexts': contexts,
            'vector_store_ids': vector_store_ids,
            'processed_text': ad_text,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"GPT 分析失敗: {e}")
        return {
            'error': f"分析過程發生錯誤: {e}",
            'compliance_check': compliance_check,
            'processed_text': ad_text,
            'timestamp': datetime.utcnow().isoformat()
        }

# ========= 7. 命令行界面 =========

def main():
    parser = argparse.ArgumentParser(description="增強版廣告檢測系統（支援 ColPali 圖片輸入）")
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引")
    parser.add_argument("--json_folder", default="legal_data", help="法規 JSON 資料夾路徑")
    parser.add_argument("--text", help="要檢測的廣告文字")
    parser.add_argument("--image", help="要檢測的廣告圖片路徑")
    parser.add_argument("--batch", help="批量處理的 CSV 檔案路徑")
    parser.add_argument("--output", help="結果輸出檔案路徑")
    
    args = parser.parse_args()
    
    # 初始化向量資料庫
    print("🔧 初始化向量資料庫...")
    vs = EnhancedVectorStore()
    
    if args.rebuild:
        print("🔄 重建向量索引...")
        vs.build(args.json_folder)
        print("✅ 索引重建完成！")
        return
    else:
        try:
            vs.load()
        except FileNotFoundError:
            print("⚠️  找不到現有索引，自動建立新索引...")
            vs.build(args.json_folder)
    
    # 初始化增強版檢測器
    print("🚀 初始化廣告檢測器...")
    detector = EnhancedAdvertisementDetector(vs)
    
    # 處理不同類型的輸入
    results = []
    
    if args.text:
        print("📝 處理文字輸入...")
        result = detector.detect_advertisement_compliance(args.text, "text")
        results.append({
            'input': args.text,
            'type': 'text',
            'result': result
        })
        print("\n" + "="*60)
        print("📊 檢測結果")
        print("="*60)
        if 'analysis' in result:
            print(result['analysis'])
        else:
            print(f"❌ 錯誤: {result.get('error', '未知錯誤')}")
    
    elif args.image:
        print("🖼️  處理圖片輸入...")
        result = detector.detect_advertisement_compliance(args.image, "image")
        results.append({
            'input': args.image,
            'type': 'image',
            'result': result
        })
        print("\n" + "="*60)
        print("📊 檢測結果")
        print("="*60)
        if 'analysis' in result:
            print(f"📷 圖片路徑: {args.image}")
            if 'extracted_text' in result:
                print(f"📝 提取文字: {result['extracted_text'][:200]}...")
            print(result['analysis'])
        else:
            print(f"❌ 錯誤: {result.get('error', '未知錯誤')}")
    
    elif args.batch:
        print("📦 處理批量輸入...")
        try:
            import pandas as pd
            df = pd.read_csv(args.batch)
            
            for idx, row in df.iterrows():
                print(f"🔄 處理第 {idx+1} 筆資料...")
                
                # 檢查是否有圖片路徑欄位
                if 'image_path' in row and pd.notna(row['image_path']):
                    result = detector.detect_advertisement_compliance(row['image_path'], "image")
                    input_data = row['image_path']
                    input_type = 'image'
                elif 'text' in row and pd.notna(row['text']):
                    result = detector.detect_advertisement_compliance(row['text'], "text")
                    input_data = row['text']
                    input_type = 'text'
                else:
                    print(f"⚠️  第 {idx+1} 筆資料缺少有效輸入")
                    continue
                
                results.append({
                    'index': idx,
                    'input': input_data,
                    'type': input_type,
                    'result': result
                })
            
            print(f"✅ 批量處理完成，共處理 {len(results)} 筆資料")
            
        except Exception as e:
            print(f"❌ 批量處理失敗: {e}")
            return
    
    else:
        print("⚠️  請提供要檢測的文字（--text）或圖片（--image）或批量檔案（--batch）")
        print("\n使用範例:")
        print("  文字檢測: python script.py --text '本產品能有效改善體質'")
        print("  圖片檢測: python script.py --image 'advertisement.jpg'")
        print("  批量處理: python script.py --batch 'data.csv'")
        print("  重建索引: python script.py --rebuild")
        return
    
    # 輸出結果
    if args.output:
        print(f"💾 儲存結果到 {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("🎉 檢測完成！")

# ========= 8. 互動式界面 =========

def interactive_mode():
    """互動式檢測模式"""
    print("=" * 60)
    print("🔍 增強版廣告檢測系統")
    print("支援文字和圖片輸入（ColPali 技術）")
    print("=" * 60)
    
    # 初始化系統
    vs = EnhancedVectorStore()
    try:
        vs.load()
        print("✅ 向量資料庫載入成功")
    except FileNotFoundError:
        print("❌ 找不到索引檔案，請先使用 --rebuild 建立索引")
        return
    
    detector = EnhancedAdvertisementDetector(vs)
    
    while True:
        print("\n" + "="*50)
        print("請選擇輸入類型:")
        print("1. 📝 文字輸入")
        print("2. 🖼️  圖片輸入 (ColPali)")
        print("3. 🚪 退出")
        
        choice = input("請輸入選項 (1-3): ").strip()
        
        if choice == "1":
            text = input("請輸入廣告文字: ").strip()
            if text:
                result = detector.detect_advertisement_compliance(text, "text")
                print("\n" + "="*60)
                print("📊 檢測結果")
                print("="*60)
                if 'analysis' in result:
                    print(result['analysis'])
                else:
                    print(f"❌ 錯誤: {result.get('error', '未知錯誤')}")
            else:
                print("⚠️  請輸入有效的文字內容")
        
        elif choice == "2":
            image_path = input("請輸入圖片路徑: ").strip()
            if image_path and os.path.exists(image_path):
                result = detector.detect_advertisement_compliance(image_path, "image")
                print("\n" + "="*60)
                print("📊 檢測結果")
                print("="*60)
                if 'analysis' in result:
                    if 'extracted_text' in result:
                        print(f"📝 提取文字: {result['extracted_text'][:200]}...")
                    print(result['analysis'])
                else:
                    print(f"❌ 錯誤: {result.get('error', '未知錯誤')}")
            else:
                print("❌ 圖片檔案不存在或路徑無效")
        
        elif choice == "3":
            print("👋 感謝使用廣告檢測系統！")
            break
        
        else:
            print("❌ 無效選項，請重新選擇")

# ========= 9. 使用範例 =========

def example_usage():
    """使用範例"""
    print("=" * 60)
    print("📚 使用範例")
    print("=" * 60)
    
    # 初始化系統
    vs = EnhancedVectorStore()
    try:
        vs.load()  # 假設已有索引
    except FileNotFoundError:
        print("❌ 請先建立索引: python script.py --rebuild")
        return
    
    detector = EnhancedAdvertisementDetector(vs)
    
    # 文字檢測範例
    print("📝 文字檢測範例:")
    text_example = "本產品能有效治療糖尿病，經臨床證實療效顯著！"
    result = detector.detect_advertisement_compliance(text_example, "text")
    print("檢測結果:")
    print(result.get('analysis', '檢測失敗'))
    
    # 圖片檢測範例（假設有圖片）
    print("\n🖼️  圖片檢測範例:")
    image_example = "advertisement.jpg"
    if os.path.exists(image_example):
        result = detector.detect_advertisement_compliance(image_example, "image")
        print("檢測結果:")
        print(result.get('analysis', '檢測失敗'))
    else:
        print("❌ 範例圖片不存在")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 沒有參數時啟動互動模式
        interactive_mode()
    else:
        # 有參數時使用命令行模式
        main()
