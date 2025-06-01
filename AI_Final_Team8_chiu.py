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
# ----------------------------
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
# ----------------------------
# Code by D13944024, 蔡宜淀, Team 8

# 配置 OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 嵌入模型與維度
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536
MAX_TOKENS = 7000
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500

# ========= 1. 通用工具函數 =========

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

# ========= 2. JSON 萃取處理器 =========

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

# ========= 3. 增強版向量資料庫 =========

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
    # -----------------------------------------------------------------
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
            # representatives.append(texts[rep_idx])
    
        return representatives
    # -----------------------------------------------------------------
    
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
        # -------------------- 將所有提取文本進行語義的聚合 --------------------
        '''
        if content_type in ["violation", "allowed"] and len(all_texts) > 10:
            print(f"🔍  原始 {content_type} 句數: {len(all_texts)}，執行語意壓縮...")
            cluster_size = max(1, len(all_texts) // 20)  # 每20句保留1句
            all_texts = self._cluster_representative_texts(all_texts, max_clusters=cluster_size)
            # all_texts = self._cluster_representative_texts(all_texts, max_clusters=5)
            print(f"✅  壓縮後代表句數: {len(all_texts)}")
        '''
        # -------------------------------------------------------------------
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

# ========= 4. 主判斷函數 =========

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
#     system_prompt = f"""## 目標
# 分析廣告文字內容，根據法律條款和案例判斷廣告用詞是否涉及誇大療效及違法，並提供違法機率評估。回應內容必須完全依照格式，且使用繁體中文。

# ## 重要判斷原則
# ### 智能檢測結果優先原則
# - 智能檢測風險等級: {risk_level}
# - 智能檢測風險分數: {risk_score}%
# - 發現允許用詞: {allowed_phrases}
# - 有官方認證: {has_approval}

# ### 判斷邏輯修正
# 1. **當智能檢測為 low 風險且發現允許用詞時**：
#  - 違法機率應 ≤ 30%
#  - 優先認定為合法
#  - 重點檢查是否有明顯療效宣稱

# 2. **當智能檢測為 low 風險但無允許用詞時**：
#  - 違法機率應在 30-60% 之間
#  - 仔細檢查是否有隱含療效

# 3. **當智能檢測為 medium/high 風險時**：
#  - 先分析廣告中的文字是否只為標題的分類，並非宣傳商品的廣告詞
#  - 若只是不相關的頁面文字，就算包含不合法文字，違法機率也應在 30-60% 之間
#  - 或為針對商品宣傳的文字，違法機率才應 ≥ 70%
#  - 嚴格檢查療效宣稱

# ### 嚴格文本分析要求
# - **禁止引用廣告文本中不存在的內容**
# - **只能分析實際出現的文字**
# - **不得自行添加或想像違法內容**

# ### 合規性判斷標準
# - **無罪判定原則**：不捏造或過度解讀廣告文字，從寬認定合法性
# - **允許使用的廣告用詞**：
# - 「調整體質」「促進新陳代謝」「幫助入睡」「青春美麗」
# - 「排便順暢」「健康維持」「營養補給」「調節生理機能」
# - 「能完整補充人體營養」「提升生理機能」「改變細菌叢生態」
# - 當這些用詞出現時，應降低違法風險評估

# ### 裁罰依據
# - **《食品安全衛生管理法》第45條**
# - 違反第28條第1項：**罰 4 萬至 400 萬元**

# ## 分析步驟
# 1. **逐字檢視廣告內容**：只分析實際存在的文字
# 2. **比對允許用詞清單**：確認是否使用合規用詞
# 3. **檢查療效宣稱**：識別明確的醫療或療效聲明
# 4. **參考智能檢測結果**：作為主要判斷依據
# 5. **給出最終評估**：確保與智能檢測結果邏輯一致

# ## 評估流程（務必依序思考並回答）
# 1. 廣告文本是否有出現「禁止用詞」？若有，評估是否為直接療效宣稱？
# 2. 若有「禁止用詞」，但並無上下文組合，評估是否只為頁面文字，與廣告宣傳無關，降低判斷的風險？
# 3. 是否出現「允許用詞」？若有，列出所有出現用詞。若沒有，仍分析廣告文字文義是否存在淺在風險？
# 4. 是否出現「上下文組合違規語句」？有的話列出組合。
# 5. 根據以上結果，對照智能檢測結果風險等級，計算合理的違法機率（從以下範圍中選一：10%, 30%, 50%, 70%, 90%）。
# 6. 引用至少 1 筆 Vector Store 法規內容，並標出相關條文。
# > 違法機率選項只能為：10%、30%、50%、70%、90%（禁止填入其他數字）

# ## 回應格式要求
# ### 若違法機率 ≤ 30% (對應智能檢測 low 風險 + 允許用詞)
# 違法機率: X% 
# ✔ 出現允許用詞: 是 / 否
# ✔ 出現違規用詞: 是 / 否
# ✔ 出現上下文違規組合: 是 / 否
# 違法內容分析: 經分析廣告內容，主要使用允許範圍內的用詞如「[具體用詞]」，違法風險較低。
# 罰款額度: 無。 
# 參考依據: [vector store 文件 ID]

# ### 若違法機率 31-69% (對應智能檢測 low 風險但需注意)
# 違法機率: X% 
# ✔ 出現允許用詞: 是 / 否
# ✔ 出現違規用詞: 是 / 否
# ✔ 出現上下文違規組合: 是 / 否
# 違法內容分析: 廣告中「[實際存在的具體文字]」可能涉及療效暗示，建議謹慎使用。
# 違反條款: [適用法規] 
# 罰款額度: [依據法規] 
# 參考依據: [vector store 文件 ID]

# ### 若違法機率 ≥ 70% (對應智能檢測 medium/high 風險)
# 違法機率: X% 
# ✔ 出現允許用詞: 是 / 否
# ✔ 出現違規用詞: 是 / 否
# ✔ 出現上下文違規組合: 是 / 否
# 違法內容分析: 廣告中「[實際存在的具體文字]」明確宣稱療效，違反相關規定。
# 違反條款: [適用法規] 
# 罰款額度: [依據法規] 
# 參考依據: [vector store 文件 ID]
# """
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
    「保證有效」「絕對有效」「神奇療效」
    「活化毛囊」「刺激毛囊細胞」「增加毛囊角質細胞增生」「刺激毛囊讓髮絲再次生長不易落脫」「刺激毛囊不萎縮」「堅固毛囊刺激新生秀髮」「頭頂不再光禿禿」「頭頂不再光溜溜」
    「避免稀疏」「避免髮量稀少問題」「有效預防落髮/抑制落髮/減少落髮」「有效預防掉髮/抑制掉髮/減少掉髮」「增強(增加)自體免疫力」「增強淋巴引流」「促進細胞活動」
    「深入細胞膜作用」「減弱角化細胞」「刺激細胞呼吸作用」「提高肌膚細胞帶氧率」「進入甲母細胞和甲床深度滋潤」「刺激增長新的健康細胞」「增加細胞新陳代謝」
    「促進肌膚神經醯胺合成」「維持上皮組織機能的運作」「重建皮脂膜」「重建角質層」「促進(刺激)膠原蛋白合成」「促進(刺激)膠原蛋白增生」「瘦身」「減肥」「去脂」「減脂」「消脂」
    「燃燒脂肪」「消耗脂肪」「減緩臀部肥油囤積」「預防脂肪細胞堆積」「刺激脂肪分解酵素」「纖(孅)體」「塑身」「雕塑曲線」「消除掰掰肉」「消除蝴蝶袖」「告別小腹婆」「減少橘皮組織」「豐胸」
    「隆乳」「使胸部堅挺不下垂」「感受托高集中的驚人效果」「漂白」「使乳暈漂成粉紅色」「不過敏」「零過敏」「減過敏」「抗過敏」「舒緩過敏」「修護過敏」「過敏測試」「醫藥級」
    「鎮靜劑」「鎮定劑」「消除浮腫」「改善微血管循環」「功能強化微血管」「增加血管含氧量提高肌膚帶氧率」「預防(防止)肥胖紋」「預防(防止)妊娠紋」「緩減妊娠紋生產」
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

## 法規依據資料(Vector Store)

你將獲得來自法規資料庫的內容(包含法條與案例說明)，這些資料有助於你比對是否構成違規，請至少引用1筆資料，並明確指出出處(檔名與編號)。

---

## 請遵守以下限制，避免生成語義幻覺：

- 僅能使用廣告文本中實際出現的字詞進行推理，不得虛構內容。
- 嚴禁編造、推測、想像或自行補充未出現的療效或內容（例如：未提到疾病卻自行聯想治療功效）。
- 所有判斷、舉例、引用皆需逐字來自於原始廣告文本、法規資料與智能檢測結果。
- 回應不得包含「可能含有」「可能暗示」「應該是」等推測性、含糊字眼。

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

## 廣告判斷範例 (以下分別為三種情況的範例)

### 低風險(違法機率 10% 或 30%)
廣告文本：「每日補充益生菌，有助於維持消化道健康與排便順暢。促進新陳代謝，幫助入睡，讓你青春美麗一整天！」

檢測步驟：
1. 描述商品：文字為廣告詞句，明確宣傳商品功效。
   - 是，開始分析。
2. 允許用詞：出現「排便順暢」「促進新陳代謝」「幫助入睡」「青春美麗」，皆屬合法用詞。
   - 出現允許用詞: 是，列舉允許用詞，合法性參考見[...]。
3. 違規用詞：無
   - 無違規用詞。
4. 上下文違規組合：無
   - 無上下文違規組合。
5. 官方認證：若有，列舉([包含的衛署字號])
6. 整合智能檢測結果:
   - 智能風險等級：low
   - 智能風險分數：0%
   - 有允許用詞：['排便順暢', '促進新陳代謝', '幫助入睡', '青春美麗']
   - 有官方認證：True
7. 違法機率判斷：
   - 依據以上分析，廣告用詞合規，無違規詞句，且智能檢測風險低，違法機率應屬低風險範圍。
   - 違法機率: 10%

最終判斷回應：

違法機率: 10%  
✔ 出現允許用詞: 是 (['排便順暢', '促進新陳代謝', '幫助入睡', '青春美麗'])
✔ 出現違規用詞: 否
✔ 出現上下文違規組合: 否
違法內容分析: 廣告中使用的詞句均屬於合法允許範圍，未涉及誇大療效或未經許可的醫療用語。智能檢測結果顯示低風險，綜合判斷違法可能性低。  
罰款額度: 無

參考依據: [...]

智能檢測: low 風險
使用的 vector store 文件: [...]

### 資料不足但判定合法 (違法機率 30%)
廣告文本：「【健康生活館】主打：代替方案、高血壓、糖尿病、失眠專區產品熱銷中！連身衣 包巾／包被 童襪/手腳套 童鞋 褲褲/帽子 穿搭配件 禮盒專區 嬰幼兒清潔 濕紙巾 紗布/棉巾 濕巾加熱器 奶瓶蔬果洗潔液/洗碗液」

檢測步驟：
1. 描述商品：文字為頁面分類或入口說明，並未宣傳單一商品。
   - 否
2. 允許用詞：無
3. 違規用詞：出現「高血壓」「糖尿病」「失眠」，但僅為分類名稱，無療效宣稱。
4. 上下文違規組合：出現「代替方案」+「高血壓、糖尿病」、「專家推薦」，但僅為分類名稱，無療效宣稱。
5. 官方認證：無
6. 整合智能檢測結果:
   - 智能風險等級：high
   - 智能風險分數：70%
   - 有允許用詞：無
   - 有官方認證：無
7. 違法機率判斷：
   - 依據以上分析，無描述產品的廣告用詞，但包含違規詞句，智能檢測風險高，但因「資料不足」，請保守估計違法機率為30%。
   - 違法機率: 30%（安全保守）

最終判斷回應：

違法機率: 30%
✔ 出現允許用詞: 否
✔ 出現違規用詞: 是 (['高血壓', '糖尿病', '失眠'])
✔ 出現上下文違規組合: 是 (「替代方案」+「X疾病」、「專家推薦」)
違法內容分析: 廣告中使用的詞句不屬於合法允許範圍。智能檢測結果顯示高風險，但因用詞僅為分類名稱，無療效宣稱，綜合判斷違法可能性低。
罰款額度: 無

參考依據: [...]

智能檢測: high 風險
使用的 vector store 文件: [...]

### 高風險 (違法機率 70% 或 90%)
廣告文本：「本產品經臨床證實療效顯著，通過藥物測試，能快速治療糖尿病與高血壓，是你最佳替代藥物的選擇！」

檢測步驟：
1. 描述商品：文字為廣告詞句，明確宣傳商品功效。
   - 是，開始分析。
2. 允許用詞：無
3. 違規用詞：出現「高血壓」「糖尿病」「治療」。
4. 上下文違規組合：出現['臨床證實+療效顯著'、'最佳替代+藥物']、「通過」+「X測試」，並宣稱療效。
5. 官方認證：無
6. 整合智能檢測結果:
   - 智能風險等級：high
   - 智能風險分數：90%
   - 有允許用詞：無
   - 有官方認證：無
7. 違法機率判斷：
   - 依據以上分析，產品的廣告用詞包含違規詞句，同時出現上下文違規，智能檢測風險高，估計違法機率為90%。
   - 違法機率: 90%

最終判斷回應：

違法機率: 90%
✔ 出現允許用詞: 否
✔ 出現違規用詞: 是 (['高血壓', '糖尿病', '治療'])
✔ 出現上下文違規組合: 是 (「臨床證實」+「療效顯著」、「最佳替代」+「藥物」)
違法內容分析: 廣告中使用的詞句不屬於合法允許範圍。智能檢測結果顯示高風險，廣告文案療效宣稱，綜合判斷違法可能性高。
違反條款: 食品安全衛生管理法第28條
罰款額度: 4萬至400萬元 

參考依據: [...]

智能檢測: high 風險
使用的 vector store 文件: [...]

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
            temperature=0.05,
            max_tokens=1600,
            top_p=0.5
        )
        
        response_text = resp.choices[0].message.content
        
        # 一致性檢查和修正
        # response_text = validate_and_fix_response(
        #     response_text, compliance_check, ad_text
        # )
        
        return {
            'ad_text': ad_text,
            'timestamp': datetime.utcnow().isoformat(),
            'response': response_text,
            'vector_store_ids': vector_store_ids,
            'source_files': list(set([c.get('source_file', c['id']) for c in contexts])),
            'compliance_analysis': compliance_check,
            'model_used': 'gpt-4.1-mini'
        }
    except Exception as e:
        return {
            'ad_text': ad_text,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'vector_store_ids': vector_store_ids,
            'compliance_analysis': compliance_check,
            'model_used': 'gpt-4.1-mini'
        }

def validate_and_fix_response(response_text: str, compliance_check: dict, ad_text: str) -> str:
    """驗證和修正 GPT 回應的一致性"""
    
    # 提取違法機率
    prob_match = re.search(r'違法機率:\s*(\d+)%', response_text)
    if not prob_match:
        return response_text
    
    gpt_probability = int(prob_match.group(1))
    risk_level = compliance_check['risk_level']
    allowed_phrases = compliance_check['allowed_phrases_found']
    
    # 檢查一致性並修正
    should_fix = False
    expected_range = None
    
    if risk_level == 'low' and allowed_phrases:
        # 應該是低違法機率 (≤30%)
        if gpt_probability > 30:
            should_fix = True
            expected_range = (10, 25)
    elif risk_level == 'low' and not allowed_phrases:
        # 應該是中等違法機率 (30-60%)
        if gpt_probability < 30 or gpt_probability > 60:
            should_fix = True
            expected_range = (35, 55)
    elif risk_level in ['medium', 'high']:
        # 應該是高違法機率 (≥70%)
        if gpt_probability < 70:
            should_fix = True
            expected_range = (75, 85)
    
    if should_fix and expected_range:
        # 修正違法機率
        new_probability = expected_range[0]
        response_text = re.sub(
            r'違法機率:\s*\d+%', 
            f'違法機率: {new_probability}%', 
            response_text
        )
        
        # 如果是低風險且有允許用詞，修正分析內容
        if risk_level == 'low' and allowed_phrases:
            allowed_text = '、'.join(allowed_phrases)
            new_analysis = f"經分析廣告內容，主要使用允許範圍內的用詞如「{allowed_text}」，違法風險較低。"
            response_text = re.sub(
                r'違法內容分析:.*?(?=違反條款|罰款額度)', 
                f'違法內容分析: {new_analysis}\n罰款額度: 無。\n',
                response_text,
                flags=re.DOTALL
            )
    
    return response_text

def check_text_hallucination(response_text: str, ad_text: str, threshold: float = 0.75) -> bool:
    """檢查 GPT 是否出現語義幻覺（引用不存在的文字）"""
    
    # 提取 GPT 回應中引用的文字片段
    quoted_patterns = [
        r'「([^」]+)」',  # 中文引號
        r'『([^』]+)』',  # 中文引號  
        r'"([^"]+)"',    # 英文雙引號
        r"'([^']+)'"     # 英文單引號
    ]
    
    quoted_texts = []
    for pattern in quoted_patterns:
        quoted_texts.extend(re.findall(pattern, response_text))
    
    if not quoted_texts:
        return False
        
    try:
        ad_emb = openai.Embedding.create(model=EMBEDDING_MODEL, input=ad_text)["data"][0]["embedding"]
        hallucination_detected = False
        
        for quoted in quoted_texts:
            if len(quoted) > 5:  # 只檢查較長的引用
                try:
                    q_emb = openai.Embedding.create(model=EMBEDDING_MODEL, input=quoted)["data"][0]["embedding"]
                    similarity = np.dot(ad_emb, q_emb) / (np.linalg.norm(ad_emb) * np.linalg.norm(q_emb))
                    
                    if similarity < threshold:
                        print(f"⚠️ 檢測到可能的語義幻覺：「{quoted}」(相似度: {similarity:.3f})")
                        hallucination_detected = True
                except Exception as e:
                    print(f"⚠️ 處理引用時發生錯誤：{quoted}, 錯誤：{str(e)}")
                    
        return hallucination_detected
        
    except Exception as e:
        print(f"⚠️ 幻覺檢測失敗：{str(e)}")
        return False
# ----------------------- 將違法機率的數字擷取 -----------------------
def extract_violation_percent(text: str, th: int) -> int:
    match = re.search(r"違法機率[:：]\s*(\d+)%", text)
    if match:
        percent = int(match.group(1))
        if percent >= th:
            violation = 0
        else:
            violation = 1
        return percent, violation
    else: return 10, 1
# ------------------------------------------------------------------

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="AI_Final_Team8_MOHW RAG system")
    parser.add_argument(
        "--json_folder", type=str, default="regulations_json",
        help="存放法規 JSON 文件的資料夾"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="強制重新建立索引"
    )
    parser.add_argument(
        "--output_json", type=str, default="results.json",
        help="結果輸出 JSON 檔案名稱"
    )
    parser.add_argument(
        "--query", type=str,
        help="單一查詢文本"
    )
    parser.add_argument(
        "--batch_file", type=str,
        help="批次處理的文本檔案（每行一個廣告文本）"
    )
    # ------------------ 新增輸入的query ------------------
    parser.add_argument(
        "--input_query_file", type=str,
        help="輸入的query文件"
    )
    parser.add_argument(
        "--threshold", type=int, default = 50,
        help="從機率轉成violation的閥值"
    )
    # ------------------ 新增輸出的query ------------------
    parser.add_argument(
        "--output_csv", type=str, default='final_template.csv',
        help="輸出的answer CSV文件"
    )
    # ----------------------------------------------------
    
    args = parser.parse_args()
    
    # 檢查 OpenAI API Key
    if not openai.api_key:
        print("錯誤：請設定 OPENAI_API_KEY 環境變數")
        return
    
    # 初始化向量存儲
    vs = EnhancedVectorStore()
    
    # 建立或載入索引
    if args.rebuild or not os.path.exists(vs.index_path):
        print("建立新的向量索引...")
        try:
            vs.build(args.json_folder)
            print("索引建立完成")
        except Exception as e:
            print(f"建立索引時發生錯誤: {e}")
            return
    else:
        print("載入現有的向量索引...")
        try:
            vs.load()
            print("索引載入完成")
        except Exception as e:
            print(f"載入索引時發生錯誤: {e}")
            return
    
    results = []
    
    # 處理單一查詢
    if args.query:
        print(f"\n分析廣告文本: {args.query}")
        result = strict_determine_legality(args.query, vs)
        print("\n=== 分析結果 ===")
        if 'error' in result:
            print(f"錯誤: {result['error']}")
        else:
            print(result['response'])
            # 檢查幻覺
            if check_text_hallucination(result['response'], args.query):
                print("\n⚠️  警告：檢測到可能的幻覺內容")
        results.append(result)
    # --------------------------------------------------------------------------------
    # 處理 CSV 檔案批次 query（每筆一欄 Question）
    elif args.input_query_file:
        if not os.path.exists(args.input_query_file):
            print(f"找不到輸入的 CSV 檔案: {args.input_query_file}")
            return
        try:
            df = pd.read_csv(args.input_query_file)
        except Exception as e:
            print(f"無法讀取 CSV 檔案: {e}")
            return

        if "Question" not in df.columns:
            print("錯誤：CSV 檔案中找不到 'Question' 欄位")
            return

        print(f"開始批次處理 CSV，共 {len(df)} 筆查詢文本...")
        answers = []
        for i, row in df.iterrows():
            ad_text = str(row["Question"])
            print(f"\n處理 {i+1}/{len(df)}: {ad_text[:60]}...")
            token_len = count_tokens(ad_text)
            
            if token_len > MAX_TOKENS:
                print(f"⚠️  文本太長（{token_len} tokens），執行 smart chunking 處理...")
                '''
                chunks = smart_text_chunker(ad_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
                chunk_results = []
                highest_prob = -1
                highest_chunk = None
                for j, chunk in enumerate(chunks):
                    print(f"\n ➤  子片段 {j+1}/{len(chunks)}")
                    result = strict_determine_legality(chunk, vs)
        
                    print("=== 分析結果 ===")
                    if 'error' in result:
                        print(f"錯誤: {result['error']}")
                    else:
                        print(result['response'])
        
                        ca = result.get("compliance_analysis", {})
                        if ca:
                            print(f"\n智能檢測: {ca.get('risk_level', '未知')} 風險")
                        if result.get("vector_store_ids"):
                            print(f"使用的 vector store 文件: {', '.join(result['vector_store_ids'])}")
                        if check_text_hallucination(result['response'], chunk):
                            print("⚠️  警告：檢測到可能的幻覺內容")
                    chunk_text = result.get("response", "")
                    chunk_percent, chunk_violation = extract_violation_percent(chunk_text)
                    if (j == 0 and highest_prob == -1) or (chunk_percent > highest_prob):
                        highest_prob = chunk_percent
                        highest_chunk = result
                    # chunk_results.append(result)
                # results.extend(chunk_results)
                results.append(highest_chunk)              
                # response_text = result.get("response", "")
                response_text = highest_chunk.get("response", "")
                _, violation = extract_violation_percent(response_text)
                answers.append({
                    "ID": row["ID"]-1,
                    "answer": violation
                })
                '''
                ad_text = ad_text[:6500]
                result = strict_determine_legality(ad_text, vs)
        
                print("=== 分析結果 ===")
                if 'error' in result:
                    print(f"錯誤: {result['error']}")
                else:
                    print(result['response'])
        
                    ca = result.get("compliance_analysis", {})
                    # if ca:
                    #     print(f"\n智能檢測: {ca.get('risk_level', '未知')} 風險")
                    # if result.get("vector_store_ids"):
                    #     print(f"使用的 vector store 文件: {', '.join(result['vector_store_ids'])}")
                    if check_text_hallucination(result['response'], ad_text):
                        print("⚠️  警告：檢測到可能的幻覺內容")
        
                results.append(result)
                response_text = result.get("response", "")
                _, violation = extract_violation_percent(response_text, args.threshold)
                answers.append({
                    "ID": row["ID"]-1,
                    "answer": violation
                })
            else:
                result = strict_determine_legality(ad_text, vs)
        
                print("=== 分析結果 ===")
                if 'error' in result:
                    print(f"錯誤: {result['error']}")
                else:
                    print(result['response'])
        
                    ca = result.get("compliance_analysis", {})
                    # if ca:
                    #     print(f"\n智能檢測: {ca.get('risk_level', '未知')} 風險")
                    # if result.get("vector_store_ids"):
                    #     print(f"使用的 vector store 文件: {', '.join(result['vector_store_ids'])}")
                    if check_text_hallucination(result['response'], ad_text):
                        print("⚠️  警告：檢測到可能的幻覺內容")
        
                results.append(result)
                response_text = result.get("response", "")
                _, violation = extract_violation_percent(response_text, args.threshold)
                answers.append({
                    "ID": row["ID"]-1,
                    "answer": violation
                })
        # for i, row in df.iterrows():
        #     ad_text = str(row["Question"])
        #     print(f"\n處理 {i+1}/{len(df)}: {ad_text[:60]}...")

        #     result = strict_determine_legality(ad_text, vs)

        #     print("\n=== 分析結果 ===")
        #     if 'error' in result:
        #         print(f"錯誤: {result['error']}")
        #     else:
        #         print(result['response'])

        #         ca = result.get("compliance_analysis", {})
        #         if ca:
        #             print(f"\n智能檢測: {ca.get('risk_level', '未知')} 風險")
                
        #         if result.get("vector_store_ids"):
        #             print(f"使用的 vector store 文件: {', '.join(result['vector_store_ids'])}")
                
        #         if check_text_hallucination(result['response'], ad_text):
        #             print("⚠️  警告：檢測到可能的幻覺內容")

        #     results.append(result)
    # --------------------------------------------------------------------------------
    
    # 處理批次檔案
    elif args.batch_file:
        if not os.path.exists(args.batch_file):
            print(f"找不到批次檔案: {args.batch_file}")
            return
        
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"開始批次處理 {len(lines)} 個廣告文本...")
        
        for i, ad_text in enumerate(lines, 1):
            print(f"\n處理 {i}/{len(lines)}: {ad_text[:50]}...")
            result = strict_determine_legality(ad_text, vs)
            
            if 'error' in result:
                print(f"錯誤: {result['error']}")
            else:
                print("分析完成")
                # 提取違法機率
                response = result['response']
                if '違法機率:' in response:
                    prob_line = response.split('\n')[0]
                    print(f"結果: {prob_line}")
                
                # 檢查幻覺
                if check_text_hallucination(result['response'], ad_text):
                    print("⚠️  警告：檢測到可能的幻覺內容")
            
            results.append(result)
    
    # 互動模式
    else:
        print("\n=== AI_Final_Team8_MOHW RAG system ===")
        print("輸入廣告文本進行檢測，輸入 'quit' 結束")
        
        while True:
            try:
                ad_text = input("\n請輸入廣告文本: ").strip()
                if ad_text.lower() == 'quit':
                    break
                
                if not ad_text:
                    continue
                
                print("分析中...")
                result = strict_determine_legality(ad_text, vs)
                
                print("\n=== 分析結果 ===")
                if 'error' in result:
                    print(f"錯誤: {result['error']}")
                else:
                    print(result['response'])
                    print(f"\n智能檢測: {result['compliance_analysis']['risk_level']} 風險")
                    print(f"使用的 vector store 文件: {', '.join(result['vector_store_ids'])}")
                    
                    # 檢查幻覺
                    if check_text_hallucination(result['response'], ad_text):
                        print("\n⚠️  警告：檢測到可能的幻覺內容")
                
                results.append(result)
                
            except KeyboardInterrupt:
                print("\n\n程式結束")
                break
    
    # 儲存結果
    if results:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至: {args.output_json}")
        
        # 顯示統計
        total = len(results)
        high_risk = sum(1 for r in results if 'compliance_analysis' in r and r['compliance_analysis']['risk_level'] == 'high')
        consistent = sum(1 for r in results if 'response' in r and not check_text_hallucination(r['response'], r['ad_text']))
        
        print(f"總共分析: {total} 個文本")
        print(f"高風險: {high_risk} 個 ({high_risk/total*100:.1f}%)")
        print(f"一致性檢查通過: {consistent} 個 ({consistent/total*100:.1f}%)")

    if answers:
        pd.DataFrame(answers).to_csv(args.output_csv, index=False)
        print(f"答案已儲存至: {args.output_csv}")
        
if __name__ == "__main__":
    main()