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
            "保證有效", "絕對有效", "神奇療效"
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
    system_prompt = f"""## 目標
分析廣告文字內容，根據法律條款和案例判斷廣告用詞是否涉及誇大療效及違法，並提供違法機率評估。回應內容必須完全依照格式，且使用繁體中文。

## 重要判斷原則
### 智能檢測結果優先原則
- 智能檢測風險等級: {risk_level}
- 智能檢測風險分數: {risk_score}%
- 發現允許用詞: {allowed_phrases}
- 有官方認證: {has_approval}

### 判斷邏輯修正
1. **當智能檢測為 low 風險且發現允許用詞時**：
 - 違法機率應 ≤ 30%
 - 優先認定為合法
 - 重點檢查是否有明顯療效宣稱

2. **當智能檢測為 low 風險但無允許用詞時**：
 - 違法機率應在 30-60% 之間
 - 仔細檢查是否有隱含療效

3. **當智能檢測為 medium/high 風險時**：
 - 違法機率應 ≥ 70%
 - 嚴格檢查療效宣稱

### 嚴格文本分析要求
- **禁止引用廣告文本中不存在的內容**
- **只能分析實際出現的文字**
- **不得自行添加或想像違法內容**

### 合規性判斷標準
- **無罪判定原則**：不捏造或過度解讀廣告文字，從寬認定合法性
- **允許使用的廣告用詞**：
- 「調整體質」「促進新陳代謝」「幫助入睡」「青春美麗」
- 「排便順暢」「健康維持」「營養補給」「調節生理機能」
- 「能完整補充人體營養」「提升生理機能」「改變細菌叢生態」
- 當這些用詞出現時，應降低違法風險評估

### 裁罰依據
- **《食品安全衛生管理法》第45條**
- 違反第28條第1項：**罰 4 萬至 400 萬元**

## 分析步驟
1. **逐字檢視廣告內容**：只分析實際存在的文字
2. **比對允許用詞清單**：確認是否使用合規用詞
3. **檢查療效宣稱**：識別明確的醫療或療效聲明
4. **參考智能檢測結果**：作為主要判斷依據
5. **給出最終評估**：確保與智能檢測結果邏輯一致

## 回應格式要求
### 若違法機率 ≤ 30% (對應智能檢測 low 風險 + 允許用詞)
違法機率: X% 
違法內容分析: 經分析廣告內容，主要使用允許範圍內的用詞如「[具體用詞]」，違法風險較低。
罰款額度: 無。 
參考依據: [vector store 文件 ID]

### 若違法機率 31-69% (對應智能檢測 low 風險但需注意)
違法機率: X% 
違法內容分析: 廣告中「[實際存在的具體文字]」可能涉及療效暗示，建議謹慎使用。
違反條款: [適用法規] 
罰款額度: [依據法規] 
參考依據: [vector store 文件 ID]

### 若違法機率 ≥ 70% (對應智能檢測 medium/high 風險)
違法機率: X% 
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
1. **嚴格按照智能檢測結果進行判斷**
2. **只分析廣告文本中實際存在的內容**
3. **不得引用或分析文本中不存在的文字**
4. **當發現允許用詞時，應從寬認定合法性**
5. **確保違法機率與智能檢測結果邏輯一致**

請根據以上資料進行分析，特別注意智能檢測結果的指導作用。
"""
    
    # 調用 GPT 模型
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.05,
            max_tokens=800,
            top_p=0.9
        )
        
        response_text = resp.choices[0].message.content
        
        # 一致性檢查和修正
        response_text = validate_and_fix_response(
            response_text, compliance_check, ad_text
        )
        
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
        print("\n=== 互動式廣告合規檢測（增強版）===")
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

if __name__ == "__main__":
    main()