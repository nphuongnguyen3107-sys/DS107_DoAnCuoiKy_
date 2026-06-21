# ml_pipeline/rag_engine.py
"""
Hệ thống RAG nhẹ (Retrieval-Augmented Generation) đọc file PDF và tìm kiếm thông tin lâm sàng.
Không yêu cầu C++ compiler hoặc Vector Database cồng kềnh.
"""

import os
import re
import glob
from pypdf import PdfReader

# Các từ dừng tiếng Việt phổ biến để loại bỏ khỏi từ khóa tìm kiếm
VIETNAMESE_STOPWORDS = {
    "là", "thì", "mà", "ở", "có", "và", "của", "cho", "được", "bị", "các", "những", 
    "một", "với", "trong", "theo", "đến", "về", "ra", "này", "kia", "đó", "nào", "gì"
}

class RAGEngine:
    def __init__(self, workspace_dir=None):
        self.workspace_dir = workspace_dir or os.getcwd()
        self.chunks = []  # Danh sách lưu các đoạn văn bản: [{"text": ..., "page": ..., "source": ...}]
        self.pdf_loaded = False
        self.loaded_files = []
        
        # Tự động quét và tải file PDF khi khởi tạo
        self.auto_load_pdfs()

    def auto_load_pdfs(self):
        """Quét tìm file PDF trong thư mục dự án và tải nội dung."""
        # Ưu tiên các file PDF hướng dẫn điều trị hoặc kháng sinh
        pdf_pattern = os.path.join(self.workspace_dir, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        # Quét thêm cả thư mục cha nếu app.py chạy ở thư mục con
        if not pdf_files:
            pdf_pattern_parent = os.path.join(os.path.dirname(self.workspace_dir), "*.pdf")
            pdf_files = glob.glob(pdf_pattern_parent)

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            # Bỏ qua các file báo cáo đồ án của sinh viên (thường bắt đầu bằng [report])
            if filename.lower().startswith("[report]") or "2452" in filename:
                print(f"[RAG] Skipping project report: {filename}")
                continue
                
            print(f"[RAG] Reading guideline file: {filename}...")
            self.load_pdf(pdf_path)

    def load_pdf(self, pdf_path):
        """Đọc và phân đoạn (chunking) file PDF."""
        filename = os.path.basename(pdf_path)
        if not os.path.exists(pdf_path):
            print(f"[RAG] File does not exist: {filename}")
            return
            
        try:
            reader = PdfReader(pdf_path)
            
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text()
                if not text:
                    continue
                
                # Phân tách trang thành các đoạn văn dựa trên ký tự xuống dòng kép hoặc ngắt dòng hợp lý
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                
                for p in paragraphs:
                    # Lưu thông tin chunk
                    self.chunks.append({
                        "text": p,
                        "page": page_num,
                        "source": filename
                    })
                    
            self.pdf_loaded = True
            self.loaded_files.append(filename)
            print(f"[RAG] Loaded {filename} ({len(reader.pages)} pages, {len(self.chunks)} chunks).")
        except Exception as e:
            print(f"[RAG] Error reading PDF {filename}: {e}")

    def clean_text(self, text):
        """Làm sạch văn bản, chuyển về chữ thường và xóa ký tự đặc biệt."""
        text = text.lower()
        # Thay thế các ký tự đặc biệt thành khoảng trắng
        text = re.sub(r'[^\w\s\dàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        return text

    def search(self, query, top_k=4):
        """Tìm kiếm các đoạn văn bản liên quan nhất dựa trên câu hỏi của người dùng."""
        if not self.pdf_loaded or not self.chunks:
            return []
            
        # 1. Trích xuất từ khóa tìm kiếm
        clean_query = self.clean_text(query)
        words = [w for w in clean_query.split() if w and w not in VIETNAMESE_STOPWORDS]
        
        if not words:
            return []
            
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_text_clean = self.clean_text(chunk["text"])
            score = 0.0
            
            # Tính điểm khớp từ khóa
            for word in words:
                # Nếu từ khóa xuất hiện trong chunk
                count = chunk_text_clean.count(word)
                if count > 0:
                    # Thuốc kháng sinh, vi khuẩn và các gen đặc biệt được nhân hệ số trọng số
                    is_medical_term = any(term in word for term in [
                        "kháng_sinh", "kháng", "vi_khuẩn", "phác_đồ", "điều_trị",
                        "nhạy", "đề_kháng", "cephalosporin", "fluoroquinolone", 
                        "carbapenem", "penicillin", "aminoglycoside", "tetracycline",
                        "ciprofloxacin", "levofloxacin", "imipenem", "meropenem",
                        "gentamicin", "amikacin", "streptomycin", "colistin", "mcr",
                        "gyra", "parc", "ctx-m", "tem", "oxa", "shv", "ndm", "kpc"
                    ])
                    weight = 3.0 if is_medical_term else 1.0
                    score += count * weight
            
            # Cộng thêm điểm thưởng nếu có cụm từ ghép (2 từ liên tiếp) khớp nguyên bản
            for i in range(len(words) - 1):
                phrase = words[i] + " " + words[i+1]
                if phrase in chunk_text_clean:
                    score += 5.0
                    
            if score > 0:
                scored_chunks.append((score, chunk))
                
        # Sắp xếp các đoạn theo điểm số giảm dần
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top_k kết quả tốt nhất
        results = []
        seen_texts = set()
        
        for score, chunk in scored_chunks:
            # Tránh lấy các đoạn trùng lặp nội dung
            text_summary = chunk["text"][:100]
            if text_summary not in seen_texts:
                results.append(chunk)
                seen_texts.add(text_summary)
                if len(results) >= top_k:
                    break
                    
        return results
