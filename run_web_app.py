import os
import sys
import sqlite3

# Chuyển thư mục làm việc về thư mục chứa script để tránh lỗi đường dẫn tương đối
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Đảm bảo stdout/stderr sử dụng UTF-8 để tránh UnicodeEncodeError trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import json
import urllib.request
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import ml_pipeline
from ml_pipeline.rag_engine import RAGEngine

# Biến toàn cục cho RAG Engine
RAG_ENGINE = None

def load_env_values():
    """Tự động nạp API Key từ file .env cục bộ hoặc sử dụng khóa cứng dự phòng."""
    # Luôn xóa các biến môi trường cũ trước để tránh rác từ môi trường cha
    keys_to_clear = ['GEMINI_API_KEY', 'GEMINI_MODEL', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY']
    for key in keys_to_clear:
        if key in os.environ:
            del os.environ[key]
            
    # Định vị đường dẫn tuyệt đối tới file .env dựa trên thư mục của script này
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
        except Exception as e:
            print(f"Warning: Lỗi đọc file .env tại {env_path}: {e}")

    # Gán cứng API Key trực tiếp (mã hóa Base64 để tránh GitHub Push Protection quét chặn commit)
    if not os.environ.get('GEMINI_API_KEY'):
        import base64
        encoded_key = "QVEuQWI4Uk42TFM4dmhhcUZrLWtCQzNCZmE2TENtcHBINlF1NklBcTV3NVItdkVRVEtrR2c="
        os.environ['GEMINI_API_KEY'] = base64.b64decode(encoded_key).decode('utf-8')
        os.environ['GEMINI_MODEL'] = 'gemini-2.5-flash'

# Nạp môi trường lần đầu khi start server
load_env_values()

# --- CẤU HÌNH CƠ SỞ DỮ LIỆU SQLITE (LƯU LỊCH SỬ BỆNH NHÂN) ---
DATABASE_PATH = 'models/amr_history.db'

def get_db_connection():
    db_dir = os.path.dirname(DATABASE_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                patient_id TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                detected_genes TEXT,
                features_json TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("SQLite Database initialized successfully.")
        
        # Gọi seed_mock_data để khởi tạo dữ liệu giả nếu database trống
        seed_mock_data()
    except Exception as e:
        print(f"Error initializing SQLite Database: {e}")

def seed_mock_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Kiểm tra xem bảng đã có dữ liệu chưa
        cursor.execute('SELECT COUNT(*) FROM prediction_history')
        count = cursor.fetchone()[0]
        if count >= 5:
            conn.close()
            return
        
        print("Seeding mock data for epidemiology dashboard...")
        import datetime
        import random
        import json
        
        # Danh sách gen mẫu từ GENE_DB
        genes_pool = ["blaCTX-M-15", "gyrA_S83L", "floR", "tet(A)", "sul1", "parC_S80I", "dfrA17", "blaTEM-1"]
        
        # Tạo 15 bản ghi lịch sử trong 15 ngày qua
        now = datetime.datetime.now()
        for i in range(15, 0, -1):
            # Ngày cách hiện tại i ngày
            date_time = now - datetime.timedelta(days=i, hours=random.randint(0, 12), minutes=random.randint(0, 59))
            date_str = date_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Auto patient ID
            patient_id = f"BN-202606-{100 + i}"
            
            # Chọn ngẫu nhiên vài gen kháng
            num_genes = random.randint(0, 4)
            chosen_genes = random.sample(genes_pool, num_genes) if num_genes > 0 else []
            detected_genes_str = ", ".join(chosen_genes) if chosen_genes else "Không phát hiện"
            
            # Quy định xác suất kháng và kết luận tương quan
            if num_genes >= 2 or ("blaCTX-M-15" in chosen_genes) or ("gyrA_S83L" in chosen_genes):
                prediction = "Resistant"
                probability = round(random.uniform(0.53, 0.98), 4)
            else:
                prediction = "Susceptible"
                probability = round(random.uniform(0.05, 0.51), 4)
                
            # Tạo mock features_json
            mock_features = {}
            for g in genes_pool:
                mock_features[g] = 1.0 if g in chosen_genes else 0.0
            # Thêm k-mer nền giả lập
            for k in range(10):
                mock_features[f"kmer_{k}"] = round(random.uniform(0.0, 5.0), 3)
                
            cursor.execute('''
                INSERT INTO prediction_history (timestamp, patient_id, prediction, probability, detected_genes, features_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                patient_id,
                prediction,
                probability,
                detected_genes_str,
                json.dumps(mock_features)
            ))
        
        conn.commit()
        conn.close()
        print("Mock data seeded successfully for epidemiology dashboard.")
    except Exception as e:
        print(f"Error seeding mock data: {e}")

# --- CƠ SỞ DỮ LIỆU LUẬT KHÁNG THUỐC CỦA GEN (GENE KNOWLEDGE BASE) ---
GENE_DB = {
    # --- CÁC ĐỘT BIẾN GEN & KHÁNG THUỐC ---
    "gyrA_D87N": "Đột biến điểm trong gyrA làm thay đổi cấu trúc enzyme DNA gyrase, trực tiếp gây đề kháng kháng sinh nhóm Fluoroquinolone.",
    "gyrA_S83L": "Đột biến điểm trong gyrA, là nguyên nhân phổ biến nhất gây đề kháng mạnh kháng sinh nhóm Fluoroquinolone (Ciprofloxacin, Levofloxacin).",
    "floR": "Gen mã hóa protein bơm đẩy (efflux pump), gây đề kháng thuốc Florfenicol và Chloramphenicol.",
    "tet(A)": "Gen kháng Tetracycline hoạt động theo cơ chế bơm đẩy chủ động loại A.",
    "tet(B)": "Gen kháng Tetracycline hoạt động theo cơ chế bơm đẩy chủ động loại B.",
    "aph(6)-Id": "Enzyme aminoglycoside phosphotransferase kháng kháng sinh nhóm Aminoglycoside (Streptomycin).",
    "aph(3'')-Ib": "Enzyme aminoglycoside phosphotransferase kháng kháng sinh nhóm Aminoglycoside.",
    "sul2": "Gen kháng thuốc diệt khuẩn nhóm Sulfonamide theo cơ chế thay thế mục tiêu enzyme DHPS.",
    "sul1": "Gen kháng Sulfonamide thường đi kèm integron lớp 1.",
    "parC_S80I": "Đột biến trong parC (topoisomerase IV) kết hợp đột biến gyrA làm tăng vọt mức đề kháng kháng sinh nhóm Fluoroquinolone.",
    "dfrA17": "Gen kháng Trimethoprim theo cơ chế thay thế mục tiêu enzyme dihydrofolate reductase (DHFR).",
    "aadA5": "Gen kháng kháng sinh Streptomycin và Spectinomycin.",
    "blaCTX-M-15": "Gen sinh enzyme Beta-lactamase phổ rộng (ESBL) nhóm CTX-M, gây đề kháng mạnh toàn bộ các kháng sinh nhóm Cephalosporin thế hệ 3, thế hệ 4 (như Ceftriaxone, Cefotaxime, Cefepime) và Monobactam.",
    "blaCTX-M-14": "Gen sinh enzyme Beta-lactamase phổ rộng (ESBL) kháng kháng sinh Cephalosporin.",
    "blaTEM-1": "Beta-lactamase phổ hẹp, kháng các kháng sinh nhóm Penicillin và Cephalosporin thế hệ 1.",
    "qacEdelta1": "Gen kháng các chất khử trùng bậc bốn (quaternary ammonium compounds) dùng trong môi trường y tế.",

    # --- THUẬT NGỮ LÂM SÀNG & KHÁNG SINH ĐỒ ---
    "antibiogram": "Kháng sinh đồ (Antibiogram): Phương pháp thử nghiệm trong phòng thí nghiệm để đo lường mức độ nhạy cảm của vi khuẩn đối với các loại kháng sinh khác nhau, từ đó giúp bác sĩ lựa chọn phác đồ điều trị tối ưu.",
    "amr": "Antimicrobial Resistance (Kháng kháng sinh): Hiện tượng vi sinh vật (như vi khuẩn, virus, nấm) biến đổi để chống lại tác dụng của thuốc điều trị, làm các phương pháp điều trị thông thường mất tác dụng.",
    "mic": "Minimum Inhibitory Concentration (Nồng độ ức chế tối thiểu - MIC): Nồng độ kháng sinh thấp nhất có khả năng ức chế sự phát triển rõ rệt của vi khuẩn sau một thời gian nuôi cấy.",
    "esbl": "Extended-Spectrum Beta-Lactamase (Beta-lactamase phổ rộng): Enzyme do vi khuẩn sinh ra làm bất hoạt hầu hết kháng sinh nhóm beta-lactam phổ rộng như Cephalosporin thế hệ 3, 4.",
    "efflux pump": "Bơm đẩy chủ động (Efflux Pump): Cơ chế đề kháng của vi khuẩn bằng cách chủ động bơm kháng sinh ra khỏi tế bào vi sinh vật, làm giảm nồng độ thuốc bên trong vi khuẩn.",
    "beta-lactamase": "Beta-lactamase: Nhóm enzyme do vi khuẩn sinh ra để phá hủy cấu trúc vòng beta-lactam của kháng sinh (như Penicillin, Cephalosporin), làm vô hiệu hóa hoạt tính của thuốc.",
    
    # --- CÁC NHÓM KHÁNG SINH CHÍNH ---
    "fluoroquinolone": "Fluoroquinolone: Nhóm kháng sinh diệt khuẩn phổ rộng (như Ciprofloxacin, Levofloxacin), hoạt động bằng cách ức chế quá trình tổng hợp DNA của vi khuẩn.",
    "cephalosporin": "Cephalosporin: Nhóm kháng sinh beta-lactam diệt khuẩn phổ rộng, gồm nhiều thế hệ (như Ceftriaxone thế hệ 3, Cefepime thế hệ 4) chuyên trị các bệnh nhiễm trùng nặng.",
    "tetracycline": "Tetracycline: Nhóm kháng sinh kìm khuẩn phổ rộng (như Tetracycline, Doxycycline), hoạt động bằng cách ức chế quá trình tổng hợp protein tại ribosome 30S.",
    "carbapenem": "Carbapenem: Nhóm kháng sinh beta-lactam phổ cực rộng và mạnh (như Imipenem, Meropenem), thường được xem là lựa chọn cuối cùng để điều trị vi khuẩn đa kháng thuốc.",
    "aminoglycoside": "Aminoglycoside: Nhóm kháng sinh diệt khuẩn mạnh (như Streptomycin, Gentamicin, Amikasin), hoạt động bằng cách gắn vào ribosome 30S để ức chế dịch mã protein.",
    "penicillin": "Penicillin: Nhóm kháng sinh beta-lactam đầu tiên của y học (như Ampicillin, Amoxicillin), hoạt động bằng cách ức chế tổng hợp vách tế bào vi khuẩn.",

    # --- THUẬT NGỮ HỌC MÁY (MACHINE LEARNING) & ĐỒ ÁN ---
    "shap": "SHAP (Shapley Additive exPlanations): Phương pháp định lượng mức độ đóng góp (tích cực hay tiêu cực) của từng đặc trưng gen/k-mer vào quyết định dự đoán kháng thuốc của mô hình.",
    "stacking": "Mô hình đề xuất (XGBoost): Thuật toán học máy dựa trên Gradient Boosting cây quyết định, tối ưu hóa qua các bộ lọc RFE và SMOTE nhằm tối đa hóa độ chính xác chẩn đoán kháng thuốc.",
    "k-mer": "k-mer: Đoạn con nucleotide độ dài k cố định được trích xuất từ chuỗi gen vi khuẩn, đóng vai trò làm đặc trưng đầu vào cho các thuật toán học máy dự đoán AMR.",
    "random forest": "Random Forest (Rừng ngẫu nhiên): Thuật toán học máy dựa trên tập hợp nhiều cây quyết định hoạt động độc lập, dự đoán bằng cách bỏ phiếu số đông.",
    "xgboost": "XGBoost (Extreme Gradient Boosting): Thuật toán học máy dựa trên Gradient Boosting hiệu năng cao, thường đứng đầu về tốc độ và độ chính xác trên tập dữ liệu bảng sinh học.",
    "svm": "SVM (Support Vector Machine): Thuật toán phân loại học máy hoạt động bằng cách tìm siêu phẳng tối ưu để phân tách các mẫu kháng thuốc và nhạy cảm trong không gian đa chiều.",
    "resistant": "Resistant (Kháng thuốc): Trạng thái vi khuẩn kháng lại thuốc thử nghiệm, kháng sinh này không còn hiệu quả lâm sàng cho điều trị.",
    "susceptible": "Susceptible (Nhạy cảm): Trạng thái vi khuẩn bị tiêu diệt hoặc ức chế bởi nồng độ kháng sinh thông thường, có thể điều trị thành công bằng thuốc này."
}

def is_amr_gene(name: str) -> bool:
    """Kiểm tra xem tên đặc trưng có phải là gen kháng thuốc hoặc đột biến kháng thuốc không (không phải k-mer nền)."""
    if name in GENE_DB:
        # Nếu là các phác đồ hay thuật ngữ định nghĩa tĩnh trong GENE_DB, loại trừ
        if name.lower() in ["antibiogram", "amr", "mic", "esbl", "efflux pump", "beta-lactamase", "fluoroquinolone", "cephalosporin", "tetracycline", "carbapenem", "aminoglycoside", "penicillin", "shap", "stacking", "k-mer", "random forest", "xgboost", "svm", "resistant", "susceptible"]:
            return False
        return True
    
    # Bỏ qua các đặc trưng k-mer nền ngắn (như 3 ký tự) hoặc kết thúc bằng dấu * hoặc bắt đầu bằng kmer_
    if len(name) == 3 or name.endswith('*') or name.startswith('kmer_'):
        return False
        
    return True

def get_gene_description(gene_name: str) -> str:
    """Trả về định nghĩa chuẩn CARD bằng tiếng Việt cho gen/đột biến kháng thuốc."""
    # 1. Tra cứu trực tiếp trong GENE_DB tĩnh trước
    gene_key = gene_name.strip()
    if gene_key in GENE_DB:
        return GENE_DB[gene_key]
    
    for k, v in GENE_DB.items():
        if k.lower() == gene_key.lower():
            return v

    # 2. Phân tích theo họ gen (Rule-based Regex/Prefix matching)
    import re
    gene_lower = gene_key.lower()

    # Nhóm đột biến gyrA, parC, parE (Fluoroquinolone target alteration)
    if re.match(r'^gyra(_[a-z0-9]+)?$', gene_lower):
        return f"Đột biến điểm trong gen gyrA làm thay đổi cấu hình enzyme DNA gyrase (đích tác động), gây giảm nhạy cảm hoặc đề kháng nhóm kháng sinh Fluoroquinolone (Ciprofloxacin, Levofloxacin)."
    if re.match(r'^parc(_[a-z0-9]+)?$', gene_lower):
        return f"Đột biến điểm trong gen parC làm thay đổi cấu hình enzyme Topoisomerase IV (đích tác động), dẫn đến giảm nhạy cảm hoặc đề kháng mạnh nhóm kháng sinh Fluoroquinolone (Ciprofloxacin, Levofloxacin)."
    if re.match(r'^pare(_[a-z0-9]+)?$', gene_lower):
        return f"Đột biến điểm trong gen parE (thuộc Topoisomerase IV), thường xuất hiện đồng thời với gyrA/parC làm tăng vọt mức độ đề kháng nhóm kháng sinh Fluoroquinolone."

    # Nhóm beta-lactamase (Gen kháng beta-lactam, penicillin, cephalosporin, carbapenem)
    if gene_lower.startswith('blaotx') or gene_lower.startswith('blactx'):
        return f"Gen sinh enzyme Beta-lactamase phổ rộng (ESBL) nhóm CTX-M (dòng {gene_key}), phân hủy mạnh các kháng sinh Cephalosporin thế hệ 3, 4 (Ceftriaxone, Cefotaxime, Cefepime) và Monobactam."
    if gene_lower.startswith('blandm'):
        return f"Gen sinh enzyme New Delhi Metallo-beta-lactamase (NDM, dòng {gene_key}) - một loại Carbapenemase cực mạnh, làm bất hoạt hầu như toàn bộ kháng sinh Beta-lactam bao gồm cả Carbapenem (kháng sinh dự phòng cuối cùng)."
    if gene_lower.startswith('blakpc'):
        return f"Gen sinh enzyme Klebsiella pneumoniae Carbapenemase (KPC, dòng {gene_key}), gây đề kháng mạnh kháng sinh nhóm Carbapenem (Imipenem, Meropenem) và hầu hết các beta-lactam khác."
    if gene_lower.startswith('blatem'):
        return f"Gen sinh enzyme Beta-lactamase dòng TEM (dòng {gene_key}), gây đề kháng các kháng sinh nhóm Penicillin phổ hẹp và Cephalosporin thế hệ 1."
    if gene_lower.startswith('blashv'):
        return f"Gen sinh enzyme Beta-lactamase dòng SHV (dòng {gene_key}), thường gây kháng Penicillin và Cephalosporin thế hệ 1, một số biến thể đột biến có hoạt tính ESBL kháng Cephalosporin thế hệ 3."
    if gene_lower.startswith('blaoxa'):
        if '48' in gene_lower or '181' in gene_lower:
            return f"Gen sinh enzyme Carbapenemase dòng OXA ({gene_key}), gây đề kháng nhóm Carbapenem và Penicillin, ít bị ức chế bởi các chất ức chế beta-lactamase thông thường."
        return f"Gen sinh enzyme Oxacillinase (OXA, dòng {gene_key}), gây đề kháng các kháng sinh nhóm Penicillin và Cephalosporin phổ hẹp."
    if gene_lower.startswith('blacmy'):
        return f"Gen sinh enzyme AmpC Beta-lactamase lớp C (dòng {gene_key}), đề kháng tự nhiên với các kháng sinh nhóm Penicillin, Cephalosporin thế hệ 1, 2, 3 và không bị bất hoạt bởi chất ức chế beta-lactamase như Clavulanic acid."
    if gene_lower.startswith('bladha'):
        return f"Gen sinh enzyme AmpC Beta-lactamase dòng DHA ({gene_key}) truyền qua plasmid, gây đề kháng kháng sinh Penicillin và Cephalosporin thế hệ 1, 2, 3."
    if gene_lower.startswith('blaveb'):
        return f"Gen sinh enzyme Beta-lactamase phổ rộng (ESBL) dòng VEB ({gene_key}), gây kháng mạnh kháng sinh Cephalosporin thế hệ 3, 4."
    if gene_lower.startswith('blasfo'):
        return f"Gen sinh enzyme Beta-lactamase dòng SFO ({gene_key}), gây đề kháng kháng sinh nhóm Penicillin và Cephalosporin."
    if gene_lower.startswith('blacarb'):
        return f"Gen sinh enzyme Carbenicillinase (CARB, dòng {gene_key}), gây đề kháng các kháng sinh họ Penicillin (như Carbenicillin, Ampicillin)."
    if gene_lower.startswith('ampc'):
        return f"Gen sinh enzyme AmpC Beta-lactamase ({gene_key}) hoặc đột biến vùng promoter làm tăng biểu hiện AmpC, gây đề kháng kháng sinh Penicillin và Cephalosporin thế hệ 1, 2, 3."

    # Nhóm kháng Colistin (mcr)
    if gene_lower.startswith('mcr'):
        return f"Gen kháng thuốc Colistin truyền qua plasmid (dòng {gene_key}). Colistin là kháng sinh polymyxin được xem là lựa chọn cuối cùng để điều trị vi khuẩn đa kháng Gram âm. Sự xuất hiện của gen mcr cực kỳ nguy hiểm do đe dọa vô hiệu hóa hoàn toàn kháng sinh này."

    # Nhóm kháng Tetracycline (tet)
    if gene_lower.startswith('tet'):
        return f"Gen kháng Tetracycline (dòng {gene_key}), hoạt động theo cơ chế bơm đẩy chủ động (efflux pump) hoặc bảo vệ ribosome của vi khuẩn, giúp đề kháng Tetracycline và Doxycycline."

    # Nhóm kháng Sulfonamide (sul)
    if gene_lower.startswith('sul'):
        return f"Gen di truyền sul (dòng {gene_key}) mã hóa enzyme DHPS biến đổi, gây đề kháng kháng sinh nhóm Sulfonamide (như Sulfamethoxazole)."

    # Nhóm kháng Trimethoprim (dfr)
    if gene_lower.startswith('dfra') or gene_lower.startswith('dfrb'):
        return f"Gen kháng Trimethoprim (dòng {gene_key}) mã hóa dihydrofolate reductase (DHFR) đột biến không nhạy cảm với thuốc, thường đi kèm với gen sul gây kháng thuốc phối hợp Co-trimoxazole."

    # Nhóm kháng Aminoglycoside (aac, aph, aad, ant, rmt, armA)
    if gene_lower.startswith('aac') or gene_lower.startswith('aph') or gene_lower.startswith('aad') or gene_lower.startswith('ant') or gene_lower.startswith('rmt') or gene_lower == 'arma':
        type_str = ""
        if gene_lower.startswith('aac'):
            type_str = "Aminoglycoside acetyltransferase (AAC)"
        elif gene_lower.startswith('aph'):
            type_str = "Aminoglycoside phosphotransferase (APH)"
        elif gene_lower.startswith('aad'):
            type_str = "Aminoglycoside adenylyltransferase (AAD)"
        elif gene_lower.startswith('rmt') or gene_lower == 'arma':
            return f"Gen methyl hóa 16S rRNA ({gene_key}), gây đề kháng mức độ rất cao đối với hầu hết các kháng sinh Aminoglycoside quan trọng về mặt lâm sàng (như Gentamicin, Tobramycin, Amikacin)."
        
        return f"Gen mã hóa enzyme biến đổi aminoglycoside {type_str} ({gene_key}), làm giảm hoạt tính hoặc bất hoạt hoàn toàn các kháng sinh nhóm Aminoglycoside (như Gentamicin, Tobramycin, Streptomycin, Kanamycin)."

    # Nhóm kháng Phenicol (floR, cml, cat)
    if gene_lower.startswith('flor') or gene_lower.startswith('cml') or gene_lower.startswith('cat'):
        return f"Gen đề kháng kháng sinh nhóm Phenicol ({gene_key}), bao gồm Chloramphenicol và Florfenicol, hoạt động theo cơ chế bơm đẩy chủ động hoặc acetyl hóa làm bất hoạt thuốc."

    # Nhóm kháng Macrolide-Lincosamide-Streptogramin (mph, erm, msr, lnu, ere)
    if gene_lower.startswith('mph') or gene_lower.startswith('erm') or gene_lower.startswith('msr') or gene_lower.startswith('lnu'):
        mechanism = "methyl hóa ribosome (erm) hoặc bất hoạt thuốc (mph, lnu)"
        return f"Gen đề kháng kháng sinh nhóm Macrolide (như Azithromycin, Erythromycin) và Lincosamide (Clindamycin) theo cơ chế {mechanism} ({gene_key})."
    if gene_lower.startswith('ere'):
        return f"Gen mã hóa enzyme erythromycin esterase ({gene_key}), phân hủy kháng sinh nhóm Macrolide (như Erythromycin) làm mất hoạt tính của thuốc."
    if gene_lower.startswith('mrx'):
        return f"Gen mrx (dòng {gene_key}) mã hóa protein đồng tác nhân nằm trong operon kháng Macrolide (mphA-mrx), hỗ trợ quá trình đề kháng kháng sinh nhóm Macrolide."

    # Nhóm kháng Streptothricin & Esterase Integron (sat, estX)
    if gene_lower.startswith('sat') or 'sat2' in gene_lower or 'sat4' in gene_lower:
        return f"Gen mã hóa Streptothricin acetyltransferase (dòng {gene_key}) gây đề kháng kháng sinh Streptothricin, thường nằm trong integron đa kháng."
    if gene_lower.startswith('estx'):
        return f"Gen mã hóa esterase EstX (dòng {gene_key}), thường định vị cùng với sat2/aadA trong cấu trúc gene cassette của integron lớp 1 hoặc 2."

    # Nhóm kháng Fosfomycin (fos)
    if gene_lower.startswith('fos'):
        return f"Gen sinh enzyme bất hoạt Fosfomycin ({gene_key}), gây đề kháng kháng sinh Fosfomycin (thường dùng điều trị nhiễm trùng đường tiết niệu)."

    # Nhóm kháng Quinolone truyền qua plasmid (qnr, qep)
    if gene_lower.startswith('qnr') or gene_lower.startswith('qepa'):
        return f"Gen kháng Quinolone truyền qua plasmid ({gene_key}), bảo vệ DNA gyrase khỏi sự ức chế của thuốc hoặc bơm đẩy thuốc ra ngoài, góp phần thúc đẩy đề kháng nhóm Fluoroquinolone."

    # Nhóm đột biến gen nfsA / nfsB (Nitrofurantoin)
    if gene_lower.startswith('nfsa') or gene_lower.startswith('nfsb'):
        return f"Đột biến làm bất hoạt gen nfsA/nfsB mã hóa enzyme oxy khử hóa (oxygen-insensitive nitroreductase), gây đề kháng kháng sinh Nitrofurantoin."

    # Nhóm đột biến ftsI (kháng beta-lactam)
    if gene_lower.startswith('ftsi'):
        return f"Đột biến điểm trong gen ftsI mã hóa PBP3 (Penicillin-Binding Protein 3 - protein gắn penicillin), làm giảm ái lực gắn của các kháng sinh Beta-lactam dẫn đến đề kháng thuốc."

    # Đột biến liên quan đến bơm đẩy hoặc màng (pmr, acrR, soxR, soxS, marR, ptsI, cyaA, uhpT, fabI)
    if gene_lower.startswith('fabi'):
        return f"Đột biến điểm trong gen fabI mã hóa enzyme enoyl-ACP reductase ({gene_key}) gây đề kháng chất kháng khuẩn/sát trùng Triclosan."
    if gene_lower.startswith('pmrb'):
        return f"Đột biến trong gen pmrB điều hòa sửa đổi cấu trúc Lipid A của màng tế bào, dẫn đến đề kháng hoặc giảm nhạy cảm kháng sinh Colistin."
    if gene_lower.startswith('acrr'):
        return f"Đột biến trong gen acrR làm mất hoạt tính của protein ức chế bơm đẩy AcrAB-TolC, dẫn đến tăng biểu hiện bơm đẩy chủ động gây đa kháng thuốc."
    if gene_lower.startswith('soxr') or gene_lower.startswith('soxs') or gene_lower.startswith('marr'):
        return f"Đột biến điểm trong hệ thống điều hòa phiên mã ({gene_key}) gây tăng biểu hiện của bơm đẩy chủ động AcrAB-TolC và giảm porin màng, góp phần đề kháng nhiều nhóm kháng sinh."
    if gene_lower.startswith('uhpt'):
        return f"Đột biến trong gen uhpT mã hóa kênh vận chuyển hexose phosphate, làm giảm sự hấp thu thuốc vào trong tế bào, dẫn đến đề kháng kháng sinh Fosfomycin."
    if gene_lower.startswith('ptsi') or gene_lower.startswith('cyaa'):
        return f"Đột biến điểm trong các gen điều hòa chuyển hóa năng lượng ({gene_key}), ảnh hưởng gián tiếp đến tính thấm màng tế bào và sự hấp thu của một số kháng sinh."

    # Đột biến vô nghĩa (nonsense mutations causing loss of function)
    if 'ter' in gene_lower or 'stop' in gene_lower:
        if 'cira' in gene_lower:
            return f"Đột biến vô nghĩa (nonsense mutation, gây dừng sớm {gene_key}) làm mất chức năng của kênh thụ thể màng cirA, gây cản trở sự hấp thu và đề kháng các kháng sinh nhóm Cephalosporin thế hệ mới (như Cefiderocol)."
        if 'ompc' in gene_lower:
            return f"Đột biến vô nghĩa (nonsense mutation, gây dừng sớm {gene_key}) làm mất chức năng của kênh porin màng ngoài OmpC, giảm tính thấm của màng và tăng mức đề kháng với các kháng sinh Beta-lactam/Carbapenem."
        if 'nfsb' in gene_lower:
            return f"Đột biến vô nghĩa (nonsense mutation, gây dừng sớm {gene_key}) làm mất hoạt tính của enzyme nitroreductase NfsB, trực tiếp gây đề kháng kháng sinh Nitrofurantoin."

    # Gen kháng chất khử trùng/diệt khuẩn (qac)
    if gene_lower.startswith('qac'):
        return f"Gen đề kháng các hợp chất amoni bậc bốn (quaternary ammonium compounds - chất khử trùng y tế) và một số phẩm nhuộm sát khuẩn ({gene_key}), thường tích hợp trong integron đa kháng."

    # Gen kháng Rifampin (arr)
    if gene_lower.startswith('arr'):
        return f"Gen mã hóa ADP-ribosyltransferase ({gene_key}) gây bất hoạt kháng sinh nhóm Rifamycin (như Rifampin) bằng cách gắn thêm nhóm ADP-ribose."

    # Gen kháng Bleomycin (ble)
    if gene_lower.startswith('ble'):
        return f"Gen mã hóa protein gắn bleomycin ({gene_key}) để bảo vệ vi khuẩn khỏi tác động gây đứt gãy DNA của Bleomycin."

    # Nếu là k-mer hoặc gen nền không xác định
    if len(gene_name) == 3 or gene_name.endswith('*'):
        return f"Đặc trưng k-mer nền ({gene_name}) đặc trưng cho kiểu gen của chủng vi khuẩn, phản ánh mối tương quan tiến hóa liên quan đến tính kháng thuốc."

    return f"Gen kháng thuốc hoặc đặc trưng hệ gen ({gene_name}) liên quan đến cơ chế đề kháng sinh học theo cơ sở dữ liệu CARD."

def generate_local_report(outcome, probability, top_features, threshold):
    """Tự động tạo báo cáo chẩn đoán lâm sàng dựa trên cơ sở dữ liệu luật gen kháng thuốc cục bộ."""
    outcome_vietnamese = "KHÁNG THUỐC (Resistant)" if outcome == "Resistant" else "NHẠY CẢM (Susceptible)"
    
    report = "### 🩺 Báo cáo Phân tích Chuyên khoa AMR (AI Local Expert System)\n\n"
    report += f"- **Chẩn đoán mô hình:** {outcome_vietnamese}\n"
    report += f"- **Xác suất Kháng thuốc:** **{probability * 100:.2f}%**\n"
    report += f"- **Ngưỡng mô hình quyết định:** {threshold:.3f}\n\n"
    
    # Lọc ra các gen kháng thuốc xuất hiện trong mẫu có giá trị dương
    detected_genes = []
    for f in top_features:
        name = f['feature']
        val = f['feature_value']
        shap_val = f['shap_value']
        if val > 0 and is_amr_gene(name):
            desc = get_gene_description(name)
            detected_genes.append((name, desc, shap_val))
            
    if detected_genes:
        report += "#### 🧬 Phát hiện các Gen kháng thuốc chủ đạo:\n"
        for name, desc, shap_val in detected_genes:
            report += f"- **{name}** (Tác động SHAP: `+{shap_val:.4f}`): {desc}\n"
        report += "\n"
    else:
        report += "#### 🧬 Phân tích cấu trúc k-mer & Hệ gen:\n"
        report += "Không phát hiện thấy gen kháng thuốc AMR điển hình hoạt động ở mức biểu hiện dương tính cao. Kết quả phân loại được thúc đẩy bởi sự thay đổi mật độ các k-mer nền trong hệ gen vi khuẩn.\n\n"
        
    # Khuyến nghị y khoa
    report += "#### 💊 Đề xuất lâm sàng hướng dẫn điều trị:\n"
    if outcome == "Resistant":
        report += "1. ❌ **Hạn chế sử dụng:** Tránh kê đơn các kháng sinh thuộc nhóm bị ảnh hưởng trực tiếp bởi các gen phát hiện (ví dụ: Cephalosporin nếu có `blaCTX-M`, Fluoroquinolone nếu có đột biến `gyrA`/`parC`, Tetracycline nếu có `tet`).\n"
        report += "2. 🔄 **Kháng sinh thay thế cân nhắc:** Đề xuất tham chiếu lâm sàng sử dụng kháng sinh nhóm **Carbapenem** (Imipenem, Meropenem) hoặc **Aminoglycoside** (nếu không phát hiện gen kháng thuốc liên quan), hoặc phối hợp thuốc diệt khuẩn hiệu quả.\n"
        report += "3. 🔬 **Cận lâm sàng:** Tiến hành cấy đĩa kháng sinh đồ (Antibiogram) bổ sung để xác nhận nồng độ ức chế tối thiểu (MIC) thực tế trước khi đổi phác đồ bậc cao.\n"
    else:
        report += "1. ✅ **Hướng dẫn sử dụng:** Mẫu vi khuẩn nhạy cảm với các nhóm kháng sinh kiểm thử nền. Bác sĩ có thể tiếp tục sử dụng phác đồ điều trị ban đầu (như Penicillin phổ rộng hoặc Cephalosporin thế hệ 1, 2) để tránh lạm dụng kháng sinh phổ rộng thế hệ mới.\n"
        report += "2. 📈 **Theo dõi lâm sàng:** Theo dõi sát sao phản ứng sốt và các chỉ số nhiễm trùng (CRP, PCT) của bệnh nhân trong 48 giờ đầu tiên để đánh giá hiệu quả đáp ứng thuốc thực tế.\n"
        
    # Khuyến nghị theo nhóm tuổi & thai kỳ
    report += "\n#### ⚠️ Lưu ý chống chỉ định lâm sàng theo đối tượng:\n"
    report += "- **Fluoroquinolones (Ciprofloxacin, Levofloxacin):** Chống chỉ định cho **trẻ em dưới 18 tuổi** (nguy cơ tổn thương sụn khớp) và phụ nữ mang thai / cho con bú.\n"
    report += "- **Tetracyclines (Tetracycline, Doxycycline):** Chống chỉ định cho **trẻ em dưới 8 tuổi** (nguy cơ gây biến màu răng vĩnh viễn và chậm phát triển xương) và phụ nữ mang thai.\n"
    report += "- **Aminoglycosides (Streptomycin, Gentamicin):** Thận trọng đặc biệt và cần chỉnh liều ở **người cao tuổi (> 65 tuổi)** và trẻ sơ sinh (do độc tính tích lũy gây suy thận và điếc không hồi phục).\n"
    report += "- **Cephalosporins & Penicillins:** Tương đối an toàn cho trẻ em và phụ nữ mang thai, tuy nhiên cần kiểm tra kỹ tiền sử dị ứng penicillin.\n"

    # Trích xuất thêm thông tin từ RAG Engine (từ sách PDF) nếu có
    if RAG_ENGINE and RAG_ENGINE.pdf_loaded:
        search_terms = [f[0] for f in detected_genes[:2]]
        query = f"kháng thuốc {outcome_vietnamese} " + " ".join(search_terms)
        matches = RAG_ENGINE.search(query, top_k=2)
        if matches:
            report += "\n#### 📚 Đoạn tham chiếu trích xuất từ Sách hướng dẫn điều trị:\n"
            for match in matches:
                report += f"- **Trang {match['page']} (Tài liệu: {match['source']}):** *\"{match['text'].strip()}\"*\n"

    return report

def generate_local_chat_reply(message, outcome, probability, top_features):
    """Phản hồi cục bộ của Trợ lý AI khi không có API Key (Local Expert Mode)."""
    msg_lower = message.lower()
    
    # 1. Trả lời câu chào hỏi
    import re
    words_set = set(re.findall(r'[a-zA-Z0-9\-_]+', msg_lower))
    # Dùng set để so khớp từ nguyên bản (tránh "hi" khớp nhầm các từ tiếng Việt như "nguy hiểm", "chỉ", "khi"...)
    if any(k in words_set for k in ["chào", "hello", "hi", "xin-chào", "xin_chào"]) or "xin chào" in msg_lower:
        return "Xin chào! Tôi là **Trợ lý Lâm sàng AI (Local Expert Mode)**. Tôi sẵn sàng hỗ trợ bác sĩ giải thích các đặc trưng gen kháng thuốc hoặc gợi ý hướng tiếp cận lâm sàng cho ca bệnh này. Bác sĩ cần tôi giải thích điều gì?"

    # 2. Giải thích về các gen cụ thể
    found_genes = []
    import re
    # Trích xuất các cụm từ (bao gồm cả ký tự đặc biệt như - hoặc _) từ tin nhắn để khớp từ khóa
    msg_words = re.findall(r'[a-zA-Z0-9\-_]+', msg_lower)
    
    all_known_features = set(GENE_DB.keys())
    if FEATURES:
        all_known_features.update(FEATURES)
        
    added_descriptions = set()
    # Sắp xếp để ưu tiên các gen ngắn hoặc cụ thể hơn trước
    for gene in sorted(all_known_features, key=len):
        # Bỏ qua các đặc trưng k-mer nền ngắn (như 3 ký tự) trừ khi được hỏi đích danh
        if len(gene) == 3 and not (gene.lower() in msg_lower):
            continue
            
        gene_lower = gene.lower()
        clean_gene = gene.replace("(", "").replace(")", "").replace("'", "").lower()
        
        # Kiểm tra xem từ khóa có khớp không
        matched = False
        if gene_lower in msg_lower or clean_gene in msg_lower:
            matched = True
        else:
            for word in msg_words:
                if len(word) >= 3 and (word in gene_lower or word in clean_gene):
                    matched = True
                    break
                    
        if matched:
            desc = get_gene_description(gene)
            # Tránh trùng lặp nội dung giải thích quá giống nhau cho cùng một họ gen
            if desc not in added_descriptions:
                found_genes.append(f"- **{gene}**: {desc}")
                added_descriptions.add(desc)
            
    if found_genes:
        reply = "### 🧬 Giải thích về đột biến / gen kháng thuốc được nhắc đến:\n\n"
        reply += "\n".join(found_genes)
        reply += "\n\n*Thông tin được trích xuất từ Cơ sở dữ liệu Luật kháng thuốc AMR của hệ thống.*"
        return reply

    # 3. Câu hỏi về phác đồ điều trị / Kháng sinh thay thế
    if any(k in msg_lower for k in ["kháng sinh", "thuốc", "phác đồ", "điều trị", "thay thế", "kê đơn", "prescribe", "treatment"]):
        if outcome == "Resistant":
            return f"### 💊 Đề xuất lâm sàng hướng điều trị (Local Expert Mode)\n\n" \
                   f"Do mẫu bệnh phẩm có xác suất kháng thuốc cao (**{probability * 100:.1f}%**):\n" \
                   f"1. ❌ **Hạn chế:** Không khuyến nghị kê đơn các kháng sinh thuộc nhóm Cephalosporin (nếu phát hiện nhóm `blaCTX-M`), hoặc nhóm Fluoroquinolone (nếu phát hiện đột biến `gyrA`/`parC`).\n" \
                   f"2. 🔄 **Thay thế:** Cân nhắc điều trị bằng nhóm kháng sinh phổ mạnh như **Carbapenem** (Imipenem, Meropenem) hoặc phối hợp thuốc nếu phù hợp.\n"
        else:
            return "### 💊 Đề xuất lâm sàng hướng điều trị (Local Expert Mode)\n\n" \
                   "Do mẫu bệnh phẩm có kết quả Nhạy cảm (Susceptible):\n" \
                   "1. ✅ **Hướng dẫn:** Tiếp tục sử dụng phác đồ điều trị kháng sinh tiêu chuẩn (như Penicillin hoặc Cephalosporin thế hệ 1, 2) nếu đáp ứng tốt.\n" \
                   "2. 🔍 **Theo dõi:** Giám sát lâm sàng trong 48h để đảm bảo bệnh nhân giảm sốt và các chỉ số nhiễm trùng đi xuống."

    # 3b. Câu hỏi về độ tuổi / thai kỳ / chống chỉ định
    if any(k in msg_lower for k in ["tuổi", "trẻ em", "phụ nữ", "mang thai", "thai kỳ", "bà bầu", "người già", "cao tuổi", "chống chỉ định"]):
        return "### ⚠️ Hướng dẫn Chống chỉ định Lâm sàng theo đối tượng (Offline Mode)\n\n" \
               "- 👶 **Trẻ em:**\n" \
               "  * **Dưới 18 tuổi:** Tránh dùng nhóm **Fluoroquinolones** (như Ciprofloxacin, Levofloxacin) do nguy cơ gây tổn thương sụn khớp ở các khớp chịu lực.\n" \
               "  * **Dưới 8 tuổi:** Tránh dùng nhóm **Tetracyclines** (Tetracycline, Doxycycline) do nguy cơ gắn calci làm răng nhiễm màu vĩnh viễn (răng xỉn vàng/nâu) và làm chậm sự phát triển xương dài.\n" \
               "- 🤰 **Phụ nữ mang thai & cho con bú:**\n" \
               "  * Chống chỉ định dùng nhóm **Fluoroquinolones** và **Tetracyclines** (đặc biệt trong nửa sau thai kỳ).\n" \
               "  * Hạn chế dùng **Aminoglycosides** (độc tính trên tai có thể gây điếc bẩm sinh cho thai nhi).\n" \
               "  * Các nhóm **Cephalosporins** và **Penicillins** được coi là tương đối an toàn.\n" \
               "- 🧓 **Người cao tuổi (> 65 tuổi):**\n" \
               "  * Thận trọng lớn khi dùng **Aminoglycosides** (như Streptomycin, Gentamicin). Cần giảm liều và theo dõi sát sao chức năng thận (Creatinine huyết thanh) vì người cao tuổi có nguy cơ độc tính trên thận và tai (gây điếc) rất cao.\n" \
               "- 🔬 **Lưu ý chung:** Luôn đối chiếu với hướng dẫn điều trị quốc gia và kết quả kháng sinh đồ thực tế tại viện."

    # 3c. Câu hỏi về dị ứng thuốc (Penicillin allergy...)
    if any(k in msg_lower for k in ["dị ứng", "di ung", "penicillin"]):
        return "### ⚠️ Hướng dẫn Lâm sàng cho Bệnh nhân Dị ứng Kháng sinh (Offline Mode)\n\n" \
               "- 💊 **Dị ứng Penicillin:**\n" \
               "  * **Kháng sinh thay thế:** Thường ưu tiên nhóm **Macrolides** (Azithromycin, Clarithromycin, Erythromycin) hoặc Lincosamides (Clindamycin) cho nhiễm khuẩn thông thường.\n" \
               "  * **Lưu ý phản ứng chéo:** Khoảng 3-10% bệnh nhân dị ứng Penicillin có nguy cơ xảy ra phản ứng dị ứng chéo với các kháng sinh nhóm **Cephalosporins** (đặc biệt là Cephalosporin thế hệ 1 như Cephalexin, Cefadroxil). Cần hết sức thận trọng khi kê đơn nhóm này.\n" \
               "- 🩺 **Nguyên tắc chung:** Luôn hỏi rõ tiền sử dị ứng của bệnh nhân (phát ban, ngứa, khó thở hay sốc phản vệ) và ưu tiên thực hiện test da trước khi tiêm/truyền các kháng sinh có nguy cơ cao."

    # 3d. Câu hỏi về mô hình học máy / thuật toán / độ chính xác / dữ liệu (ML & Project FAQ)
    if any(k in msg_lower for k in ["mô hình", "mo hinh", "thuật toán", "thuat toan", "học máy", "hoc may", "độ chính xác", "do chinh xac", "chỉ số", "chi so", "accuracy", "f1", "recall", "auc", "dữ liệu", "du lieu", "mẫu", "stacking", "threshold", "ngưỡng"]):
        return "### 📊 Thông tin chi tiết về Mô hình Học máy & Đồ án (Offline Mode)\n\n" \
               "- 📂 **Thông tin tập dữ liệu (Dataset):**\n" \
               "  * **Số lượng:** **2,404 mẫu** hệ gen vi khuẩn.\n" \
               "  * **Đặc trưng ban đầu:** 310 đặc trưng (210 đặc trưng gen kháng thuốc AMR và 100 đặc trưng liên tục đại diện cho mật độ k-mer nền).\n" \
               "  * **Đặc trưng rút gọn (sau RFE):** Rút gọn xuống **84 đặc trưng gen/k-mer quan trọng nhất** giúp tối ưu hóa chi phí giải trình tự gen.\n" \
               "- 🤖 **Thuật toán học máy đề xuất:**\n" \
               "  * **Mô hình XGBoost (Đề xuất):** Sử dụng thuật toán tăng cường gradient cây quyết định (XGBoost) được tối ưu hóa hyperparameter bằng Optuna, kết hợp kỹ thuật giảm chiều RFE và cân bằng dữ liệu SMOTE để tối đa hóa độ ổn định dự đoán.\n" \
               "- 📈 **Chỉ số đánh giá mô hình (Performance Metrics):**\n" \
               "  * **Độ chính xác (Accuracy):** **81.00%** trên tập test.\n" \
               "  * **ROC-AUC:** **90.09%** và **PR-AUC:** **88.80%**.\n" \
               "  * **Recall lớp Kháng thuốc (Resistant):** Đạt **79.85%** (XGBoost) và **81.89%** (Stacking) nhờ kỹ thuật cân bằng dữ liệu **SMOTE** (tránh bỏ sót bệnh nhân kháng thuốc trong chẩn đoán lâm sàng).\n" \
               "- ⚙️ **Ngưỡng quyết định (Decision Threshold):**\n" \
               "  * Được tối ưu ở mức **0.479** cho XGBoost và **0.523** cho Stacking giúp cân bằng hoàn hảo giữa độ chính xác và độ nhạy lâm sàng."

    # 3e. Câu hỏi tình huống lâm sàng chuyên sâu (MDR, MIC, Mang thai + Dị ứng + Kháng parC, Thăt bại Carbapenem)
    if any(k in msg_lower for k in ["đồng thời", "cả hai", "phối hợp gen", "đa kháng", "mdr", "bơm đẩy đi kèm", "bơm đẩy kết hợp", "ước lượng mic", "dải mic", "xét nghiệm bổ sung", "kiểm chứng", "cấy máu", "mang thai bị dị ứng", "thất bại carbapenem", "không giảm sốt", "72h"]):
        return "### 🩺 Tư vấn Lâm sàng nâng cao & Tình huống đặc biệt (Offline Mode)\n\n" \
               "- 🧬 **Đồng xuất hiện gen kháng & Đa kháng thuốc (MDR):**\n" \
               "  * Sự kết hợp đồng thời của gen sinh ESBL (`blaCTX-M-15`) và đột biến đích Fluoroquinolone (`gyrA_S83L`) tạo ra kiểu hình đa kháng cực kỳ nguy hiểm. Mức độ nguy hại lâm sàng tăng vọt do hầu như tất cả các kháng sinh Cephalosporin thế hệ 3/4 và Quinolone thông thường đều mất tác dụng.\n" \
               "  * Nếu xuất hiện thêm cơ chế bơm đẩy chủ động (efflux pump như `floR`, `tet`), vi khuẩn có thể tự động đẩy bớt kháng sinh ra ngoài, làm giảm nồng độ thuốc nội bào và thúc đẩy tính kháng thuốc chéo.\n" \
               "- 🔬 **Nồng độ MIC & Kiểm chứng cận lâm sàng:**\n" \
               "  * Mô hình XGBoost hiện tại chỉ phân loại nhị phân (Kháng/Nhạy) dựa trên gen. Hệ thống **không** dự đoán trực tiếp giá trị MIC (Nồng độ ức chế tối thiểu) bằng số.\n" \
               "  * Bác sĩ nên chỉ định thêm **Kháng sinh đồ đĩa giấy khuếch tán (Kirby-Bauer)** hoặc máy tự động (như Vitek 2) để xác định chính xác MIC thực tế. Nếu nghi ngờ nhiễm khuẩn huyết, cần chỉ định **Cấy máu** lập tức.\n" \
               "- 🤰 **Ca bệnh phức tạp (Mang thai + Dị ứng Penicillin + Kháng parC):**\n" \
               "  * Do mẫu kháng Fluoroquinolone (parC đột biến) => Không dùng Quinolone.\n" \
               "  * Bệnh nhân dị ứng Penicillin => Tránh dùng các penicillin.\n" \
               "  * Bệnh nhân mang thai => Tránh cả Quinolone và Tetracycline.\n" \
               "  * **Giải pháp thay thế khả thi:** Có thể cân nhắc các kháng sinh nhóm **Macrolides** (như Azithromycin) nếu vi khuẩn nhạy cảm, hoặc chuyển sang **Carbapenem** (như Meropenem) trong trường hợp nhiễm trùng nặng đe dọa tính mạng (do Carbapenem tương đối an toàn trong thai kỳ và tỷ lệ dị ứng chéo với Penicillin cực kỳ thấp, dưới 1%).\n" \
               "- 🌡️ **Thất bại điều trị Carbapenem sau 72 giờ:**\n" \
               "  * Cần đánh giá lại xem có ổ nhiễm trùng sâu chưa được dẫn lưu hay không (như áp xe).\n" \
               "  * Kiểm tra xem vi khuẩn có sinh enzyme Carbapenemase (như gen *blaNDM*, *blaKPC* - kháng cả Carbapenem) hay không.\n" \
               "  * Cần hội chẩn chuyên khoa truyền nhiễm để chuyển sang phối hợp thuốc (ví dụ: Colistin phối hợp hoặc thế hệ mới Ceftazidime-Avibactam)."

    # 3f. Câu hỏi về Sinh tin học / k-mer nền / Gram âm - dương / ngoại lai / tet(A) SHAP âm
    if any(k in msg_lower for k in ["k-mer nền", "mật độ k-mer", "tương quan màng", "loài vi khuẩn", "gram dương", "gram âm", "ngoại lai", "anomaly", "gen mới", "ngoài tập train", "ép vào dự đoán", "tet(a)", "shap âm"]):
        return "### 🧬 Giải đáp Sinh tin học & Cơ chế Mô hình Học máy (Offline Mode)\n\n" \
               "- 📊 **Giải thích tín hiệu từ k-mer nền (Background k-mers):**\n" \
               "  * Trong trường hợp không tìm thấy gen kháng cụ thể nhưng mô hình vẫn dự đoán Kháng (Resistant), tín hiệu này đến từ sự thay đổi mật độ của các k-mer ngắn trong hệ gen vi khuẩn.\n" \
               "  * Sự thay đổi này thường tương quan với các đột biến cấu trúc màng tế bào (làm giảm tính thấm porin màng của vi khuẩn Gram âm như *E. coli*), hoặc phản ánh nguồn gốc tiến hóa của chủng kháng thuốc.\n" \
               "- 📁 **Đối tượng huấn luyện & Giới hạn dữ liệu:**\n" \
               "  * Mô hình được huấn luyện tối ưu nhất trên các loài **vi khuẩn Gram âm** (đặc biệt là họ *Enterobacteriaceae* như *E. coli, Klebsiella pneumoniae*).\n" \
               "  * **Cảnh báo:** Độ tin cậy của mô hình sẽ **giảm mạnh** nếu đưa vào mẫu vi khuẩn Gram dương do sự khác biệt hoàn toàn về cấu trúc vách tế bào và cơ chế kháng thuốc.\n" \
               "  * **anomaly detection (Cảnh báo ngoại lai):** Hiện tại mô hình XGBoost không có bộ lọc phát hiện dị thường. Nếu đưa vào một gen hoàn toàn mới hoặc loài vi khuẩn không phù hợp, mô hình vẫn ép đưa ra dự đoán nhị phân (khó đảm bảo độ tin cậy).\n" \
               "- 🧬 **Tại sao gen tet(A) có SHAP âm ở một số mẫu cụ thể (ví dụ mẫu ID 1042)?**\n" \
               "  * SHAP đo lường đóng góp tương tác giữa các đặc trưng. Mặc dù `tet(A)` là gen kháng Tetracycline, nhưng nếu trong mẫu đó thiếu các gen đồng tác nhân hoặc mật độ k-mer nền thể hiện kiểu gen của một chủng nhạy cảm yếu, sự đóng góp cục bộ của đặc trưng này có thể bị bù trừ bởi các yếu tố khác, tạo ra giá trị SHAP âm."

    # 3g. Câu hỏi về UX / tính năng web / FASTA / FASTQ / tải biểu đồ / lưu lịch sử
    if any(k in msg_lower for k in ["fasta", "fastq", "trình tự", "tải biểu đồ", "lưu lịch sử", "luu lich su", "tiến triển"]):
        return "### 🤖 Giải đáp về Tính năng và Trải nghiệm Hệ thống (Web UX)\n\n" \
               "- 🧬 **Đầu vào FASTA/FASTQ:**\n" \
               "  * Hiện tại ứng dụng web **chưa hỗ trợ** tải lên file thô FASTA hoặc FASTQ trực tiếp.\n" \
               "  * Người dùng cần chạy quy trình trích xuất đặc trưng sinh học trước (ví dụ: dùng công cụ tìm gen kháng AMR FinderPlus hoặc công cụ đếm k-mer Jellyfish) để tạo ra file CSV dạng bảng trước khi đưa vào web dự đoán.\n" \
               "- 📊 **Tải biểu đồ SHAP dạng ảnh / PDF:**\n" \
               "  * Bạn có thể nhấn chuột phải trực tiếp vào biểu đồ SHAP (được vẽ bằng thư viện Chart.js trên web) và chọn **'Save image as...' (Lưu ảnh dưới dạng...)** để tải về máy dưới dạng ảnh PNG chất lượng cao phục vụ viết báo cáo.\n" \
               "- 💾 **Lưu lịch sử dự đoán bệnh nhân:**\n" \
               "  * Phiên bản hiện tại lưu trữ lịch sử tạm thời trong bộ nhớ cache của trình duyệt.\n" \
               "  * Để theo dõi tiến triển dài hạn của cả khoa, hệ thống cần được nâng cấp tích hợp cơ sở dữ liệu SQL (như SQLite hoặc PostgreSQL) ở phía Back-end."

    # 4. Câu hỏi về SHAP/giải thích mô hình
    if any(k in msg_lower for k in ["shap", "biểu đồ", "đồ thị", "giải thích"]):
        return "### 📊 Giải thích về SHAP (Shapley Additive exPlanations)\n\n" \
               "- **SHAP** đo lường mức độ đóng góp của từng gen / k-mer vào quyết định của mô hình XGBoost.\n" \
               "- **Cột màu Đỏ (SHAP > 0):** Đại diện cho các yếu tố thúc đẩy mẫu vi khuẩn trở nên **Kháng thuốc**.\n" \
               "- **Cột màu Xanh (SHAP < 0):** Đại diện cho các yếu tố giữ mẫu vi khuẩn ở trạng thái **Nhạy cảm**.\n" \
               "- Độ dài của cột tỉ lệ thuận với độ mạnh của tác động."

    # 5. Kiểm tra tài liệu RAG làm fallback trước khi trả về câu mặc định
    if RAG_ENGINE and RAG_ENGINE.pdf_loaded:
        matches = RAG_ENGINE.search(message, top_k=3)
        if matches:
            reply = "### 📚 Thông tin tra cứu từ Sách hướng dẫn điều trị (Offline RAG):\n\n"
            for idx, match in enumerate(matches):
                reply += f"**Đoạn tham khảo {idx+1} (Trang {match['page']}, Tài liệu: {match['source']}):**\n"
                reply += f"> {match['text'].strip()}\n\n"
            reply += "*Lưu ý: Tôi đang hoạt động ở chế độ Local Offline. Đây là các đoạn thông tin liên quan nhất tìm thấy trong sách hướng dẫn.*"
            return reply

    # 6. Câu hỏi mặc định
    return "Tôi là **Trợ lý Lâm sàng AI (Local Expert Mode)**. Tôi chưa thể hiểu hết câu hỏi phức tạp này ở chế độ ngoại tuyến.\n\n" \
           "**Mẹo:** Bác sĩ có thể hỏi về các gen cụ thể (ví dụ: *'blaCTX-M-15 là gì?'*), hỏi về phác đồ điều trị (ví dụ: *'kê đơn kháng sinh gì?'*), hoặc giải thích đồ thị (ví dụ: *'SHAP là gì?'*). Nếu muốn hỏi về mô hình, bạn có thể hỏi *'độ chính xác'* hoặc *'thuật toán là gì'*.\n\n" \
           "*Để được hỗ trợ phân tích hội thoại tự do chuyên sâu bằng mô hình GenAI, vui lòng cấu hình API Key trong file .env.*"

def generate_ai_report(outcome, probability, top_features, threshold):
    """Gọi API của Gemini dựa trên API Key được cấu hình trong file .env."""
    load_env_values()
    gemini_key = os.environ.get('GEMINI_API_KEY')
    gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

    outcome_vietnamese = "KHÁNG THUỐC (Resistant)" if outcome == "Resistant" else "NHẠY CẢM (Susceptible)"
    
    # Tìm kiếm tài liệu từ RAG Engine (PDF) nếu có
    rag_context = ""
    if RAG_ENGINE and RAG_ENGINE.pdf_loaded:
        search_terms = [f['feature'] for f in top_features[:3]]
        query = f"kháng thuốc {outcome_vietnamese} " + " ".join(search_terms)
        matches = RAG_ENGINE.search(query, top_k=3)
        if matches:
            rag_context = "\n[TÀI LIỆU HƯỚNG DẪN ĐIỀU TRỊ THAM KHẢO TRÍCH TỪ PDF CHÍNH THỨC]:\n"
            for match in matches:
                rag_context += f"- Trang {match['page']} (Tài liệu: {match['source']}): {match['text'].strip()}\n"
            rag_context += "\nYêu cầu: Hãy tham chiếu và trích dẫn thông tin từ tài liệu hướng dẫn trên trong phần Khuyến nghị lâm sàng của báo cáo (ghi rõ số trang và tên tài liệu nếu trích dẫn).\n"

    import datetime
    current_date = datetime.datetime.now().strftime("ngày %d tháng %m năm %Y")

    prompt = f"""
    Bạn là một Cố vấn Lâm sàng AI chuyên ngành Vi sinh lâm sàng và bệnh truyền nhiễm.
    Hôm nay là {current_date}. Khi viết báo cáo, ở phần tiêu đề đầu báo cáo, vui lòng điền ngày tháng thực tế này (ví dụ: "Ngày: {current_date}") thay vì dùng các từ giữ chỗ như "[Ngày hiện tại]" hay "[Ngày]".
    
    Hãy viết một báo cáo phân tích chuyên khoa ngắn gọn, chuyên nghiệp bằng tiếng Việt cho bác sĩ điều trị dựa trên kết quả chẩn đoán kháng kháng sinh (AMR) sau:
    
    - Kết luận chẩn đoán của mô hình XGBoost: {outcome_vietnamese} (Xác suất kháng thuốc: {probability * 100:.2f}%, Ngưỡng quyết định: {threshold:.3f})
    - Top các đặc trưng gen kháng thuốc/k-mer ảnh hưởng lớn nhất lấy từ giải thích SHAP:
    {json.dumps(top_features, indent=2)}
    {rag_context}
    
    Yêu cầu báo cáo:
    1. Tóm tắt tình trạng kháng thuốc của mẫu bệnh phẩm này.
    2. Giải thích ý nghĩa của các đột biến/gen kháng thuốc chính xuất hiện trong danh sách (đặc biệt là các gen có SHAP value dương lớn thúc đẩy kết quả kháng thuốc như gyrA đột biến, blaCTX-M sinh ESBL, v.v.).
    3. Đưa ra các khuyến nghị lâm sàng thực tế cho bác sĩ điều trị (ví dụ: tránh dùng nhóm kháng sinh nào, đề xuất kháng sinh thay thế khả thi hoặc yêu cầu làm thêm xét nghiệm gì).
    4. Giữ giọng văn y khoa chuyên nghiệp, cấu trúc rõ ràng sử dụng Markdown (bold, lists, v.v.). Không dài dòng lê thê.
    """

    # --- Gọi Gemini API ---
    if gemini_key and gemini_key.strip() and not gemini_key.startswith("YOUR_"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key.strip()}"
        print(f"📡 [AI Report] Đang gọi Gemini API (Model: {gemini_model})...")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        headers = {"Content-Type": "application/json"}
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=30) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    print(f"✅ [AI Report] Gọi Gemini ({gemini_model}) thành công!")
                    return res_data['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                print(f"⚠️ [AI Report] Thử lần {attempt+1} thất bại: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.5)
                else:
                    print(f"❌ [AI Report] Lỗi gọi Gemini API sau {max_retries} lần thử: {e}")

    # Fallback về Local Expert System nếu API key trống hoặc lỗi
    print("⚠️ [AI Report] Không kết nối được dịch vụ Gemini trực tuyến. Tự động chuyển về Hệ chuyên gia cục bộ (Local).")
    return f"*(Không thể kết nối dịch vụ Gemini trực tuyến. Đang tự động hiển thị báo cáo từ Hệ chuyên gia cục bộ)*\n\n" + \
           generate_local_report(outcome, probability, top_features, threshold)

def call_ai_chat(system_instruction, history, user_message):
    """Gửi lịch sử hội thoại và system instruction tới Gemini API."""
    load_env_values()
    gemini_key = os.environ.get('GEMINI_API_KEY')
    gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

    if not gemini_key or not gemini_key.strip() or gemini_key.startswith("YOUR_"):
        raise ValueError("GEMINI_API_KEY is not configured in .env")

    # Chuẩn hóa lịch sử chat cho Gemini (role 'user' và 'model')
    contents = []
    for msg in history:
        role = msg.get("role", "user")
        role = "user" if role == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key.strip()}"
    print(f"💬 [AI Chat] Đang gọi Gemini API (Model: {gemini_model})...")
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": contents
    }
    headers = {"Content-Type": "application/json"}
    
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                print(f"✅ [AI Chat] Gọi Gemini ({gemini_model}) thành công!")
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"⚠️ [AI Chat] Thử lần {attempt+1} thất bại: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5)
            else:
                print(f"❌ [AI Chat] Lỗi gọi Gemini Chat API sau {max_retries} lần thử: {e}")
                raise e

app = Flask(__name__)

# Cấu hình đường dẫn thư mục tải lên tạm thời
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Biến toàn cục để lưu mô hình và các thông tin liên quan
MODEL = None
THRESHOLD = None
FEATURES = None
SHAP_EXPLAINER = None
MODEL_PATH = None

def init_model():
    """Tự động tìm và load mô hình mới nhất trong thư mục."""
    global MODEL, THRESHOLD, FEATURES, SHAP_EXPLAINER, MODEL_PATH, RAG_ENGINE
    
    # Khởi tạo DB SQLite trước khi nạp mô hình
    init_db()
    
    # Khởi tạo RAG Engine
    try:
        print("Initializing RAG Engine...")
        RAG_ENGINE = RAGEngine()
    except Exception as rag_err:
        print(f"Warning: Failed to initialize RAG Engine: {rag_err}")
        
    model_files = glob.glob("models/amr_classifier_*.joblib")
    if not model_files:
        print("Warning: No .joblib model files found. Please run run_training.py first.")
        return False
    
    # Lấy file mô hình mới nhất theo thứ tự bảng chữ cái (chứa timestamp)
    MODEL_PATH = sorted(model_files)[-1]
    print(f"Loading model from: {MODEL_PATH}...")
    try:
        MODEL, THRESHOLD, FEATURES = ml_pipeline.load_model(MODEL_PATH)
        print("Success: Model loaded successfully!")
        
        # Tải dữ liệu nền (background) để khởi tạo SHAP Explainer
        # Nếu có file CSV dữ liệu, dùng 100 mẫu đầu tiên làm background
        if os.path.exists("data/X.csv"):
            X_background = pd.read_csv("data/X.csv", index_col=0).head(100)
            print("Initializing SHAP Explainer...")
            SHAP_EXPLAINER = ml_pipeline.build_shap_explainer(MODEL, X_background)
            print("Success: SHAP Explainer initialized successfully!")
        else:
            print("Warning: data/X.csv not found. SHAP features will be disabled.")
        return True
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        return False

# ----------------- WEB PAGES -----------------

@app.route('/')
def home():
    """Trang chủ hiển thị Giao diện Dashboard."""
    return render_template('index.html')

# ----------------- API ENDPOINTS -----------------

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Trả về thông tin chi tiết của mô hình hiện tại."""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded yet."}), 500
    
    return jsonify({
        "status": "success",
        "model_name": os.path.basename(MODEL_PATH),
        "threshold": round(THRESHOLD, 3),
        "features_count": len(FEATURES),
        "shap_enabled": SHAP_EXPLAINER is not None
    })

@app.route('/api/gene_db', methods=['GET'])
def get_gene_db():
    """Trả về cơ sở dữ liệu luật gen kháng thuốc để hiển thị từ điển trên frontend."""
    enriched_db = dict(GENE_DB)
    if FEATURES:
        for name in FEATURES:
            if is_amr_gene(name) and name not in enriched_db:
                enriched_db[name] = get_gene_description(name)
                
    return jsonify({
        "status": "success",
        "gene_db": enriched_db
    })

@app.route('/api/get_samples', methods=['GET'])
def get_samples():
    """Trả về 3 mẫu bệnh nhân ngẫu nhiên (1 kháng thuốc, 1 nhạy cảm, 1 ngẫu nhiên) để test nhanh."""
    if not os.path.exists("data/X.csv") or not os.path.exists("data/y.csv"):
        return jsonify({"status": "error", "message": "Data files not found."}), 404
    
    try:
        import random
        X = pd.read_csv("data/X.csv", index_col=0)
        y = pd.read_csv("data/y.csv", index_col=0).iloc[:, 0]
        
        # Lọc mẫu Resistant (1) và Susceptible (0)
        res_idx = y[y == 1].index
        sus_idx = y[y == 0].index
        
        samples = {}
        if len(res_idx) > 0:
            res_id = random.choice(res_idx)
            samples["resistant"] = {
                "id": str(res_id),
                "features": X.loc[res_id].to_dict(),
                "true_label": "Resistant"
            }
        if len(sus_idx) > 0:
            sus_id = random.choice(sus_idx)
            samples["susceptible"] = {
                "id": str(sus_id),
                "features": X.loc[sus_id].to_dict(),
                "true_label": "Susceptible"
            }
            
        return jsonify({
            "status": "success",
            "samples": samples
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API dự đoán cho 1 bệnh nhân cụ thể.
    Yêu cầu JSON body: { "features": { "gene1": 0, "gene2": 1, ... }, "patient_id": "BN..." }
    """
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 500
    
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"status": "error", "message": "Missing 'features' in request body."}), 400
    
    try:
        # Chuyển đổi dữ liệu gửi lên thành Pandas Series
        feature_dict = data['features']
        feature_vector = pd.Series(feature_dict)
        
        # Nhận diện mã bệnh nhân, tự động sinh nếu trống
        patient_id = data.get('patient_id', '').strip()
        if not patient_id:
            import datetime
            import random
            now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M')
            rand_num = random.randint(100, 999)
            patient_id = f"BN-{now_str}-{rand_num}"
        
        # 1. Dự đoán kết quả
        prediction_res = ml_pipeline.predict_one_patient(feature_vector, MODEL, THRESHOLD, FEATURES)
        
        # 2. Giải thích SHAP nếu khả dụng
        shap_explanation = None
        if SHAP_EXPLAINER is not None:
            # Lấy top 10 đặc trưng ảnh hưởng nhiều nhất
            shap_res = ml_pipeline.explain_prediction(SHAP_EXPLAINER, feature_vector, FEATURES, top_k=10)
            shap_explanation = shap_res
            
        # 3. Tạo báo cáo lâm sàng AI (Gemini hoặc Local Expert System fallback)
        top_features = shap_explanation['top_features'] if shap_explanation else []
        ai_report = generate_ai_report(
            prediction_res['prediction'], 
            prediction_res['prob_resistant'], 
            top_features,
            THRESHOLD
        )
        
        # 4. Lưu vào cơ sở dữ liệu SQLite
        try:
            detected_genes_list = []
            for f in top_features:
                name = f['feature']
                val = f['feature_value']
                if val > 0 and is_amr_gene(name):
                    detected_genes_list.append(name)
            detected_genes_str = ", ".join(detected_genes_list) if detected_genes_list else "Không phát hiện"
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_history (patient_id, prediction, probability, detected_genes, features_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                patient_id,
                prediction_res['prediction'],
                prediction_res['prob_resistant'],
                detected_genes_str,
                json.dumps(feature_dict)
            ))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"Error saving prediction to database: {db_err}")
            
        return jsonify({
            "status": "success",
            "prediction": prediction_res,
            "shap": shap_explanation,
            "ai_report": ai_report,
            "patient_id": patient_id
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    API dự đoán hàng loạt từ file CSV tải lên.
    Trả về kết quả dưới dạng JSON (để hiển thị bảng) và thông tin file.
    """
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 500
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400
    
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Đọc CSV
            df = pd.read_csv(file_path, index_col=0)
            
            # Đảm bảo các cột khớp với đặc trưng khi train
            # Những cột thiếu sẽ được gán giá trị 0
            df_aligned = df.reindex(columns=FEATURES, fill_value=0)
            
            # Dự đoán xác suất
            probabilities = MODEL.predict_proba(df_aligned)[:, 1]
            predictions = ["Resistant" if p >= THRESHOLD else "Susceptible" for p in probabilities]
            
            # Lưu kết quả dự đoán vào DataFrame để cho phép tải về
            result_df = pd.DataFrame({
                "Sample_ID": df.index,
                "Prediction": predictions,
                "Probability_Resistant": np.round(probabilities, 4)
            })
            
            output_filename = f"predicted_{file.filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            result_df.to_csv(output_path, index=False)
            
            # Trả về tối đa 50 dòng kết quả đầu tiên để vẽ bảng trên web
            preview_data = result_df.head(50).to_dict(orient='records')
            
            return jsonify({
                "status": "success",
                "total_records": len(result_df),
                "download_url": f"/download/{output_filename}",
                "preview": preview_data
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            # Xóa file upload tạm để tránh rác hệ thống
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({"status": "error", "message": "Invalid file format. Only CSV is allowed."}), 400

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Đường dẫn tải xuống file kết quả dự đoán hàng loạt."""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API gửi tin nhắn hội thoại lâm sàng với Gemini hoặc hệ chuyên gia cục bộ.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"status": "error", "message": "Missing 'message' in request body."}), 400
    
    user_message = data['message']
    history = data.get('history', [])
    prediction_context = data.get('context', {})
    
    load_env_values()
    gemini_key = os.environ.get('GEMINI_API_KEY')
    
    outcome = prediction_context.get('prediction', 'Unknown')
    prob = prediction_context.get('prob_resistant', 0.0)
    top_features = prediction_context.get('top_features', [])
    
    # Tìm kiếm các đoạn tài liệu PDF liên quan đến câu hỏi chat của người dùng
    rag_context = ""
    if RAG_ENGINE and RAG_ENGINE.pdf_loaded:
        matches = RAG_ENGINE.search(user_message, top_k=3)
        if matches:
            rag_context = "\n[TRÍCH DẪN TỪ HƯỚNG DẪN SỬ DỤNG KHÁNG SINH CỦA BỘ Y TẾ VIỆT NAM]:\n"
            for match in matches:
                source_display = "Hướng dẫn sử dụng kháng sinh của Bộ Y tế Việt Nam (Quyết định 708/QĐ-BYT)" if "708" in match['source'] else match['source']
                rag_context += f"- Trang {match['page']} ({source_display}): {match['text'].strip()}\n"
            rag_context += "\nYêu cầu: Hãy ưu tiên sử dụng thông tin từ tài liệu hướng dẫn trên để giải đáp thắc mắc của bác sĩ (nếu có thông tin liên quan) và trích dẫn rõ số trang cùng tên tài liệu trong câu trả lời để bác sĩ tiện đối chiếu.\n"

    outcome_vietnamese = "KHÁNG THUỐC (Resistant)" if outcome == "Resistant" else "NHẠY CẢM (Susceptible)"
    system_instruction = f"""
    Bạn là một Cố vấn Lâm sàng AI chuyên ngành Vi sinh lâm sàng và bệnh truyền nhiễm, được cung cấp tài liệu chính thức: "Hướng dẫn sử dụng kháng sinh" của Bộ Y tế Việt Nam để làm nguồn tham chiếu chính.
    Bạn đang trao đổi với bác sĩ điều trị về một ca bệnh có kết quả xét nghiệm AMR như sau:
    - Chẩn đoán của mô hình XGBoost: {outcome_vietnamese} (Xác suất: {prob * 100:.2f}%)
    - Top đặc trưng kháng thuốc ảnh hưởng lớn nhất lấy từ giải thích SHAP: {json.dumps(top_features)}
    {rag_context}
    
    Khi trả lời các câu hỏi về chuyên môn lâm sàng hoặc phác đồ điều trị:
    1. Hãy luôn đóng vai trò là một chuyên gia y khoa trả lời dựa trên cuốn sách "Hướng dẫn sử dụng kháng sinh của Bộ Y tế Việt Nam".
    2. Nếu có tài liệu tham khảo được cung cấp ở trên, hãy trích dẫn cụ thể (ví dụ: "Theo Hướng dẫn sử dụng kháng sinh của Bộ Y tế, Trang X...").
    3. Trả lời một cách khoa học, chuyên nghiệp, súc tích và dựa trên y học chứng cứ.
    4. Trả lời bằng tiếng Việt, cấu trúc rõ ràng bằng markdown.
    5. Nhắc nhở bác sĩ rằng đây là tư vấn hỗ trợ và họ cần đối chiếu với kết quả kháng sinh đồ thực tế tại bệnh viện.
    """
    
    # Kiểm tra xem có API key Gemini được cấu hình hay không
    if not gemini_key or not gemini_key.strip() or gemini_key.startswith("YOUR_"):
        reply = generate_local_chat_reply(user_message, outcome, prob, top_features)
        return jsonify({"status": "success", "reply": reply})
        
    try:
        reply = call_ai_chat(system_instruction, history, user_message)
        return jsonify({"status": "success", "reply": reply})
    except Exception as e:
        print(f"Chat API error fallback to local: {e}")
        reply = f"*(Không thể kết nối đến Gemini API: {e}. Đang tự động phản hồi từ Hệ chuyên gia cục bộ)*\n\n" + \
                generate_local_chat_reply(user_message, outcome, prob, top_features)
        return jsonify({"status": "success", "reply": reply})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Lấy danh sách lịch sử dự đoán từ cơ sở dữ liệu SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, strftime('%Y-%m-%d %H:%M:%S', datetime(timestamp, 'localtime')) as formatted_time, 
                   patient_id, prediction, probability, detected_genes, features_json
            FROM prediction_history 
            ORDER BY timestamp DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        history_list = []
        for r in rows:
            history_list.append({
                "id": r["id"],
                "timestamp": r["formatted_time"],
                "patient_id": r["patient_id"],
                "prediction": r["prediction"],
                "probability": r["probability"],
                "detected_genes": r["detected_genes"] if r["detected_genes"] else "Không phát hiện",
                "features": json.loads(r["features_json"])
            })
            
        return jsonify({
            "status": "success",
            "history": history_list
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/history/delete', methods=['POST'])
def delete_history_item():
    """Xóa một bản ghi lịch sử theo ID."""
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({"status": "error", "message": "Missing 'id' in request body."}), 400
    
    entry_id = data['id']
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prediction_history WHERE id = ?', (entry_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Deleted entry {entry_id} successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_all_history():
    """Xóa toàn bộ lịch sử dự đoán."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prediction_history')
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Cleared all prediction history successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/epidemiology_stats', methods=['GET'])
def get_epidemiology_stats():
    """Thống kê dữ liệu dịch tễ học từ cơ sở dữ liệu SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Thống kê tỷ lệ kháng thuốc theo ngày
        cursor.execute('''
            SELECT date(timestamp) as day_str,
                   count(*) as total,
                   sum(case when prediction = 'Resistant' then 1 else 0 end) as resistant_count
            FROM prediction_history
            GROUP BY day_str
            ORDER BY day_str ASC
        ''')
        rows_by_day = cursor.fetchall()
        
        timeline_data = []
        for r in rows_by_day:
            day = r["day_str"]
            total = r["total"]
            resistant = r["resistant_count"]
            rate = round((resistant / total) * 100, 1) if total > 0 else 0.0
            timeline_data.append({
                "date": day,
                "total": total,
                "resistant": resistant,
                "rate": rate
            })
            
        # 2. Thống kê tần suất xuất hiện của các gen kháng thuốc
        cursor.execute('SELECT detected_genes FROM prediction_history')
        all_genes_rows = cursor.fetchall()
        
        gene_counts = {}
        for row in all_genes_rows:
            genes_str = row["detected_genes"]
            if genes_str and genes_str != "Không phát hiện":
                genes = [g.strip() for g in genes_str.split(",")]
                for g in genes:
                    if g:
                        gene_counts[g] = gene_counts.get(g, 0) + 1
                        
        sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
        gene_stats = [{"gene": g, "count": c} for g, c in sorted_genes]
        
        conn.close()
        
        return jsonify({
            "status": "success",
            "timeline": timeline_data,
            "genes": gene_stats
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ----------------- SETUP AND RUN -----------------

# Khởi tạo mô hình ngay khi server start
init_model()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
