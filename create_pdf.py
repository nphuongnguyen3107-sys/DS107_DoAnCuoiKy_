# create_pdf.py
import os

def build_pdf():
    # Nội dung Trang 1
    p1_lines = [
        "HUONG DAN SU DUNG KHANG SINH CUA BO Y TE VIET NAM",
        "Chuong I: Phac do dieu tri viem phoi o tre em",
        "Huong dan lam sang: Su dung Ampicillin phoi hop voi Gentamicin lam lua chon dau tay.",
        "Doi voi nhiem khuan nang nghi ngo do vi khuan sinh ESBL hoac co gen blaCTX-M,",
        "khuyen cao su dung khang sinh nhom Carbapenem nhu Imipenem hoac Meropenem."
    ]
    
    # Nội dung Trang 2
    p2_lines = [
        "HUONG DAN DIEU TRI NHIEM TRUNG DUONG TIET NIEU",
        "Chuong II: Huong dan dieu tri nhiem trung tieu do vi khuan Gram am",
        "Khuyen cao lam sang: Lua chon dau tien va hieu qua nhat la Nitrofurantoin hoac Fosfomycin.",
        "Neu phat hien co dot bien gen gyrA hoac parC gay de khang khang sinh nhom Fluoroquinolone,",
        "tuyet doi tranh su dung cac khang sinh nhu Ciprofloxacin hoac Levofloxacin trong phac do."
    ]
    
    # Tạo stream định dạng PDF chuẩn (sử dụng 0 -20 Td để xuống dòng thay vì T*)
    def make_stream(lines):
        stream = "BT\n/F1 12 Tf\n50 750 Td\n"
        for i, line in enumerate(lines):
            if i > 0:
                stream += "0 -20 Td\n"
            stream += f"({line}) Tj\n"
        stream += "ET\n"
        return stream
    
    stream1 = make_stream(p1_lines)
    stream2 = make_stream(p2_lines)
    
    # Khởi tạo các phần của file PDF
    pdf_parts = {}
    
    pdf_parts[1] = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pdf_parts[2] = "2 0 obj\n<< /Type /Pages /Kids [ 3 0 R 5 0 R ] /Count 2 >>\nendobj\n"
    pdf_parts[3] = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [ 0 0 595 842 ] /Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n"
    pdf_parts[4] = f"4 0 obj\n<< /Length {len(stream1)} >>\nstream\n{stream1}endstream\nendobj\n"
    pdf_parts[5] = "5 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [ 0 0 595 842 ] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n"
    pdf_parts[6] = f"6 0 obj\n<< /Length {len(stream2)} >>\nstream\n{stream2}endstream\nendobj\n"
    pdf_parts[7] = "7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    
    # Ghi file và tính toán byte offset chính xác cho bảng xref
    header = "%PDF-1.4\n"
    offsets = {}
    
    current_offset = len(header)
    for obj_num in sorted(pdf_parts.keys()):
        offsets[obj_num] = current_offset
        current_offset += len(pdf_parts[obj_num])
        
    # Tạo bảng xref
    xref = "xref\n0 8\n0000000000 65535 f \n"
    for obj_num in sorted(pdf_parts.keys()):
        xref += f"{offsets[obj_num]:010d} 00000 n \n"
        
    start_xref = current_offset
    
    # Tạo trailer
    trailer = f"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF\n"
    
    # Kết hợp tất cả
    final_pdf_bytes = header.encode('utf-8')
    for obj_num in sorted(pdf_parts.keys()):
        final_pdf_bytes += pdf_parts[obj_num].encode('utf-8')
    final_pdf_bytes += xref.encode('utf-8')
    final_pdf_bytes += trailer.encode('utf-8')
    
    # Ghi ra file
    out_path = "huong_dan_su_dung_khang_sinh_708.pdf"
    with open(out_path, "wb") as f:
        f.write(final_pdf_bytes)
    print(f"Successfully created PDF: {out_path} ({len(final_pdf_bytes)} bytes)")

if __name__ == "__main__":
    build_pdf()
