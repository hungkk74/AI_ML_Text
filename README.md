# Phân loại cảm xúc bình luận tiếng Việt

So sánh 5 thuật toán (MultinomialNB, ComplementNB, LogisticRegression, LinearSVM, RandomForest) trên TF-IDF, 2 bộ dữ liệu:

- **Bộ 1** – Vietnamese Sentiment Analyst (Kaggle, `linhlpv/vietnamese-sentiment-analyst`)
- **Bộ 2** – UIT-VSFC (HuggingFace, `uitnlp/vietnamese_students_feedback`)

## Cấu trúc

| File / thư mục | Mô tả |
|---|---|
| `train_compare.py` | Tải dữ liệu, tiền xử lý, huấn luyện, **lưu mô hình vào `models/`** |
| `train_versions.py` | Huấn luyện 5 mô hình × 3 phiên bản (Base/Tune 1/Tune 2) → `results/` (CSV, XLSX, ảnh) |
| `app.py` | Giao diện web Gradio để thử mô hình |
| `demo.py` | Demo dòng lệnh |
| `test/` | File bình luận mẫu để kiểm thử hàng loạt |
| `models/`, `cache/`, `results/` | Sinh ra khi chạy (mô hình, cache tách từ, kết quả) |

## Cài đặt

Yêu cầu Python 3.14 (theo `.venv` hiện tại), có Internet ở lần chạy đầu (tải dataset).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install scikit-learn pandas numpy underthesea datasets kagglehub matplotlib seaborn pyarrow openpyxl gradio joblib
```

> Bộ 1 tải qua `kagglehub`: có thể cần đăng nhập Kaggle (`KAGGLE_USERNAME`/`KAGGLE_KEY`) nếu dataset yêu cầu.

## Chạy

Thực hiện theo thứ tự (chạy tại thư mục gốc `D:\AI`):

```powershell
# 1. Huấn luyện + lưu mô hình (bắt buộc ít nhất 1 lần, tạo models/ds1.joblib, models/ds2.joblib)
python train_compare.py

# 2. (Tuỳ chọn) Bảng so sánh 3 phiên bản + biểu đồ -> results/
python train_versions.py

# 3a. Giao diện web -> http://127.0.0.1:7860 (tự mở trình duyệt)
python app.py

# 3b. Hoặc demo dòng lệnh (gõ "exit" để thoát)
python demo.py
```

Lần đầu `underthesea` tách từ khá chậm; kết quả được cache trong `cache/` nên các lần sau nhanh hơn.
Nếu đã có sẵn `models/*.joblib` thì có thể bỏ qua bước 1–2 và chạy thẳng `app.py`.

## Cách test

### 1. Thử một câu
- **Web:** mở `app.py`, chọn bộ dữ liệu, nhập câu (có sẵn ví dụ) → xem dự đoán của 5 mô hình và kết luận đa số.
- **CLI:** `python demo.py`, chọn bộ dữ liệu, nhập từng câu.

### 2. Kiểm thử hàng loạt bằng file trong `test/`
Trên giao diện web, tải file CSV/XLSX lên mục **"File CSV / Excel"** (có thể nhập thêm câu thủ công, mỗi dòng một câu). Kết quả hiển thị bảng chi tiết, thống kê theo mô hình / nhóm, và cho tải về `ket_qua_phan_loai.csv` / `.xlsx`.

| File | Số dòng | Cột | Ghi chú |
|---|---|---|---|
| `test_comments.csv` | 66 | `text, expected, category, domain` | Có nhãn kỳ vọng + nhóm |
| `test_muasam.csv` | 40 | `text, expected, category` | Mua sắm, có nhãn |
| `test_giaoduc.csv` | 40 | `text, expected, category` | Giáo dục, có nhãn |
| `test_cot_khac.csv` | 30 | `id, comment, rating, label` | Tên cột khác chuẩn (cột văn bản `comment`, nhãn `label`) |
| `test_khong_nhan.csv` | 30 | `text` | Không có nhãn → chỉ dự đoán |

Quy tắc đọc file:
- Cột văn bản được nhận diện theo tên: `text, sentence, comment, content, review, câu, bình luận` (không khớp thì lấy cột đầu tiên).
- Cột nhãn kỳ vọng: `expected, label, nhãn, kỳ vọng`; giá trị `TÍCH CỰC/POS/POSITIVE/1` hoặc `TIÊU CỰC/NEG/NEGATIVE/0`. Có nhãn thì tính được độ chính xác.
- Cột nhóm: `category, nhóm, group` → thống kê theo nhóm.

### 3. Đánh giá mô hình trên tập test gốc
`train_compare.py` và `train_versions.py` tự chia/đánh giá (Accuracy, Precision, Recall, Macro-F1, thời gian) và xuất:
- `results/results_summary.csv|xlsx` – bảng tổng hợp
- `results/data_audit.csv` – số mẫu sau mỗi bước làm sạch (loại trùng, rò rỉ train/test)
- `results/images/` – ma trận nhầm lẫn, biểu đồ cột, ảnh bảng

## Lỗi thường gặp
- `Chưa có mô hình...` → chưa chạy `python train_compare.py`.
- Cảnh báo `huấn luyện với kiểu tách từ ... hiện dùng ...` → mô hình train khi có/không có `underthesea` khác với hiện tại; cài `underthesea` và huấn luyện lại.
- Câu trả về cảnh báo "Không có từ nào trong từ vựng" → câu quá ngắn/ngoài từ vựng, kết quả không đáng tin.
