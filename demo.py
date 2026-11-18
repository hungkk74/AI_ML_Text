"""
Demo: nhập bình luận tiếng Việt -> xem dự đoán của 5 mô hình đã huấn luyện.
Cần chạy train_compare.py ít nhất 1 lần trước (để có thư mục models/).

python demo.py
"""
import glob
import sys

import joblib

from train_compare import clean_text, TOKENIZER_TAG   # dùng chung bước tiền xử lý

LABEL = {1: "TÍCH CỰC", 0: "TIÊU CỰC"}


def load_bundles():
    files = sorted(glob.glob("models/*.joblib"))
    if not files:
        sys.exit("Chưa có mô hình. Hãy chạy `python train_compare.py` trước.")
    bundles = [joblib.load(f) for f in files]
    for b in bundles:
        if b.get("tokenizer") != TOKENIZER_TAG:
            print(f"[!] '{b['title']}' huấn luyện với kiểu tách từ '{b.get('tokenizer')}' "
                  f"nhưng hiện đang dùng '{TOKENIZER_TAG}' -> kết quả có thể lệch.")
    return bundles


def main():
    bundles = load_bundles()
    print("Bộ dữ liệu đã nạp:")
    for i, b in enumerate(bundles, 1):
        print(f"  [{i}] {b['title']}")
    choice = input(f"Chọn bộ dữ liệu (1-{len(bundles)}, mặc định 1): ").strip()
    b = bundles[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(bundles) else bundles[0]
    print(f"\nĐang dùng: {b['title']}\nNhập bình luận (gõ 'exit' để thoát).\n")

    while True:
        text = input(">> ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue
        vec = b["tfidf"].transform([clean_text(text)])
        votes = {}
        for name, model in b["models"].items():
            votes[name] = int(model.predict(vec)[0])
            print(f"  {name:<20} {LABEL[votes[name]]}")
        pos = sum(votes.values())
        print(f"  -> Đa số: {LABEL[1] if pos > len(votes) / 2 else LABEL[0]} "
              f"({pos}/{len(votes)} mô hình chọn tích cực)\n")


if __name__ == "__main__":
    main()
