"""
Giao diện web để thử các mô hình đã huấn luyện (5 mô hình, 2 bộ dữ liệu).
Cần chạy train_compare.py ít nhất 1 lần trước (để có thư mục models/).

    pip install gradio
    python app.py          -> tự mở trình duyệt tại http://127.0.0.1:7860
"""
import glob
import os
import tempfile

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from train_compare import clean_text, TOKENIZER_TAG

MODEL_DIR = "models"
LABEL = {1: "TÍCH CỰC", 0: "TIÊU CỰC"}
ORDER = ["MultinomialNB", "ComplementNB", "LogisticRegression", "LinearSVM", "RandomForest"]
EXAMPLES = [
    "Sản phẩm dùng rất tốt, giao hàng nhanh, sẽ ủng hộ tiếp",
    "Hàng kém chất lượng, dùng 2 hôm là hỏng, thất vọng quá",
    "Đẹp nhưng giao hàng chậm quá",
    "Không tệ như mình nghĩ",
    "Thầy dạy rất nhiệt tình, dễ hiểu",
    "Phòng học nóng, micro hay hỏng, môn này chán",
    "ok",
]


def load_bundles():
    out = {}
    for f in sorted(glob.glob(f"{MODEL_DIR}/*.joblib")):
        b = joblib.load(f)
        out[b["title"]] = b
        if b.get("tokenizer") != TOKENIZER_TAG:
            print(f"[!] '{b['title']}' huấn luyện với kiểu tách từ '{b.get('tokenizer')}', "
                  f"hiện dùng '{TOKENIZER_TAG}' -> kết quả có thể lệch.")
    return out


BUNDLES = load_bundles()
TITLES = list(BUNDLES)


# ----------------------------------------------------------------------------- logic
def run_models(bundle, text):
    """Trả về (danh sách kết quả từng mô hình, vector TF-IDF của câu)."""
    vec = bundle["tfidf"].transform([clean_text(text)])
    rows = []
    for name in ORDER:
        m = bundle["models"].get(name)
        if m is None:
            continue
        pred = int(m.predict(vec)[0])
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(vec)[0]
            cls = list(m.classes_)
            conf, p_pos = f"{p[cls.index(pred)]:.0%}", float(p[cls.index(1)])
        else:                                   # LinearSVC: không có xác suất, dùng margin
            conf, p_pos = f"margin {float(m.decision_function(vec)[0]):+.2f}", None
        rows.append({"name": name, "pred": pred, "conf": conf, "p_pos": p_pos})
    return rows, vec


def verdict(rows):
    n = len(rows)
    pos = sum(r["pred"] for r in rows)
    agree = max(pos, n - pos)
    label = LABEL[1] if pos > n - pos else LABEL[0]
    if agree == n:
        note = f"{n}/{n} mô hình đồng thuận"
    elif agree == n - 1:
        note = f"{agree}/{n} mô hình đồng ý, khá chắc chắn"
    else:
        note = f"chỉ {agree}/{n} mô hình đồng ý, **câu lưỡng lự / khó**"
    ps = [r["p_pos"] for r in rows if r["p_pos"] is not None]
    return label, agree, n, note, (float(np.mean(ps)) if ps else None)


def word_effects(model, name, vec):
    """Mức đẩy về 'tích cực' (dương) / 'tiêu cực' (âm) của từng từ trong câu, theo từng mô hình.
    Trả về (mảng cột đặc trưng, mảng hiệu ứng, mô tả cách tính)."""
    coo = vec.tocoo()
    cols, vals = coo.col, coo.data
    if name in ("LogisticRegression", "LinearSVM"):          # tuyến tính: trọng số x giá trị TF-IDF
        return cols, model.coef_[0][cols] * vals, "trọng số × TF-IDF"
    if name in ("MultinomialNB", "ComplementNB"):            # chênh lệch log-xác suất giữa 2 lớp
        cls = list(model.classes_)
        d = model.feature_log_prob_[cls.index(1)] - model.feature_log_prob_[cls.index(0)]
        return cols, d[cols] * vals, "chênh log-xác suất 2 lớp × TF-IDF"
    # RandomForest (phi tuyến): bỏ từng từ ra, xem xác suất tích cực đổi bao nhiêu
    import scipy.sparse as sp
    order = np.argsort(-vals)[:40]
    cols, vals = cols[order], vals[order]
    X = vec.tocsr()
    pi = list(model.classes_).index(1)
    base = model.predict_proba(X)[0][pi]
    mats = []
    for c in cols:
        r = X.copy().tolil()
        r[0, c] = 0
        mats.append(r.tocsr())
    without = model.predict_proba(sp.vstack(mats))[:, pi]
    return cols, base - without, "bỏ từng từ ra, đo thay đổi xác suất"


def explain(bundle, vec, top=3):
    """Bảng: mỗi mô hình dựa vào những từ nào để nghiêng về tích cực / tiêu cực."""
    if vec.nnz == 0:
        return None
    names = bundle["tfidf"].get_feature_names_out()
    rows = []
    for name in ORDER:
        m = bundle["models"].get(name)
        if m is None:
            continue
        cols, eff, how = word_effects(m, name, vec)
        pairs = sorted(((names[c].replace("_", " "), float(e)) for c, e in zip(cols, eff)),
                       key=lambda x: x[1])
        pos = [w for w, e in pairs[::-1] if e > 1e-9][:top]
        neg = [w for w, e in pairs if e < -1e-9][:top]
        rows.append({"Mô hình": name, "Từ đẩy về TÍCH CỰC": ", ".join(pos) or "-",
                     "Từ đẩy về TIÊU CỰC": ", ".join(neg) or "-", "Cách tính": how})
    return pd.DataFrame(rows)


def predict_single(title, text):
    if not BUNDLES:
        return "⚠️ Chưa có mô hình. Hãy chạy `python train_compare.py` trước.", None, None, ""
    text = (text or "").strip()
    if not text:
        return "Hãy nhập một bình luận.", None, None, ""
    b = BUNDLES[title]
    rows, vec = run_models(b, text)
    label, agree, n, note, avg = verdict(rows)
    head = f"## Kết luận: {label}\n{note}"
    if avg is not None:
        head += f"\n\nXác suất tích cực trung bình (các mô hình có xác suất): **{avg:.0%}**"
    if vec.nnz == 0:
        head += ("\n\n⚠️ Không có từ nào của câu nằm trong từ vựng mô hình, "
                 "kết quả này **không đáng tin**.")
    table = pd.DataFrame([{"Mô hình": r["name"], "Dự đoán": LABEL[r["pred"]], "Độ tự tin": r["conf"]}
                          for r in rows])
    return head, table, explain(b, vec), clean_text(text)


# ----------------------------------------------------------------------------- xử lý hàng loạt
TEXT_COLS = ("text", "sentence", "comment", "content", "review", "câu", "bình luận")
GROUP_COLS = ("category", "nhóm", "nhom", "group")
POS_SET = {"TÍCH CỰC", "TICH CUC", "POS", "POSITIVE", "1"}
NEG_SET = {"TIÊU CỰC", "TIEU CUC", "NEG", "NEGATIVE", "0"}


def read_input(lines, file):
    """Gộp các dòng nhập tay và file (CSV/XLSX). Giữ nguyên mọi cột phụ trong file."""
    frames = []
    texts = [t.strip() for t in (lines or "").splitlines() if t.strip()]
    if texts:
        frames.append(pd.DataFrame({"text": texts}))
    if file is not None:
        path = file if isinstance(file, str) else getattr(file, "name", str(file))
        d = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) \
            else pd.read_csv(path, encoding="utf-8-sig")
        col = next((c for c in d.columns if str(c).strip().lower() in TEXT_COLS), d.columns[0])
        d = d.rename(columns={col: "text"})
        d = d[d["text"].notna()].copy()
        d["text"] = d["text"].astype(str)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def norm_expected(v):
    s = str(v).strip().upper()
    return 1 if s in POS_SET else 0 if s in NEG_SET else None


def build_batch(bundle, df):
    """Trả về (bảng kết quả, các bảng tổng hợp, thông tin)."""
    extra = [c for c in df.columns if c != "text"]
    exp_col = next((c for c in extra if str(c).strip().lower() in ("expected", "kỳ vọng", "ky vong", "label", "nhãn")), None)
    grp_col = next((c for c in extra if str(c).strip().lower() in GROUP_COLS), None)

    recs, preds = [], {name: [] for name in ORDER}
    for i, row in df.iterrows():
        rows, vec = run_models(bundle, row["text"])
        label, agree, n, _, avg = verdict(rows)
        level = ("Rõ ràng" if agree == n else "Khá chắc" if agree == n - 1 else "Lưỡng lự")
        rec = {"STT": i + 1, "Câu": row["text"]}
        for c in extra:                                        # giữ cột phụ của file gốc
            rec[c] = row[c]
        rec["Kết luận"] = label
        rec["Mức chắc chắn"] = f"{level} ({agree}/{n})"
        rec["P(tích cực) TB"] = f"{avg:.0%}" if avg is not None else ""
        if vec.nnz == 0:
            rec["Cảnh báo"] = "Không có từ nào trong từ vựng - kết quả không đáng tin"
        for r in rows:
            rec[r["name"]] = f"{LABEL[r['pred']]} ({r['conf']})"
            preds[r["name"]].append(r["pred"])
        pos = sum(r["pred"] for r in rows)
        preds.setdefault("_vote", []).append(1 if pos > n - pos else 0)
        preds.setdefault("_agree", []).append(agree)
        if exp_col is not None:
            e = norm_expected(row[exp_col])
            rec["Đối chiếu kỳ vọng"] = "Không chấm" if e is None else ("ĐÚNG" if e == preds["_vote"][-1] else "SAI")
            preds.setdefault("_exp", []).append(e)
        recs.append(rec)
    res = pd.DataFrame(recs)
    n_all, n_models = len(res), len(ORDER)
    unsure = sum(1 for a in preds["_agree"] if a < n_models - 1)

    overview = [("Tổng số câu", n_all),
                ("Số câu lưỡng lự (chỉ 3/5 mô hình đồng ý)", f"{unsure} ({unsure / n_all:.0%})")]
    by_model = by_group = None
    if exp_col is not None:
        exp = pd.Series(preds["_exp"], dtype="object")
        mask = exp.notna()
        k = int(mask.sum())
        overview.append(("Số câu có kỳ vọng rõ (TÍCH CỰC/TIÊU CỰC) để chấm", k))
        if k:
            ev = exp[mask].astype(int).to_numpy()
            rows_m = []
            for name in ["Đa số (5 mô hình)"] + ORDER:
                p = np.array(preds["_vote"] if name.startswith("Đa số") else preds[name])[mask.to_numpy()]
                ok = int((p == ev).sum())
                rows_m.append({"Mô hình": name, "Đúng": ok, "Số câu chấm": k, "Độ chính xác": f"{ok / k:.1%}"})
            by_model = pd.DataFrame(rows_m)
            overview.append(("Độ chính xác của kết luận đa số", f"{by_model.iloc[0]['Độ chính xác']} "
                                                              f"({by_model.iloc[0]['Đúng']}/{k})"))
            if grp_col is not None:
                tmp = pd.DataFrame({"g": res[grp_col].astype(str), "ok": (np.array(preds["_vote"]) == exp.fillna(-1).to_numpy()),
                                    "rated": mask.to_numpy(), "unsure": np.array(preds["_agree"]) < n_models - 1})
                g = tmp.groupby("g", sort=False).agg(**{"Số câu": ("g", "size"), "Số câu chấm": ("rated", "sum"),
                                                        "Đúng": ("ok", lambda s: int((s & tmp.loc[s.index, "rated"]).sum())),
                                                        "Số câu lưỡng lự": ("unsure", "sum")}).reset_index()
                g["Độ chính xác"] = [f"{o / r:.0%}" if r else "-" for o, r in zip(g["Đúng"], g["Số câu chấm"])]
                by_group = g.rename(columns={"g": "Nhóm"})
    ov = pd.DataFrame(overview, columns=["Chỉ số", "Giá trị"])
    return res, ov, by_model, by_group


GUIDE = [
    ("Kết luận", "Nhãn theo đa số phiếu của 5 mô hình."),
    ("Mức chắc chắn", "Rõ ràng = 5/5 mô hình đồng ý; Khá chắc = 4/5; Lưỡng lự = chỉ 3/5 (câu khó hoặc mơ hồ)."),
    ("P(tích cực) TB", "Xác suất tích cực trung bình của các mô hình có xác suất (NB, LR, RF). "
                       "Xác suất của Naive Bayes thường bị đẩy về gần 0%/100%."),
    ("Cột từng mô hình", "Nhãn và độ tự tin: % là xác suất; 'margin' là điểm số của Linear SVM (xa 0 = chắc hơn)."),
    ("Đối chiếu kỳ vọng", "Chỉ có khi file có cột expected: ĐÚNG/SAI so với kết luận đa số. Nhãn khác TÍCH CỰC/"
                          "TIÊU CỰC (LẪN LỘN, KHÔNG RÕ, TRUNG TÍNH) ghi 'Không chấm'."),
    ("Cảnh báo", "Câu không có từ nào nằm trong từ vựng mô hình: kết quả chỉ là mặc định, không đáng tin."),
    ("Lưu ý", "Mô hình chỉ có 2 nhãn (Tích cực/Tiêu cực), không có Trung tính. Cột expected do người nhập gán, "
              "không phải đáp án chuẩn."),
]


def style_excel(path):
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    fills = {"TÍCH CỰC": "C6EFCE", "TIÊU CỰC": "FFC7CE", "Lưỡng lự": "FFEB9C", "Khá chắc": "FFF2CC",
             "Rõ ràng": "E2EFDA", "ĐÚNG": "C6EFCE", "SAI": "FFC7CE"}
    wb = load_workbook(path)
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="305496")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"
        for j, col in enumerate(ws.columns, 1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, width + 2), 70 if j == 2 and ws.title == "Kết quả" else 34)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(vertical="top", wrap_text=True)
                v = str(c.value) if c.value is not None else ""
                for key, color in fills.items():
                    if v.startswith(key):
                        c.fill = PatternFill("solid", fgColor=color)
                        break
    ws = wb["Kết quả"]
    ws.auto_filter.ref = ws.dimensions
    wb.save(path)


def predict_batch(title, lines, file):
    """Trả về: tóm tắt, bảng kết quả, bảng theo mô hình, bảng theo nhóm, file xlsx, file csv."""
    empty = (None, None, None, None, None, None)
    if not BUNDLES:
        return ("⚠️ Chưa có mô hình. Hãy chạy `python train_compare.py` trước.",) + empty[1:]
    try:
        df = read_input(lines, file)
    except Exception as e:
        return (f"⚠️ Không đọc được file: {e}",) + empty[1:]
    if df is None or df.empty:
        return ("Hãy nhập bình luận hoặc tải file lên.",) + empty[1:]

    res, ov, by_model, by_group = build_batch(BUNDLES[title], df)
    folder = tempfile.mkdtemp()
    csv_path = os.path.join(folder, "ket_qua_phan_loai.csv")
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    xlsx_path = os.path.join(folder, "ket_qua_phan_loai.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            res.to_excel(w, sheet_name="Kết quả", index=False)
            ov.to_excel(w, sheet_name="Tổng quan", index=False)
            if by_model is not None:
                by_model.to_excel(w, sheet_name="Theo mô hình", index=False)
            if by_group is not None:
                by_group.to_excel(w, sheet_name="Theo nhóm", index=False)
            pd.DataFrame(GUIDE, columns=["Cột", "Ý nghĩa"]).to_excel(w, sheet_name="Hướng dẫn đọc", index=False)
        style_excel(xlsx_path)
    except ImportError:
        xlsx_path = None                                   # thiếu openpyxl -> chỉ có CSV
    summary = "### Tóm tắt\n" + "\n".join(f"- **{k}**: {v}" for k, v in ov.itertuples(index=False))
    if xlsx_path is None:
        summary += "\n\n(Cài `pip install openpyxl` để xuất thêm file Excel có định dạng màu.)"
    return summary, res, by_model, by_group, xlsx_path, csv_path


# ----------------------------------------------------------------------------- giao diện
def build_ui():
    with gr.Blocks(title="Phân loại cảm xúc văn bản tiếng Việt") as demo:
        gr.Markdown("# Phân loại cảm xúc văn bản tiếng Việt\n"
                    "TF-IDF + 5 thuật toán học máy (MultinomialNB, ComplementNB, "
                    "Logistic Regression, Linear SVM, Random Forest). "
                    "Câu lưỡng lự là câu các mô hình **không đồng thuận**.")
        title = gr.Dropdown(TITLES, value=TITLES[0] if TITLES else None,
                            label="Mô hình học từ bộ dữ liệu nào")
        with gr.Tabs():
            with gr.Tab("Một bình luận"):
                txt = gr.Textbox(label="Nhập bình luận tiếng Việt", lines=3,
                                 placeholder="Ví dụ: Sản phẩm dùng rất tốt, giao hàng nhanh")
                btn = gr.Button("Phân loại", variant="primary")
                verdict_md = gr.Markdown()
                tbl = gr.Dataframe(label="Dự đoán của từng mô hình", interactive=False)
                with gr.Accordion("Vì sao ra kết quả này? (từ khóa mà từng mô hình dựa vào)", open=True):
                    why = gr.Dataframe(interactive=False, wrap=True)
                    gr.Markdown("Đây là các từ/cụm từ mô hình **dựa vào** để nghiêng về một phía, "
                                "không có nghĩa mô hình hiểu câu. Cụm 2 từ (bigram) có thể trùng với "
                                "từ đơn bên trong nó. Mỗi mô hình tính theo cách khác nhau nên không so "
                                "sánh số giữa các mô hình, chỉ so từ nào được nêu.")
                with gr.Accordion("Văn bản sau khi làm sạch + tách từ", open=False):
                    cleaned = gr.Textbox(show_label=False, interactive=False)
                gr.Examples(EXAMPLES, inputs=txt)
                for trig in (btn.click, txt.submit):
                    trig(predict_single, [title, txt], [verdict_md, tbl, why, cleaned])
            with gr.Tab("Nhiều bình luận"):
                gr.Markdown("Mỗi dòng một bình luận, hoặc tải lên file **CSV/Excel** (cột `text`/`sentence`; "
                            "không có thì lấy cột đầu tiên). Các cột khác trong file được **giữ nguyên** ở kết quả. "
                            "Nếu file có cột `expected` (TÍCH CỰC/TIÊU CỰC) và `category`, "
                            "bảng kết quả sẽ tự chấm đúng/sai và thống kê theo mô hình và theo nhóm.")
                lines = gr.Textbox(label="Danh sách bình luận (mỗi dòng một câu)", lines=5)
                file = gr.File(label="File CSV / Excel (tùy chọn)", file_types=[".csv", ".xlsx"])
                btn2 = gr.Button("Phân loại tất cả", variant="primary")
                summ_md = gr.Markdown()
                out_tbl = gr.Dataframe(label="Kết quả chi tiết", interactive=False, wrap=True)
                with gr.Row():
                    m_tbl = gr.Dataframe(label="Độ chính xác theo mô hình (nếu có cột expected)", interactive=False)
                    g_tbl = gr.Dataframe(label="Theo nhóm (nếu có cột category)", interactive=False)
                with gr.Row():
                    out_xlsx = gr.File(label="Tải kết quả (Excel, có màu + nhiều sheet)")
                    out_csv = gr.File(label="Tải kết quả (CSV)")
                btn2.click(predict_batch, [title, lines, file],
                           [summ_md, out_tbl, m_tbl, g_tbl, out_xlsx, out_csv])
        if not BUNDLES:
            gr.Markdown("⚠️ **Chưa có mô hình** trong thư mục `models/`. "
                        "Chạy `python train_compare.py` trước rồi mở lại giao diện.")
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
