"""
So sánh 5 thuật toán học máy trên TF-IDF cho phân loại cảm xúc tiếng Việt.
Mỗi mô hình chạy 2 phiên bản:  Baseline (tham số theo báo cáo)  và  Tuned (GridSearchCV).
Chạy trên 2 bộ dữ liệu -> bảng tổng hợp 3.2.3 + biểu đồ.

pip install scikit-learn pandas numpy underthesea datasets kagglehub matplotlib seaborn pyarrow
python train_compare.py
"""
import glob
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
SEED = 42
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
N_TIMING_RUNS = 3          # số lần đo lặp lại thời gian rồi lấy trung bình
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _import_tokenizer():
    try:
        from underthesea import word_tokenize
        return word_tokenize
    except ImportError:
        pass
    print("[i] Chưa có underthesea -> đang tự cài (pip install underthesea)...")
    import subprocess, sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "underthesea"])
        from underthesea import word_tokenize
        return word_tokenize
    except Exception as e:
        print(f"[!] Không cài được underthesea ({e}) -> tách từ theo khoảng trắng "
              "(kết quả sẽ khác báo cáo).")
        return None


word_tokenize = _import_tokenizer()
TOKENIZER_TAG = "uts" if word_tokenize else "space"   # tách cache theo kiểu tách từ


# ============================ 1. DỮ LIỆU & TIỀN XỬ LÝ ============================
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\w\s]|\d|_", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return word_tokenize(t, format="text") if word_tokenize else t


def clean_cached(series, key):
    """Làm sạch + tách từ, lưu cache để lần sau không phải chạy lại (underthesea chậm).
    Tên cache có mã băm nội dung -> dữ liệu đổi (vd. sau khi loại trùng) thì tự tạo cache mới,
    không bao giờ dùng nhầm cache cũ."""
    import hashlib
    h = hashlib.md5("\n".join(map(str, series)).encode("utf-8")).hexdigest()[:10]
    path = f"{CACHE_DIR}/{key}_{TOKENIZER_TAG}_{h}.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    out = series.apply(clean_text)
    out.to_pickle(path)
    return out


# ---- Kiểm toán & làm sạch dữ liệu (ghi lại số mẫu ở từng bước -> results/data_audit.csv) ----
REMOVE_TEST_OVERLAP_DS2 = True   # True: loại khỏi test các câu trùng với train (chặt chẽ hơn tập chuẩn gốc)
AUDIT = []


def log_step(ds, step, before, after=None):
    after = before if after is None else after
    AUDIT.append({"Bộ dữ liệu": ds, "Bước": step, "Trước": before, "Sau": after,
                  "Loại bỏ": before - after})
    print(f"  [{ds}] {step}: {before} -> {after}" + (f"  (loại {before - after})" if before != after else ""))


def save_audit():
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(AUDIT).to_csv("results/data_audit.csv", index=False, encoding="utf-8-sig")


def _key(t):
    """Khóa chuẩn hóa nhẹ (không tách từ) để phát hiện câu rỗng/trùng."""
    s = str(t).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^\w\s]|\d|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def dedup_clean(df, ds, tag):
    """df có cột text, label. Bỏ câu rỗng sau làm sạch, câu mâu thuẫn nhãn, câu trùng."""
    df = df.copy()
    df["_k"] = df["text"].map(_key)
    n = len(df); df = df[df["_k"] != ""]
    log_step(ds, f"{tag}: bỏ câu rỗng sau làm sạch", n, len(df))
    n = len(df); df = df[df.groupby("_k")["label"].transform("nunique") == 1]
    log_step(ds, f"{tag}: bỏ câu trùng nhưng nhãn mâu thuẫn", n, len(df))
    n = len(df); df = df.drop_duplicates("_k")
    log_step(ds, f"{tag}: bỏ câu trùng nội dung", n, len(df))
    return df.reset_index(drop=True)


def load_ds1():
    import kagglehub
    ds = "Bộ 1"
    p = kagglehub.dataset_download("linhlpv/vietnamese-sentiment-analyst")
    files = sorted(glob.glob(f"{p}/**/*.csv", recursive=True))
    TEXT = ("text", "sentence", "comment", "content", "review")
    LAB = ("label", "sentiment", "labels", "rating")
    parts, total = [], 0
    for f in files:                      # nhận diện cột RIÊNG từng file -> không mất dòng khi tên cột khác nhau
        d = pd.read_csv(f)
        tcol = next((c for c in d.columns if str(c).lower() in TEXT), None)
        lcol = next((c for c in d.columns if str(c).lower() in LAB), None)
        print(f"  [{ds}] file {os.path.basename(f)}: {len(d)} dòng, cột văn bản={tcol}, cột nhãn={lcol}")
        total += len(d)
        if tcol is None or lcol is None:
            print("     -> bỏ qua file này (không nhận diện được cột văn bản/nhãn)")
            continue
        d = d[[tcol, lcol]].copy()
        d.columns = ["text", "label"]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    log_step(ds, "Tổng số dòng gốc (ghép các file)", total, len(df))
    n = len(df)
    df = df.dropna()
    log_step(ds, "Bỏ dòng thiếu văn bản hoặc nhãn", n, len(df))
    print(f"  [{ds}] phân bố nhãn gốc: {df['label'].value_counts().to_dict()}")
    n = len(df)
    df["label"] = df["label"].map({"POS": 1, "NEG": 0, "pos": 1, "neg": 0})
    df = df.dropna().astype({"label": int})
    log_step(ds, "Bỏ nhãn trung tính (NEU)/nhãn lạ", n, len(df))

    df = dedup_clean(df, ds, "toàn bộ")          # loại trùng TRƯỚC khi chia -> không rò rỉ train/test
    out = train_test_split(df["text"], df["label"], test_size=0.2,
                           stratify=df["label"], random_state=SEED)
    log_step(ds, f"Sau chia 80/20: train (%Tích cực={out[2].mean():.2%})", len(out[0]))
    log_step(ds, f"Sau chia 80/20: test (%Tích cực={out[3].mean():.2%})", len(out[1]))
    save_audit()
    return tuple(o.reset_index(drop=True) for o in out)


def load_ds2():
    ds = "Bộ 2"
    base = ("https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/"
            "resolve/refs%2Fconvert%2Fparquet/default/{s}/0000.parquet")
    try:
        fr = {s: pd.read_parquet(base.format(s=s)) for s in ("train", "test")}
    except Exception:
        from datasets import load_dataset
        d = load_dataset("uitnlp/vietnamese_students_feedback", revision="refs/convert/parquet")
        fr = {s: d[s].to_pandas() for s in ("train", "test")}

    sets = {}
    for s in ("train", "test"):
        d = fr[s][["sentence", "sentiment"]].rename(columns={"sentence": "text"})
        log_step(ds, f"{s}: số dòng gốc", len(d))
        n = len(d); d = d[d["sentiment"] != 1]
        log_step(ds, f"{s}: bỏ nhãn trung tính", n, len(d))
        d = d.assign(label=(d["sentiment"] == 2).astype(int))[["text", "label"]]
        sets[s] = dedup_clean(d, ds, s)

    keys_tr = set(sets["train"]["_k"])
    overlap = sets["test"]["_k"].isin(keys_tr)
    print(f"  [{ds}] số câu test trùng với train: {int(overlap.sum())} "
          f"({overlap.mean():.2%} tập test)")
    if REMOVE_TEST_OVERLAP_DS2:
        n = len(sets["test"])
        sets["test"] = sets["test"][~overlap].reset_index(drop=True)
        log_step(ds, "test: loại câu trùng với train", n, len(sets["test"]))
    save_audit()
    tr, te = sets["train"], sets["test"]
    return tr["text"], te["text"], tr["label"], te["label"]


# ============================ 2. MÔ HÌNH & LƯỚI THAM SỐ ============================
# (baseline = tham số trong báo cáo, grid = không gian tìm kiếm cho bản Tuned)
def model_specs():
    return {
        "MultinomialNB": (MultinomialNB(),
                          {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}),
        "ComplementNB": (ComplementNB(),
                         {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=SEED),
                               {"C": [0.1, 1, 3, 10, 30],
                                "class_weight": [None, "balanced"]}),
        "LinearSVM": (LinearSVC(random_state=SEED),
                      {"C": [0.03, 0.1, 0.3, 1, 3],
                       "class_weight": [None, "balanced"]}),
        "RandomForest": (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
                         {"n_estimators": [100, 200],
                          "class_weight": [None, "balanced"]}),
    }


# ============================ 3. ĐÁNH GIÁ ============================
def timed_fit(est, X, y):
    """Trả về (mô hình đã fit, thời gian fit trung bình theo giây)."""
    from sklearn.base import clone
    ts = []
    for _ in range(N_TIMING_RUNS):
        m = clone(est)
        t0 = time.perf_counter()
        m.fit(X, y)
        ts.append(time.perf_counter() - t0)
    return m, float(np.mean(ts))


def timed_predict(m, X):
    ts = []
    for _ in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        pred = m.predict(X)
        ts.append((time.perf_counter() - t0) * 1000)
    return pred, float(np.mean(ts))


def evaluate(ds, name, version, est, Xtr, ytr, Xte, yte, params="-"):
    m, t_train = timed_fit(est, Xtr, ytr)
    pred, t_infer = timed_predict(m, Xte)
    return {
        "Dataset": ds, "Model": name, "Version": version, "Params": str(params),
        "Accuracy": accuracy_score(yte, pred),
        "Macro-Precision": precision_score(yte, pred, average="macro"),
        "Macro-Recall": recall_score(yte, pred, average="macro"),
        "Macro-F1": f1_score(yte, pred, average="macro"),
        "T_train (s)": t_train, "T_infer (ms)": t_infer,
    }, pred


def run_dataset(ds_name, loader, key):
    print(f"\n{'=' * 78}\n{ds_name}\n{'=' * 78}")
    Xtr_raw, Xte_raw, ytr, yte = loader()
    Xtr_txt = clean_cached(Xtr_raw, f"{key}_train")
    Xte_txt = clean_cached(Xte_raw, f"{key}_test")
    print(f"Train={len(ytr)}  Test={len(yte)}  %Tích cực(train)={ytr.mean():.2%}")

    # TF-IDF: chỉ fit trên train. sublinear_tf giảm ảnh hưởng của từ lặp nhiều lần.
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = tfidf.fit_transform(Xtr_txt)
    Xte = tfidf.transform(Xte_txt)

    rows, preds, saved = [], {}, {}
    for name, (base_est, grid) in model_specs().items():
        # --- Baseline
        r, p = evaluate(ds_name, name, "Baseline", base_est, Xtr, ytr, Xte, yte)
        rows.append(r)

        # --- Tuned: GridSearchCV theo Macro-F1, chỉ dùng tập train
        t0 = time.perf_counter()
        gs = GridSearchCV(base_est, grid, scoring="f1_macro", cv=CV, n_jobs=-1)
        gs.fit(Xtr, ytr)
        t_search = time.perf_counter() - t0
        r, p = evaluate(ds_name, name, "Tuned", gs.best_estimator_, Xtr, ytr, Xte, yte,
                        params=gs.best_params_)
        r["T_search (s)"] = t_search
        r["CV Macro-F1"] = gs.best_score_
        rows.append(r)
        preds[name] = p
        saved[name] = gs.best_estimator_      # đã refit trên toàn bộ tập train
        print(f"{name:<20} base F1={rows[-2]['Macro-F1']:.4f} -> tuned F1={r['Macro-F1']:.4f} "
              f"| {gs.best_params_} | search {t_search:.0f}s")

    # lưu vectorizer + 5 mô hình Tuned để demo.py dùng lại, không cần huấn luyện lại
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump({"title": ds_name, "tfidf": tfidf, "models": saved,
                 "tokenizer": TOKENIZER_TAG}, f"models/{key}.joblib")
    print(f"Đã lưu mô hình -> models/{key}.joblib")

    # ma trận nhầm lẫn của 5 mô hình bản Tuned
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (name, p) in zip(axes, preds.items()):
        sns.heatmap(confusion_matrix(yte, p), annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=ax, xticklabels=["Tiêu cực", "Tích cực"], yticklabels=["Tiêu cực", "Tích cực"])
        ax.set_title(name); ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    fig.suptitle(f"Ma trận nhầm lẫn (Tuned) - {ds_name}")
    plt.tight_layout(); plt.savefig(f"confusion_{key}.png", dpi=150); plt.close()
    return pd.DataFrame(rows)


# ============================ 4. TỔNG HỢP ============================
def summarize(df):
    pd.set_option("display.width", 220)
    cols = ["Model", "Version", "Accuracy", "Macro-Precision", "Macro-Recall",
            "Macro-F1", "T_train (s)", "T_infer (ms)"]
    for ds, g in df.groupby("Dataset", sort=False):
        print(f"\n### {ds}")
        print(g.sort_values("Macro-F1", ascending=False)[cols].round(4).to_string(index=False))

    # bảng đối chiếu 2 bộ dữ liệu (bản Tuned) - mục 3.2.3
    t = df[df["Version"] == "Tuned"]
    piv = t.pivot_table(index="Model", columns="Dataset",
                        values=["Macro-F1", "Accuracy", "T_train (s)"])
    print("\n### Đối chiếu Tuned trên 2 bộ dữ liệu")
    print(piv.round(4).to_string())

    df.to_csv("results_all.csv", index=False)

    # biểu đồ Macro-F1: Baseline vs Tuned
    dss = df["Dataset"].unique()
    fig, axes = plt.subplots(1, len(dss), figsize=(7 * len(dss), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        sns.barplot(data=df[df["Dataset"] == ds], x="Model", y="Macro-F1", hue="Version", ax=ax)
        ax.set_title(ds, fontsize=10); ax.tick_params(axis="x", rotation=20)
        lo = df["Macro-F1"].min()
        ax.set_ylim(max(0, lo - 0.05), 1)
    plt.tight_layout(); plt.savefig("compare_macro_f1.png", dpi=150); plt.close()

    # đánh đổi độ chính xác - thời gian huấn luyện (log)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=t, x="T_train (s)", y="Macro-F1", hue="Model", style="Dataset", s=120)
    plt.xscale("log"); plt.title("Macro-F1 vs thời gian huấn luyện (Tuned)")
    plt.tight_layout(); plt.savefig("tradeoff_f1_time.png", dpi=150); plt.close()
    print("\nĐã lưu: results_all.csv, compare_macro_f1.png, tradeoff_f1_time.png, confusion_*.png")


if __name__ == "__main__":
    all_res = pd.concat([
        run_dataset("Bộ 1: Vietnamese Sentiment Analyst", load_ds1, "ds1"),
        run_dataset("Bộ 2: UIT-VSFC", load_ds2, "ds2"),
    ], ignore_index=True)
    summarize(all_res)
