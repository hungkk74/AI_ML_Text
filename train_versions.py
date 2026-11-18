"""
Huấn luyện 5 mô hình x 3 phiên bản (Base, Tune 1, Tune 2) trên 2 bộ dữ liệu.
  Base   : tham số theo báo cáo (còn lại là mặc định scikit-learn)
  Tune 1 : cấu hình tốt nhất theo CV ở Bộ 1
  Tune 2 : cấu hình tốt nhất theo CV ở Bộ 2
Cả hai cấu hình tune được áp dụng cho cả 2 bộ dữ liệu -> kiểm tra độ bền của tham số.

Đầu ra (thư mục results/):
  results_summary.csv / results_summary.xlsx   bảng tổng hợp duy nhất (30 dòng)
  images/cm_<bộ>_<mô hình>.png                 3 ma trận nhầm lẫn (Base/Tune1/Tune2) của mỗi mô hình
  images/bars_<bộ>.png                         Macro-F1 và thời gian huấn luyện của mọi phiên bản
  images/table_<bộ>.png                        ảnh bảng kết quả từng bộ

Cần train_compare.py cùng thư mục (dùng lại hàm tải dữ liệu & tiền xử lý).
python train_versions.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

import train_compare as tc            # load_ds1, load_ds2, clean_cached, timed_fit, ...

SEED = 42
OUT = "results"
IMG = f"{OUT}/images"
os.makedirs(IMG, exist_ok=True)

# ---------------------------------------------------------------------------
# 3 phiên bản / mô hình: (lớp, tham số)
# ---------------------------------------------------------------------------
VERSIONS = {
    "MultinomialNB": [
        ("Base",   MultinomialNB, {}),
        ("Tune 1", MultinomialNB, {"alpha": 0.05}),
        ("Tune 2", MultinomialNB, {"alpha": 0.5}),
    ],
    "ComplementNB": [
        ("Base",   ComplementNB, {}),
        ("Tune 1", ComplementNB, {"alpha": 0.01}),
        ("Tune 2", ComplementNB, {"alpha": 0.5}),
    ],
    "LogisticRegression": [
        ("Base",   LogisticRegression, {"max_iter": 1000, "random_state": SEED}),
        ("Tune 1", LogisticRegression, {"max_iter": 1000, "random_state": SEED, "C": 3}),
        ("Tune 2", LogisticRegression, {"max_iter": 1000, "random_state": SEED, "C": 10,
                                        "class_weight": "balanced"}),
    ],
    "LinearSVM": [
        ("Base",   LinearSVC, {"random_state": SEED}),
        ("Tune 1", LinearSVC, {"random_state": SEED, "C": 0.3}),
        ("Tune 2", LinearSVC, {"random_state": SEED, "C": 0.3, "class_weight": "balanced"}),
    ],
    "RandomForest": [
        ("Base",   RandomForestClassifier, {"n_estimators": 100, "n_jobs": -1, "random_state": SEED}),
        ("Tune 1", RandomForestClassifier, {"n_estimators": 200, "n_jobs": -1, "random_state": SEED}),
        ("Tune 2", RandomForestClassifier, {"n_estimators": 200, "n_jobs": -1, "random_state": SEED,
                                            "class_weight": "balanced"}),
    ],
}
HIDDEN = {"random_state", "n_jobs", "max_iter"}      # không hiển thị trong cột Params


def show_params(p):
    s = ", ".join(f"{k}={v}" for k, v in p.items() if k not in HIDDEN)
    return s or "mặc định"


# ---------------------------------------------------------------------------
def run_dataset(ds_name, loader, key):
    print(f"\n{'=' * 78}\n{ds_name}\n{'=' * 78}")
    Xtr_raw, Xte_raw, ytr, yte = loader()
    Xtr_txt = tc.clean_cached(Xtr_raw, f"{key}_train")
    Xte_txt = tc.clean_cached(Xte_raw, f"{key}_test")
    print(f"Train={len(ytr)}  Test={len(yte)}  %Tích cực(train)={ytr.mean():.2%}")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = tfidf.fit_transform(Xtr_txt)          # chỉ fit trên train
    Xte = tfidf.transform(Xte_txt)

    rows, preds = [], {}
    for name, versions in VERSIONS.items():
        for ver, cls, params in versions:
            est = cls(**params)
            cv = cross_val_score(est, Xtr, ytr, scoring="f1_macro",
                                 cv=tc.CV, n_jobs=-1).mean()
            model, t_train = tc.timed_fit(est, Xtr, ytr)
            pred, t_infer = tc.timed_predict(model, Xte)
            preds[(name, ver)] = pred
            rows.append({
                "Dataset": ds_name, "Model": name, "Version": ver,
                "Params": show_params(params),
                "Accuracy": accuracy_score(yte, pred),
                "Macro-Precision": precision_score(yte, pred, average="macro"),
                "Macro-Recall": recall_score(yte, pred, average="macro"),
                "Macro-F1": f1_score(yte, pred, average="macro"),
                "CV Macro-F1": cv,
                "T_train (s)": t_train, "T_infer (ms)": t_infer,
            })
            print(f"{name:<19}{ver:<8} F1={rows[-1]['Macro-F1']:.4f}  CV={cv:.4f}  "
                  f"train={t_train:.3f}s  | {show_params(params)}")
    df = pd.DataFrame(rows)
    plot_confusions(df, preds, yte, key, ds_name)
    return df


# ---------------------------------------------------------------------------
# HÌNH ẢNH
# ---------------------------------------------------------------------------
def plot_confusions(df, preds, yte, key, ds_name):
    """Mỗi mô hình 1 ảnh: 3 ma trận nhầm lẫn (Base / Tune 1 / Tune 2)."""
    for name in VERSIONS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (ver, _, params) in zip(axes, VERSIONS[name]):
            r = df[(df["Model"] == name) & (df["Version"] == ver)].iloc[0]
            sns.heatmap(confusion_matrix(yte, preds[(name, ver)]), annot=True, fmt="d",
                        cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Tiêu cực", "Tích cực"],
                        yticklabels=["Tiêu cực", "Tích cực"])
            ax.set_title(f"{ver}: {show_params(params)}\n"
                         f"Acc={r['Accuracy']:.4f}  Macro-F1={r['Macro-F1']:.4f}", fontsize=9)
            ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        fig.suptitle(f"{name} - {ds_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{IMG}/cm_{key}_{name}.png", dpi=150)
        plt.close()


def plot_bars(df, key, ds_name):
    g = df[df["Dataset"] == ds_name]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    sns.barplot(data=g, x="Model", y="Macro-F1", hue="Version", ax=axes[0])
    axes[0].set_ylim(max(0, g["Macro-F1"].min() - 0.03), min(1, g["Macro-F1"].max() + 0.02))
    axes[0].set_title("Macro-F1 theo phiên bản"); axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=g, x="Model", y="T_train (s)", hue="Version", ax=axes[1])
    axes[1].set_yscale("log"); axes[1].set_title("Thời gian huấn luyện (log)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(ds_name, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{IMG}/bars_{key}.png", dpi=150); plt.close()


def plot_table(df, key, ds_name):
    cols = ["Model", "Version", "Params", "Accuracy", "Macro-Precision",
            "Macro-Recall", "Macro-F1", "CV Macro-F1", "T_train (s)"]
    g = df[df["Dataset"] == ds_name][cols].copy()
    for c in cols[3:8]:
        g[c] = g[c].map("{:.4f}".format)
    g["T_train (s)"] = g["T_train (s)"].map("{:.3f}".format)
    fig, ax = plt.subplots(figsize=(16, 0.42 * len(g) + 1.2))
    ax.axis("off")
    tb = ax.table(cellText=g.values, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.35)
    tb.auto_set_column_width(list(range(len(cols))))
    for j in range(len(cols)):
        tb[0, j].set_facecolor("#dbe4f0"); tb[0, j].set_text_props(fontweight="bold")
    for i, v in enumerate(g["Version"], start=1):          # tô nhạt theo phiên bản
        if v == "Base":
            for j in range(len(cols)):
                tb[i, j].set_facecolor("#f3f3f3")
    ax.set_title(ds_name, fontweight="bold")
    plt.savefig(f"{IMG}/table_{key}.png", dpi=150, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sets = [("Bộ 1: Vietnamese Sentiment Analyst", tc.load_ds1, "ds1"),
            ("Bộ 2: UIT-VSFC", tc.load_ds2, "ds2")]
    df = pd.concat([run_dataset(n, l, k) for n, l, k in sets], ignore_index=True)

    # bảng tổng hợp duy nhất
    df.to_csv(f"{OUT}/results_summary.csv", index=False, encoding="utf-8-sig")
    try:
        df.round(4).to_excel(f"{OUT}/results_summary.xlsx", index=False)
    except ImportError:
        print("[i] Chưa có openpyxl -> chỉ xuất CSV (pip install openpyxl để có .xlsx)")

    for n, _, k in sets:
        plot_bars(df, k, n)
        plot_table(df, k, n)

    pd.set_option("display.width", 220)
    print("\n" + df.drop(columns="Dataset").round(4).to_string(index=False))
    print(f"\nĐã lưu kết quả vào thư mục '{OUT}/'")
