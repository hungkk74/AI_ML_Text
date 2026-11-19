"""
Huấn luyện 5 mô hình x 3 phiên bản (Base, Tune 1, Tune 2) trên 2 bộ dữ liệu.
  Base   : tham số mặc định (xem tc.model_specs)
  Tune 1 : cấu hình tốt nhất theo GridSearchCV (CV Macro-F1, chỉ trên tập train) của CHÍNH bộ dữ liệu đó
  Tune 2 : cùng cấu hình Tune 1 nhưng đổi class_weight (None <-> 'balanced') với LR / SVM / RF;
           MultinomialNB / ComplementNB không có class_weight nên Tune 2 = alpha tốt thứ 2 theo CV.
=> Tune 1/2 luôn tự cập nhật theo dữ liệu đã làm sạch, không còn số cố định.
   Cấu hình đã chọn được lưu results/configs_<bộ>.json (tự chọn lại khi dữ liệu train thay đổi).

Đầu ra (thư mục results/):
  results_summary.csv / results_summary.xlsx   bảng tổng hợp duy nhất (30 dòng)
  cv_ranking_<bộ>.csv                          xếp hạng đầy đủ mọi cấu hình theo CV (mean, std, hạng)
  grid_<bộ>.csv + images/grid_<bộ>_<mô hình>.png  kết quả TỪNG cấu hình trong lưới (CV + test), đánh dấu Base/Tune 1/Tune 2
  images/cm_<bộ>_<mô hình>.png                 3 ma trận nhầm lẫn (Base/Tune1/Tune2) của mỗi mô hình
  images/bars_<bộ>.png, images/table_<bộ>.png

Cần train_compare.py cùng thư mục.   python train_versions.py
"""
import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score

import train_compare as tc            # load_ds1, load_ds2, clean_cached, model_specs, timed_fit, ...

OUT = "results"
IMG = f"{OUT}/images"
os.makedirs(IMG, exist_ok=True)
VERS = ("Base", "Tune 1", "Tune 2")
HIDDEN = {"random_state", "n_jobs", "max_iter"}


def show_params(p):
    s = ", ".join(f"{k}={v}" for k, v in p.items() if k not in HIDDEN)
    return s or "mặc định"


PCOLS = ["alpha", "C", "class_weight", "n_estimators"]


def param_cols(est):
    """Giá trị THỰC của từng tham số (kể cả mặc định); '-' nếu mô hình không có tham số đó."""
    p = est.get_params()
    return {c: (str(p[c]) if c in p else "-") for c in PCOLS}


def make_estimator(name, params):
    """Ước lượng gốc trong tc.model_specs() + tham số chọn thêm."""
    return clone(tc.model_specs()[name][0]).set_params(**params)


def _signature(Xtr, ytr):
    h = hashlib.md5()
    h.update(str(Xtr.shape).encode())
    h.update(np.asarray(ytr).tobytes())
    h.update(str(Xtr.nnz).encode() + str(round(float(Xtr.sum()), 6)).encode())
    h.update(json.dumps({k: v[1] for k, v in tc.model_specs().items()}, sort_keys=True,
                        default=str).encode())
    return h.hexdigest()


def _flip(cw):
    return "balanced" if cw is None else None


def select_configs(key, Xtr, ytr):
    """Trả {model: {"Base": {}, "Tune 1": {...}, "Tune 2": {...}}}; chọn bằng CV trên train."""
    sig = _signature(Xtr, ytr)
    path = f"{OUT}/configs_{key}.json"
    if os.path.exists(path):
        saved = json.load(open(path, encoding="utf-8"))
        if saved.get("signature") == sig:
            print(f"[{key}] dùng cấu hình đã chọn ({path})")
            return saved["configs"]
    print(f"[{key}] chọn Tune 1 / Tune 2 bằng GridSearchCV trên tập train ...")
    configs, ranks = {}, []
    for name, (est, grid) in tc.model_specs().items():
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=tc.CV, n_jobs=-1, refit=False)
        gs.fit(Xtr, ytr)
        r = pd.DataFrame(gs.cv_results_["params"])
        r["mean"] = gs.cv_results_["mean_test_score"]
        r["std"] = gs.cv_results_["std_test_score"]
        r = r.sort_values("mean", ascending=False, kind="stable").reset_index(drop=True)
        r.insert(0, "rank", r.index + 1)
        best = {k: v for k, v in r.iloc[0].items() if k not in ("rank", "mean", "std")}
        best = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in best.items()}
        if "class_weight" in best:
            t2 = dict(best); t2["class_weight"] = _flip(best["class_weight"])
            how2 = "đổi class_weight"
        else:                                    # Naive Bayes: alpha tốt thứ 2
            t2 = {k: v for k, v in r.iloc[1].items() if k not in ("rank", "mean", "std")}
            how2 = "alpha tốt thứ 2"
        for k, v in list(best.items()):          # numpy -> python thuần để lưu JSON
            best[k] = v.item() if hasattr(v, "item") else v
        for k, v in list(t2.items()):
            t2[k] = v.item() if hasattr(v, "item") else v
        configs[name] = {"Base": {}, "Tune 1": best, "Tune 2": t2, "how2": how2}
        r.insert(0, "model", name)
        ranks.append(r)
        print(f"\n  {name}: xếp hạng CV Macro-F1 (3-fold, trung bình ± độ lệch chuẩn)")
        for _, x in r.head(5).iterrows():
            p = {k: x[k] for k in r.columns if k not in ("model", "rank", "mean", "std") and not pd.isna(x[k])}
            print(f"    #{int(x['rank'])}  {x['mean']:.4f} ± {x['std']:.4f}  {p}")
        print(f"    -> Tune 1 = {best} | Tune 2 = {t2} ({how2})")
    pd.concat(ranks, ignore_index=True).to_csv(f"{OUT}/cv_ranking_{key}.csv", index=False,
                                               encoding="utf-8-sig")
    json.dump({"signature": sig, "configs": configs}, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return configs


def prepare(loader, key):
    Xtr_raw, Xte_raw, ytr, yte = loader()
    Xtr_txt = tc.clean_cached(Xtr_raw, f"{key}_train")
    Xte_txt = tc.clean_cached(Xte_raw, f"{key}_test")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = tfidf.fit_transform(Xtr_txt)          # chỉ fit trên train
    return Xtr, tfidf.transform(Xte_txt), ytr, yte


# ---------------------------------------------------------------------------
def run_dataset(ds_name, loader, key):
    print(f"\n{'=' * 78}\n{ds_name}\n{'=' * 78}")
    Xtr, Xte, ytr, yte = prepare(loader, key)
    print(f"Train={len(ytr)}  Test={len(yte)}  %Tích cực(train)={ytr.mean():.2%}")
    cfgs = select_configs(key, Xtr, ytr)

    rows, preds = [], {}
    for name in tc.model_specs():
        for ver in VERS:
            params = cfgs[name][ver]
            est = make_estimator(name, params)
            cv = cross_val_score(est, Xtr, ytr, scoring="f1_macro", cv=tc.CV, n_jobs=-1).mean()
            model, t_train = tc.timed_fit(est, Xtr, ytr)
            pred, t_infer = tc.timed_predict(model, Xte)
            preds[(name, ver)] = pred
            how = {"Base": "mặc định", "Tune 1": "tốt nhất theo CV",
                   "Tune 2": cfgs[name]["how2"]}[ver]
            rows.append({
                "Dataset": ds_name, "Model": name, "Version": ver,
                "Params": show_params(params), **param_cols(est), "Cách chọn": how,
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
    grid_report(key, ds_name, Xtr, ytr, Xte, yte, cfgs)
    return df



# ---------------------------------------------------------------------------
# KẾT QUẢ TỪNG THAM SỐ TRONG LƯỚI
# ---------------------------------------------------------------------------
def grid_report(key, ds_name, Xtr, ytr, Xte, yte, cfgs):
    """Chạy MỌI cấu hình trong lưới: CV (trên train) + kết quả test để minh hoạ.
    Chỉ CV được dùng để chọn Tune 1/2; cột test chỉ để quan sát, không dùng chọn tham số."""
    rows = []
    for name, (est, grid) in tc.model_specs().items():
        for params in ParameterGrid(grid):
            e = clone(est).set_params(**params)
            sc = cross_val_score(e, Xtr, ytr, scoring="f1_macro", cv=tc.CV, n_jobs=-1)
            m, t_train = tc.timed_fit(e, Xtr, ytr)
            pred, _ = tc.timed_predict(m, Xte)
            tags = [v for v in ("Tune 1", "Tune 2") if cfgs[name][v] == params]
            rows.append({"Dataset": ds_name, "Model": name, "Params": show_params(params),
                         **param_cols(e),
                         "CV Macro-F1": sc.mean(), "CV std": sc.std(),
                         "Test Accuracy": accuracy_score(yte, pred),
                         "Test Macro-F1": f1_score(yte, pred, average="macro"),
                         "T_train (s)": t_train, "Được chọn làm": " + ".join(tags)})
    g = pd.DataFrame(rows)
    g["Hạng CV"] = g.groupby("Model")["CV Macro-F1"].rank(ascending=False, method="first").astype(int)
    g = g.sort_values(["Model", "Hạng CV"]).reset_index(drop=True)
    g.to_csv(f"{OUT}/grid_{key}.csv", index=False, encoding="utf-8-sig")

    print(f"\n--- Kết quả từng tham số trong lưới ({ds_name}) ---")
    for name, sub in g.groupby("Model", sort=False):
        print(f"\n{name}")
        cols_p = [c for c in PCOLS if (sub[c] != "-").any()]
        print(sub[["Hạng CV", *cols_p, "CV Macro-F1", "CV std", "Test Macro-F1",
                   "T_train (s)", "Được chọn làm"]].round(4).to_string(index=False))
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(sub) + 1.6))
        sub = sub.iloc[::-1]
        colors = ["#d9534f" if "Tune 1" in t else "#f0ad4e" if "Tune 2" in t else "#4a7db8"
                  for t in sub["Được chọn làm"]]
        ax.barh(sub["Params"], sub["CV Macro-F1"], xerr=sub["CV std"], color=colors, alpha=.85)
        ax.scatter(sub["Test Macro-F1"], sub["Params"], color="black", marker="D", s=22, zorder=3)
        lo = (sub["CV Macro-F1"] - sub["CV std"]).min()
        ax.set_xlim(max(0, lo - 0.02), min(1, sub[["CV Macro-F1", "Test Macro-F1"]].max().max() + 0.02))
        for y, (_, r) in enumerate(sub.iterrows()):
            if r["Được chọn làm"]:
                ax.text(ax.get_xlim()[0], y, f" {r['Được chọn làm']}", va="center", fontsize=8, color="white",
                        fontweight="bold")
        ax.set_xlabel("Macro-F1   (thanh = CV trên train, ◆ = test)")
        ax.set_title(f"{name} - {ds_name}\nđỏ = Tune 1, cam = Tune 2", fontsize=10)
        plt.tight_layout(); plt.savefig(f"{IMG}/grid_{key}_{name}.png", dpi=150); plt.close()
    return g

# ---------------------------------------------------------------------------
# HÌNH ẢNH
# ---------------------------------------------------------------------------
def plot_confusions(df, preds, yte, key, ds_name):
    """Mỗi mô hình 1 ảnh: 3 ma trận nhầm lẫn (Base / Tune 1 / Tune 2)."""
    for name in df["Model"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, ver in zip(axes, VERS):
            r = df[(df["Model"] == name) & (df["Version"] == ver)].iloc[0]
            sns.heatmap(confusion_matrix(yte, preds[(name, ver)]), annot=True, fmt="d",
                        cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Tiêu cực", "Tích cực"],
                        yticklabels=["Tiêu cực", "Tích cực"])
            ax.set_title(f"{ver}: {r['Params']}\n"
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
