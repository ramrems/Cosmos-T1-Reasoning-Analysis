"""
Top1-Top2 Uncertainty Analizi
50 feature + özet görseller + ilginç örnek grafikler
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import find_peaks
from pathlib import Path

# ── Ayarlar ───────────────────────────────────────────────────
PKL_PATH   = "data\samples_infos_combined.pkl"   # dosya yolunu güncelle
OUTPUT_DIR = Path("analiz_ciktilari")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── 1. Veri Yükle ─────────────────────────────────────────────
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
samples = data["samples"]
print(f"Toplam: {len(samples)} örnek")

# ── 2. 50 Feature Çıkar ───────────────────────────────────────
def extract_features(seq):
    seq = np.array(seq, dtype=float)
    L   = len(seq)
    if L == 0:
        return {k: np.nan for k in feature_names()}

    # Pozisyon bölgeleri
    q1 = max(1, L // 4)
    q2 = max(2, L // 2)
    q3 = max(3, 3 * L // 4)

    early  = seq[:q1]
    mid    = seq[q1:q3]
    late   = seq[q3:]

    # Trend (slope + r²)
    x      = np.arange(L)
    slope, intercept, r, _, _ = stats.linregress(x, seq)
    r2     = r ** 2

    # Dip (düşük güven) analizi
    inv    = 1 - seq
    peaks, props = find_peaks(inv, height=0.3, distance=5)
    n_dips = len(peaks)
    dip_heights = props["peak_heights"] if n_dips > 0 else [0]

    # Entropy yaklaşımı: top1-top2 farkından
    # p_top1 ≈ 0.5 + diff/2,  p_top2 ≈ 0.5 - diff/2
    p1 = np.clip(0.5 + seq / 2, 1e-9, 1 - 1e-9)
    p2 = np.clip(0.5 - seq / 2, 1e-9, 1 - 1e-9)
    entropy_approx = -(p1 * np.log(p1) + p2 * np.log(p2))

    return {
        # --- Temel istatistikler ---
        "mean":              np.mean(seq),
        "std":               np.std(seq),
        "min":               np.min(seq),
        "max":               np.max(seq),
        "median":            np.median(seq),
        "range":             np.max(seq) - np.min(seq),
        "iqr":               np.percentile(seq, 75) - np.percentile(seq, 25),

        # --- Percentile'lar ---
        "p10":               np.percentile(seq, 10),
        "p25":               np.percentile(seq, 25),
        "p75":               np.percentile(seq, 75),
        "p90":               np.percentile(seq, 90),

        # --- Literatürdeki 3 bölge ortalaması ---
        "mu_early":          np.mean(early),
        "mu_mid":            np.mean(mid),
        "mu_late":           np.mean(late),

        # --- Bölgeler arası farklar ---
        "early_mid_diff":    np.mean(early) - np.mean(mid),
        "mid_late_diff":     np.mean(mid)   - np.mean(late),
        "early_late_diff":   np.mean(early) - np.mean(late),

        # --- Trend (literatürdeki slope ve r²) ---
        "slope":             slope,
        "r2":                r2,
        "intercept":         intercept,

        # --- Eşik bazlı oranlar ---
        "ratio_below_01":    np.mean(seq < 0.1),
        "ratio_below_03":    np.mean(seq < 0.3),
        "ratio_below_05":    np.mean(seq < 0.5),
        "ratio_below_08":    np.mean(seq < 0.8),
        "ratio_above_09":    np.mean(seq > 0.9),
        "ratio_above_099":   np.mean(seq > 0.99),

        # --- Dip (belirsizlik) analizi ---
        "n_dips":            n_dips,
        "mean_dip_height":   np.mean(dip_heights),
        "max_dip_height":    np.max(dip_heights),

        # --- İlk / son token bölgeleri ---
        "first_10_mean":     np.mean(seq[:10])  if L >= 10 else np.mean(seq),
        "last_10_mean":      np.mean(seq[-10:]) if L >= 10 else np.mean(seq),
        "first_50_mean":     np.mean(seq[:50])  if L >= 50 else np.mean(seq),
        "last_50_mean":      np.mean(seq[-50:]) if L >= 50 else np.mean(seq),
        "first_last_diff":   np.mean(seq[:10]) - np.mean(seq[-10:]) if L >= 20 else 0,

        # --- Entropy yaklaşımı ---
        "entropy_mean":      np.mean(entropy_approx),
        "entropy_std":       np.std(entropy_approx),
        "entropy_early":     np.mean(entropy_approx[:q1]),
        "entropy_late":      np.mean(entropy_approx[q3:]),

        # --- Varyasyon ---
        "cv":                np.std(seq) / (np.mean(seq) + 1e-9),
        "skewness":          stats.skew(seq),
        "kurtosis":          stats.kurtosis(seq),

        # --- Seri otokorelayon (komşu tokenlar ne kadar benzer) ---
        "autocorr_lag1":     np.corrcoef(seq[:-1], seq[1:])[0, 1] if L > 2 else 0,

        # --- Uzunluk ---
        "length":            L,
        "length_normalized": L / 512,

        # --- Son çeyrek slope (cevap bölgesi) ---
        "slope_late":        stats.linregress(np.arange(len(late)), late)[0] if len(late) > 2 else 0,

        # --- Minimum pozisyonu ---
        "min_position":      np.argmin(seq) / L,
    }

def feature_names():
    return list(extract_features(np.ones(10)).keys())

print("Feature'lar çıkarılıyor...")
rows = []
for s in samples:
    feats = extract_features(s["top1_top2_list"])
    feats["is_correct"] = int(s["is_correct"])
    feats["question_id"] = s["question_id"]
    rows.append(feats)

df = pd.DataFrame(rows)
feat_cols = [c for c in df.columns if c not in ("is_correct", "question_id")]
print(f"Toplam feature sayısı: {len(feat_cols)}")
print(df[feat_cols].describe().round(3).to_string())

# Feature'ları kaydet
df.to_csv(OUTPUT_DIR / "features_50.csv", index=False)
print(f"\n✓ features_50.csv kaydedildi ({OUTPUT_DIR})")

# ── 3. Özet: Doğru vs Yanlış Dağılımları ─────────────────────
correct   = df[df.is_correct == 1]
incorrect = df[df.is_correct == 0]
print(f"\nDoğru: {len(correct)} | Yanlış: {len(incorrect)}")

key_features = [
    "mean", "slope", "r2", "mu_early", "mu_mid", "mu_late",
    "n_dips", "entropy_mean", "length", "ratio_below_05",
    "first_last_diff", "autocorr_lag1"
]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Doğru vs Yanlış — Feature Dağılımları (1200 örnek)", fontsize=14, y=1.01)

for ax, feat in zip(axes.flat, key_features):
    c_vals = correct[feat].dropna()
    i_vals = incorrect[feat].dropna()
    ax.hist(c_vals, bins=40, alpha=0.6, color="#1D9E75", label=f"Doğru (n={len(c_vals)})", density=True)
    ax.hist(i_vals, bins=40, alpha=0.6, color="#D85A30", label=f"Yanlış (n={len(i_vals)})", density=True)
    t_stat, p_val = stats.ttest_ind(c_vals, i_vals, equal_var=False)
    ax.set_title(f"{feat}\np={p_val:.3f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel("Yoğunluk", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_dagilimlar.png", bbox_inches="tight")
plt.close()
print("✓ 1_dagilimlar.png")

# ── 4. Pozisyonel Analiz: Bölge Bazlı Güven Profili ──────────
N_BINS = 20
bin_correct   = [[] for _ in range(N_BINS)]
bin_incorrect = [[] for _ in range(N_BINS)]

for s in samples:
    seq = np.array(s["top1_top2_list"])
    if len(seq) == 0:
        continue
    for b in range(N_BINS):
        start = int(b * len(seq) / N_BINS)
        end   = int((b + 1) * len(seq) / N_BINS)
        chunk = seq[start:end]
        if len(chunk) > 0:
            val = np.mean(chunk)
            if s["is_correct"]:
                bin_correct[b].append(val)
            else:
                bin_incorrect[b].append(val)

means_c = [np.mean(b) if b else np.nan for b in bin_correct]
means_i = [np.mean(b) if b else np.nan for b in bin_incorrect]
stds_c  = [np.std(b)  if b else np.nan for b in bin_correct]
stds_i  = [np.std(b)  if b else np.nan for b in bin_incorrect]
xs      = np.linspace(0, 1, N_BINS)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Pozisyonel Güven Profili — Doğru vs Yanlış", fontsize=13)

ax = axes[0]
ax.plot(xs, means_c, color="#1D9E75", lw=2, label="Doğru")
ax.fill_between(xs,
    np.array(means_c) - np.array(stds_c),
    np.array(means_c) + np.array(stds_c),
    alpha=0.2, color="#1D9E75")
ax.plot(xs, means_i, color="#D85A30", lw=2, label="Yanlış")
ax.fill_between(xs,
    np.array(means_i) - np.array(stds_i),
    np.array(means_i) + np.array(stds_i),
    alpha=0.2, color="#D85A30")
ax.set_xlabel("Token pozisyonu (normalize)")
ax.set_ylabel("Ortalama top1-top2 farkı")
ax.set_title("Ortalama ± std")
ax.legend()

ax = axes[1]
diff = np.array(means_c) - np.array(means_i)
colors = ["#1D9E75" if d > 0 else "#D85A30" for d in diff]
ax.bar(xs, diff, width=0.04, color=colors, alpha=0.8)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Token pozisyonu (normalize)")
ax.set_ylabel("Fark (doğru − yanlış)")
ax.set_title("Güven farkı (pozitif = doğrular daha emin)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_pozisyonel_profil.png", bbox_inches="tight")
plt.close()
print("✓ 2_pozisyonel_profil.png")

# ── 5. Slope ve r² Detay ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Slope ve r² Analizi", fontsize=13)

# Slope dağılımı
ax = axes[0, 0]
ax.hist(correct.slope.dropna(),   bins=50, alpha=0.6, color="#1D9E75", label="Doğru",  density=True)
ax.hist(incorrect.slope.dropna(), bins=50, alpha=0.6, color="#D85A30", label="Yanlış", density=True)
ax.set_title("Slope dağılımı")
ax.set_xlabel("Slope (negatif = uncertainty azalıyor)")
ax.legend()

# r² dağılımı
ax = axes[0, 1]
ax.hist(correct.r2.dropna(),   bins=50, alpha=0.6, color="#1D9E75", label="Doğru",  density=True)
ax.hist(incorrect.r2.dropna(), bins=50, alpha=0.6, color="#D85A30", label="Yanlış", density=True)
ax.set_title("r² dağılımı")
ax.set_xlabel("r² (1 = mükemmel lineer)")
ax.legend()

# Slope vs r² scatter
ax = axes[1, 0]
ax.scatter(correct.slope,   correct.r2,   alpha=0.3, s=8, color="#1D9E75", label="Doğru")
ax.scatter(incorrect.slope, incorrect.r2, alpha=0.3, s=8, color="#D85A30", label="Yanlış")
ax.set_xlabel("Slope")
ax.set_ylabel("r²")
ax.set_title("Slope vs r²")
ax.legend()

# µearly, µmid, µlate karşılaştırması
ax = axes[1, 1]
positions  = [1, 2, 3]
labels_pos = ["µearly", "µmid", "µlate"]
data_c = [correct.mu_early, correct.mu_mid, correct.mu_late]
data_i = [incorrect.mu_early, incorrect.mu_mid, incorrect.mu_late]
bp1 = ax.boxplot(data_c, positions=[p - 0.2 for p in positions], widths=0.3,
                 patch_artist=True, boxprops=dict(facecolor="#1D9E75", alpha=0.6))
bp2 = ax.boxplot(data_i, positions=[p + 0.2 for p in positions], widths=0.3,
                 patch_artist=True, boxprops=dict(facecolor="#D85A30", alpha=0.6))
ax.set_xticks(positions)
ax.set_xticklabels(labels_pos)
ax.set_title("µearly / µmid / µlate")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Doğru", "Yanlış"])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_slope_r2.png", bbox_inches="tight")
plt.close()
print("✓ 3_slope_r2.png")

# ── 6. Feature Korelasyon Isı Haritası ────────────────────────
fig, ax = plt.subplots(figsize=(16, 14))
corr = df[feat_cols + ["is_correct"]].corr()["is_correct"].drop("is_correct").sort_values()
colors_bar = ["#D85A30" if v < 0 else "#1D9E75" for v in corr]
bars = ax.barh(corr.index, corr.values, color=colors_bar, alpha=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_title("Her feature'ın is_correct ile korelasyonu", fontsize=13)
ax.set_xlabel("Korelasyon katsayısı")
for bar, val in zip(bars, corr.values):
    ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_korelasyon.png", bbox_inches="tight")
plt.close()
print("✓ 4_korelasyon.png")

# ── 7. İlginç Örnekler: En Çarpıcı 12 Soru ───────────────────
# Seçim kriterleri: slope farkı en büyük, dip sayısı yüksek vb.
df_sort = df.copy()

def plot_single(ax, seq, title, color, feats):
    seq = np.array(seq)
    L   = len(seq)
    x   = np.arange(L)
    ax.plot(x, seq, color=color, lw=0.8, alpha=0.7)
    # Trend çizgisi
    slope, intercept, *_ = stats.linregress(x, seq)
    ax.plot(x, slope * x + intercept, color=color, lw=1.5, ls="--", alpha=0.9)
    # µearly, µmid, µlate
    q1, q3 = L // 4, 3 * L // 4
    ax.axhline(np.mean(seq[:q1]),  color="gray", lw=0.7, ls=":", alpha=0.7, label="µearly")
    ax.axhline(np.mean(seq[q1:q3]),color="gray", lw=1.0, ls=":", alpha=0.7, label="µmid")
    ax.axhline(np.mean(seq[q3:]),  color="gray", lw=0.7, ls=":", alpha=0.7, label="µlate")
    ax.axvline(q1, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(q3, color="gray", lw=0.5, alpha=0.4)
    # Dip'ler
    inv = 1 - seq
    peaks, _ = find_peaks(inv, height=0.3, distance=5)
    if len(peaks) > 0:
        ax.scatter(peaks, seq[peaks], color="red", s=12, zorder=5, label="Dip")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    info = f"slope={feats['slope']:.4f}  r²={feats['r2']:.3f}  dips={int(feats['n_dips'])}"
    ax.set_xlabel(info, fontsize=7)

# İlginç örnek kategorileri
categories = {
    "En dik slope (doğru)":   df[df.is_correct == 1].nsmallest(3, "slope"),
    "En düz slope (yanlış)":  df[df.is_correct == 0].nlargest(3, "slope"),
    "En yüksek r² (doğru)":  df[df.is_correct == 1].nlargest(3, "r2"),
    "En düşük r² (yanlış)":  df[df.is_correct == 0].nsmallest(3, "r2"),
}

for cat_name, subset in categories.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"İlginç Örnekler: {cat_name}", fontsize=12)
    color = "#1D9E75" if "doğru" in cat_name else "#D85A30"
    for ax, (_, row) in zip(axes, subset.iterrows()):
        sid = int(row.question_id)
        s   = next(x for x in samples if x["question_id"] == sid)
        plot_single(ax, s["top1_top2_list"],
                    f"ID#{sid} | {'✓' if row.is_correct else '✗'}",
                    color, row)
    plt.tight_layout()
    fname = cat_name.replace(" ", "_").replace("(", "").replace(")", "").replace("²","2")
    plt.savefig(OUTPUT_DIR / f"5_{fname}.png", bbox_inches="tight")
    plt.close()
    print(f"✓ 5_{fname}.png")

# ── 8. En İlginç Tek Örnekler ─────────────────────────────────
special_cases = [
    ("En uzun cevap", df.nlargest(1, "length").iloc[0]),
    ("En kısa cevap", df.nsmallest(1, "length").iloc[0]),
    ("En çok dip",    df.nlargest(1, "n_dips").iloc[0]),
    ("En yüksek entropy", df.nlargest(1, "entropy_mean").iloc[0]),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Özel Durumlar", fontsize=13)

for ax, (label, row) in zip(axes.flat, special_cases):
    sid = int(row.question_id)
    s   = next(x for x in samples if x["question_id"] == sid)
    color = "#1D9E75" if row.is_correct else "#D85A30"
    plot_single(ax, s["top1_top2_list"],
                f"{label} | ID#{sid} | {'Doğru ✓' if row.is_correct else 'Yanlış ✗'}",
                color, row)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "6_ozel_durumlar.png", bbox_inches="tight")
plt.close()
print("✓ 6_ozel_durumlar.png")

# ── 9. Özet İstatistik Tablosu ────────────────────────────────
print("\n" + "="*70)
print("ÖZET: Doğru vs Yanlış — Temel Feature Karşılaştırması")
print("="*70)
compare_feats = ["mean", "slope", "r2", "mu_early", "mu_mid", "mu_late",
                 "n_dips", "entropy_mean", "length", "ratio_below_05",
                 "first_last_diff", "autocorr_lag1"]
for feat in compare_feats:
    c_m = correct[feat].mean()
    i_m = incorrect[feat].mean()
    t, p = stats.ttest_ind(correct[feat].dropna(), incorrect[feat].dropna())
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"{feat:20s} | Doğru: {c_m:7.4f} | Yanlış: {i_m:7.4f} | p={p:.4f} {sig}")

print(f"\n✓ Tüm görseller → {OUTPUT_DIR}/")
print("Dosyalar:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")