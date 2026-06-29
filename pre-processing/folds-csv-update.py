import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

# =====================================================================
# 1. KULLANICI FONKSİYONU (Senin Orijinal Kodun)
# =====================================================================
def extract_features(seq):
    seq = np.array(seq, dtype=float)
    L   = len(seq)
    
    # feature_names'i dinamik çekmek için önce fonksiyonun içini belirliyoruz
    if L == 0:
        # Hata durumunda dönecek boş yapıyı aşağıda manuel tanımladım ki hata vermesin
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "median": np.nan, 
            "range": np.nan, "iqr": np.nan, "p10": np.nan, "p25": np.nan, "p75": np.nan, 
            "p90": np.nan, "mu_early": np.nan, "mu_mid": np.nan, "mu_late": np.nan, 
            "early_mid_diff": np.nan, "mid_late_diff": np.nan, "early_late_diff": np.nan, 
            "slope": np.nan, "r2": np.nan, "intercept": np.nan, "ratio_below_01": np.nan, 
            "ratio_below_03": np.nan, "ratio_below_05": np.nan, "ratio_below_08": np.nan, 
            "ratio_above_09": np.nan, "ratio_above_099": np.nan, "n_dips": 0, 
            "mean_dip_height": np.nan, "max_dip_height": np.nan, "first_10_mean": np.nan, 
            "last_10_mean": np.nan, "first_50_mean": np.nan, "last_50_mean": np.nan, 
            "first_last_diff": np.nan, "entropy_mean": np.nan, "entropy_std": np.nan, 
            "entropy_early": np.nan, "entropy_late": np.nan, "cv": np.nan, "skewness": np.nan, 
            "kurtosis": np.nan, "autocorr_lag1": np.nan, "length": 0, "length_normalized": np.nan, 
            "slope_late": np.nan, "min_position": np.nan
        }

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
    p1 = np.clip(0.5 + seq / 2, 1e-9, 1 - 1e-9)
    p2 = np.clip(0.5 - seq / 2, 1e-9, 1 - 1e-9)
    entropy_approx = -(p1 * np.log(p1) + p2 * np.log(p2))

    return {
        "mean":              np.mean(seq),
        "std":               np.std(seq),
        "min":               np.min(seq),
        "max":               np.max(seq),
        "median":            np.median(seq),
        "range":             np.max(seq) - np.min(seq),
        "iqr":               np.percentile(seq, 75) - np.percentile(seq, 25),
        "p10":               np.percentile(seq, 10),
        "p25":               np.percentile(seq, 25),
        "p75":               np.percentile(seq, 75),
        "p90":               np.percentile(seq, 90),
        "mu_early":          np.mean(early),
        "mu_mid":            np.mean(mid),
        "mu_late":           np.mean(late),
        "early_mid_diff":    np.mean(early) - np.mean(mid),
        "mid_late_diff":     np.mean(mid)   - np.mean(late),
        "early_late_diff":   np.mean(early) - np.mean(late),
        "slope":             slope,
        "r2":                r2,
        "intercept":         intercept,
        "ratio_below_01":    np.mean(seq < 0.1),
        "ratio_below_03":    np.mean(seq < 0.3),
        "ratio_below_05":    np.mean(seq < 0.5),
        "ratio_below_08":    np.mean(seq < 0.8),
        "ratio_above_09":    np.mean(seq > 0.9),
        "ratio_above_099":   np.mean(seq > 0.99),
        "n_dips":            n_dips,
        "mean_dip_height":   np.mean(dip_heights),
        "max_dip_height":    np.max(dip_heights),
        "first_10_mean":     np.mean(seq[:10])  if L >= 10 else np.mean(seq),
        "last_10_mean":      np.mean(seq[-10:]) if L >= 10 else np.mean(seq),
        "first_50_mean":     np.mean(seq[:50])  if L >= 50 else np.mean(seq),
        "last_50_mean":      np.mean(seq[-50:]) if L >= 50 else np.mean(seq),
        "first_last_diff":   np.mean(seq[:10]) - np.mean(seq[-10:]) if L >= 20 else 0,
        "entropy_mean":      np.mean(entropy_approx),
        "entropy_std":       np.std(entropy_approx),
        "entropy_early":     np.mean(entropy_approx[:q1]),
        "entropy_late":      np.mean(entropy_approx[q3:]),
        "cv":                np.std(seq) / (np.mean(seq) + 1e-9),
        "skewness":          stats.skew(seq),
        "kurtosis":          stats.kurtosis(seq),
        "autocorr_lag1":     np.corrcoef(seq[:-1], seq[1:])[0, 1] if L > 2 else 0,
        "length":            L,
        "length_normalized": L / 512,
        "slope_late":        stats.linregress(np.arange(len(late)), late)[0] if len(late) > 2 else 0,
        "min_position":      np.argmin(seq) / L,
    }

# Pandas apply için sarmalayıcı fonksiyon (DataFrame'in bir satırını alır)
def process_row_to_features(row):
    # Orijinal kodundaki gibi ham dizi "top1_top2_list" anahtarında/sütununda tutuluyor
    ham_dizi = row['top1_top2_list']
    
    # Senin fonksiyonunu çağırıp özellik sözlüğünü al
    features = extract_features(ham_dizi)
    
    # Orijinal kodunda yaptığın gibi hedef ve ID değişkenlerini ekle
    features['is_correct'] = int(row['is_correct'])
    if 'question_id' in row:
        features['question_id'] = row['question_id']
        
    return features

# =====================================================================
# 2. OTOMATİK FOLD İŞLEYİCİ
# =====================================================================
base_dir = "g_gemma_aligned_folds"

print(f"{base_dir} klasöründeki fold'lar işleniyor...\n")

for fold in range(1, 6):
    fold_dir = os.path.join(base_dir, f"fold_{fold}")
    train_pkl_path = os.path.join(fold_dir, "train_raw.pkl")
    test_pkl_path = os.path.join(fold_dir, "test_raw.pkl")
    
    if not os.path.exists(train_pkl_path):
        print(f"[UYARI] {fold_dir} içinde train_raw.pkl bulunamadı. Atlanıyor.")
        continue

    print(f"--- FOLD {fold} İçin Özellikler Çıkarılıyor ---")
    
    # PKL'leri DataFrame olarak yükle
    train_df_raw = pd.read_pickle(train_pkl_path)
    test_df_raw = pd.read_pickle(test_pkl_path)
    
    # Eğitim Seti (Train) için tüm satırlara fonksiyonu uygula ve DataFrame'e çevir
    train_features_list = train_df_raw.apply(process_row_to_features, axis=1).tolist()
    train_csv = pd.DataFrame(train_features_list)
    train_csv.to_csv(os.path.join(fold_dir, "train_features.csv"), index=False)
    
    # Test Seti (Test) için tüm satırlara fonksiyonu uygula ve DataFrame'e çevir
    test_features_list = test_df_raw.apply(process_row_to_features, axis=1).tolist()
    test_csv = pd.DataFrame(test_features_list)
    test_csv.to_csv(os.path.join(fold_dir, "test_features.csv"), index=False)
    
    print(f"  > Bitti! Train CSV: {len(train_csv)} satır, Test CSV: {len(test_csv)} satır.")

print("\nTüm fold'lar için CSV dosyaları başarıyla oluşturuldu!")
print("Artık daha önce yazdığımız XGBoost/RandomForest test kodunu çalıştırıp gerçek performansını görebilirsin.")