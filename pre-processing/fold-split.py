import os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

# =====================================================================
# 1. KULLANICIYA AİT ÖZNİTELİK ÇIKARMA FONKSİYONU (BURAYI DOLDURACAKSIN)
# =====================================================================
def ozellik_cikar(ham_satir):
    """
    DİKKAT: Kendi CSV oluşturma (mean, std, slope vb. hesaplama) kodunu buraya entegre etmelisin.
    Girdi: DataFrame'in tek bir satırı (Series objesi). 
           İçinde ham olasılık dizileri nerede tutuluyorsa (örn: ham_satir['logits']) onu alıp işlemelisin.
    Çıktı: 50 özniteliği içeren bir dictionary (sözlük)
    """
    # ÖRNEK TASLAK:
    # diziler = ham_satir['top1_top2_diff'] # Kendi sütun adını yaz
    # features = {
    #     'mean': np.mean(diziler),
    #     'std': np.std(diziler),
    #     # ... diğer özelliklerin ...
    # }
    # return features
    pass

# =====================================================================
# 2. PKL DOSYASININ YÜKLENMESİ VE DATAFRAME'E ÇEVRİLMESİ (DÜZELTİLEN KISIM)
# =====================================================================
print("Ana PKL dosyası yükleniyor...")
pkl_dosya_yolu = 'data\samples_infos_combined.pkl' # Kendi dosya adını buraya yaz

# PKL'yi sözlük olarak okuyoruz
ham_sozluk = pd.read_pickle(pkl_dosya_yolu)

# 'samples' anahtarının içindeki veriyi alıp bir DataFrame (Tablo) yapıyoruz!
df_pkl = pd.DataFrame(ham_sozluk['samples'])

print(f"Veri başarıyla tabloya çevrildi. Toplam Soru (Satır) Sayısı: {len(df_pkl)}")
print("Mevcut Sütunlar:", df_pkl.columns.tolist())

# Hedef değişken (CSV'dekiyle aynı isimde olduğunu varsayıyoruz)
if 'is_correct' in df_pkl.columns:
    y = df_pkl['is_correct']
else:
    # Eğer PKL içinde is_correct farklı bir isimdeyse (örn: label), kodu durdurup uyarı veriyoruz.
    raise KeyError("Tabloda 'is_correct' sütunu bulunamadı. Lütfen yukarıdaki 'Mevcut Sütunlar' çıktısına bakıp doğru ismi yazın.")

# =====================================================================
# 3. KLASÖR YAPISI VE STRATIFIED 5-FOLD BÖLME
# =====================================================================
base_dir = "g_gemma_aligned_folds"
os.makedirs(base_dir, exist_ok=True)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)

for fold_idx, (train_index, test_index) in enumerate(sss.split(df_pkl, y), 1):
    print(f"\n--- FOLD {fold_idx} İŞLENİYOR ---")
    fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # A) PKL verisini bölme (Ham veri)
    train_pkl = df_pkl.iloc[train_index].reset_index(drop=True)
    test_pkl = df_pkl.iloc[test_index].reset_index(drop=True)
    
    # B) Bölünmüş PKL dosyalarını kaydetme (Derin Öğrenme modelleri için)
    train_pkl.to_pickle(os.path.join(fold_dir, "train_raw.pkl"))
    test_pkl.to_pickle(os.path.join(fold_dir, "test_raw.pkl"))
    print("  -> train_raw.pkl ve test_raw.pkl kaydedildi.")
    
    # C) PKL üzerinden özellikleri çıkarıp CSV oluşturma (XGBoost/RF için)
    print("  -> Özellikler çıkarılıyor (CSV oluşturuluyor)... Bu biraz sürebilir.")
    
    try:
        # Train için özellik çıkarımı
        train_features_list = train_pkl.apply(ozellik_cikar, axis=1).tolist()
        train_csv = pd.DataFrame(train_features_list)
        train_csv['is_correct'] = train_pkl['is_correct'] 
        if 'question_id' in train_pkl.columns:
            train_csv['question_id'] = train_pkl['question_id']
        train_csv.to_csv(os.path.join(fold_dir, "train_features.csv"), index=False)
        
        # Test için özellik çıkarımı
        test_features_list = test_pkl.apply(ozellik_cikar, axis=1).tolist()
        test_csv = pd.DataFrame(test_features_list)
        test_csv['is_correct'] = test_pkl['is_correct']
        if 'question_id' in test_pkl.columns:
            test_csv['question_id'] = test_pkl['question_id']
        test_csv.to_csv(os.path.join(fold_dir, "test_features.csv"), index=False)
        
        print("  -> train_features.csv ve test_features.csv kaydedildi.")
    except Exception as e:
        print(f"  [HATA] Özellik çıkarma (ozellik_cikar) sırasında bir hata oluştu: {e}")
        print("  Lütfen 1. Adımdaki fonksiyonu kendi verinize göre güncellediğinizden emin olun.")
        break # Hatayı spamlamaması için döngüyü kırıyoruz

print("\nİşlem tamamlandı!")