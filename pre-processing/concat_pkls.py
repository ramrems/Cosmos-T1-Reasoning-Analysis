import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# 1. Parçaları yükle ve birleştir
with open("data\samples_infos_part1.pkl", "rb") as f:
    part1 = pickle.load(f)["samples"]
with open("data\samples_infos_part2.pkl", "rb") as f:
    part2 = pickle.load(f)["samples"]

all_samples = part1 + part2

# 2. Birleştirilmiş veriyi tek bir pkl dosyası olarak kaydet
combined_filename = "samples_infos_combined.pkl"
with open(combined_filename, "wb") as f:
    pickle.dump({"samples": all_samples}, f)

print(f"Birleştirilmiş dosya '{combined_filename}' olarak kaydedildi.\n")

# 3. İçerik Kontrolü (Toplam, Doğru, Yanlış sayısı)
total_questions = len(all_samples)
correct_count = sum(1 for s in all_samples if int(s['is_correct']) == 1)
incorrect_count = total_questions - correct_count

print("--- DOSYA İÇERİK KONTROLÜ ---")
print(f"Toplam soru sayısı: {total_questions}")
print(f"Doğru soru sayısı : {correct_count}")
print(f"Yanlış soru sayısı: {incorrect_count}")
print("-----------------------------\n")

# 4. Matrisleri Oluşturma (Senin yazdığın kısım)
MAX_LEN = 512

X = np.zeros((total_questions, MAX_LEN))
y = np.zeros(total_questions)
lengths = np.zeros(total_questions)  # ekstra feature olarak kullanılabilir

for i, s in enumerate(all_samples):
    seq = s['top1_top2_list']
    L = min(len(seq), MAX_LEN)
    X[i, :L] = seq[:L]
    lengths[i] = len(seq)
    y[i] = int(s['is_correct'])

# 5. Veriyi Eğitim (1000) ve Test (100) olarak ayırma
if total_questions >= 1100:
    # stratify=y parametresi, doğru/yanlış soru dağılımının dengeli olmasını sağlar.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=1000, 
        test_size=100, 
        random_state=42, 
        stratify=y       
    )
    
    # --- Eğitim ve Test Setlerindeki Doğru/Yanlış Sayımlarını Hesapla ---
    train_correct = np.sum(y_train == 1)
    train_incorrect = np.sum(y_train == 0)
    
    test_correct = np.sum(y_test == 1)
    test_incorrect = np.sum(y_test == 0)
    
    print("--- EĞİTİM / TEST AYIRMA SONUCU ---")
    print(f"Eğitim Seti (X_train) boyutu : {X_train.shape}")
    print(f"  -> Eğitim Seti Doğru Sayısı  : {train_correct}")
    print(f"  -> Eğitim Seti Yanlış Sayısı : {train_incorrect}")
    print("-")
    print(f"Test Seti (X_test) boyutu    : {X_test.shape}")
    print(f"  -> Test Seti Doğru Sayısı    : {test_correct}")
    print(f"  -> Test Seti Yanlış Sayısı   : {test_incorrect}")
    print("-----------------------------------")
    
else:
    print(f"HATA: 1000 eğitim ve 100 test verisi ayırmak için en az 1100 örneğe ihtiyacınız var. Mevcut örnek sayısı: {total_questions}")