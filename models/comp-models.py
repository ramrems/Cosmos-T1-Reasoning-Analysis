import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# =====================================================================
# 1. AYARLAR VE KLASÖR YOLU
# =====================================================================
base_dir = "g_gemma_aligned_folds"

# =====================================================================
# 2. MODELLERİN TANIMLANMASI
# =====================================================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss', n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        # Scikit-learn'ün yerleşik Gradient Boosting algoritması
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        # Temel doğrusal sınıflandırıcı
        max_iter=1000, random_state=42
    ),
    "SVM (RBF Kernel)": SVC(
        # Doğrusal olmayan ilişkileri ayırmak için RBF kernel (ROC-AUC için probability=True şart)
        probability=True, random_state=42
    ),
    "MLP (Sinir Ağı)": MLPClassifier(
        # Sklearn içindeki temel Derin Öğrenme / Çok Katmanlı Algılayıcı
        hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
    )
}

# Sonuçları saklayacağımız yapı
results = {name: {"acc": [], "auc": []} for name in models.keys()}

print("===== 6 FARKLI ALGORİTMA İLE 5-FOLD ÇAPRAZ DOĞRULAMA =====\n")

# =====================================================================
# 3. EĞİTİM VE TEST DÖNGÜSÜ
# =====================================================================
for fold in range(1, 6):
    fold_dir = os.path.join(base_dir, f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_features.csv")
    test_path = os.path.join(fold_dir, "test_features.csv")
    
    if not os.path.exists(train_path):
        print(f"[UYARI] {fold_dir} içinde CSV bulunamadı!")
        continue

    # Verileri Yükle
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Hedef Değişken (y) Hazırlığı
    y_train = train_df['is_correct'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(train_df['is_correct']).astype(int)
    y_test = test_df['is_correct'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(test_df['is_correct']).astype(int)

    # Girdi Özellikleri (X) Hazırlığı
    cols_to_drop = ['is_correct']
    if 'question_id' in train_df.columns:
        cols_to_drop.append('question_id')
        
    X_train_raw = train_df.drop(columns=cols_to_drop, errors='ignore')
    X_test_raw = test_df.drop(columns=cols_to_drop, errors='ignore')

    # --- ÖLÇEKLENDİRME (SCALING) ADIMI ÇOK ÖNEMLİ ---
    # SVM, Lojistik Regresyon ve MLP scaling olmadan çalışamaz/yanlış çalışır.
    # Veri sızıntısını önlemek için scaler sadece Train verisi ile eğitilir (fit), Test'e sadece uygulanır (transform).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"--- FOLD {fold} İşleniyor ---")

    for model_name, model in models.items():
        # Eğitim
        model.fit(X_train, y_train)
        
        # Tahmin
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrik
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[model_name]["acc"].append(acc)
        results[model_name]["auc"].append(auc)
        
        print(f"  > {model_name:<20} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    print()

# =====================================================================
# 4. SONUÇ RAPORLAMA
# =====================================================================
print("================================================================")
print("              TÜM ALGORİTMALARIN PERFORMANS ÖZETİ               ")
print("================================================================")
print(f"{'Algoritma':<22} | {'Ortalama Accuracy':<20} | {'Ortalama ROC-AUC'}")
print("-" * 64)

# Sonuçları ROC-AUC'ye göre büyükten küçüğe sıralayarak basalım
summary_list = []
for model_name, metrics in results.items():
    if len(metrics["auc"]) > 0:
        mean_acc = np.mean(metrics["acc"])
        std_acc = np.std(metrics["acc"])
        mean_auc = np.mean(metrics["auc"])
        std_auc = np.std(metrics["auc"])
        summary_list.append((model_name, mean_acc, std_acc, mean_auc, std_auc))

summary_list.sort(key=lambda x: x[3], reverse=True) # AUC'ye göre sırala

for name, m_acc, s_acc, m_auc, s_auc in summary_list:
    acc_str = f"{m_acc:.4f} (±{s_acc:.4f})"
    auc_str = f"{m_auc:.4f} (±{s_auc:.4f})"
    print(f"{name:<22} | {acc_str:<20} | {auc_str}")

print("================================================================")