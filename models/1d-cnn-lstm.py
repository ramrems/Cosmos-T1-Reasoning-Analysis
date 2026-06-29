import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================
# 1. AYARLAR VE HİPERPARAMETRELER
# ==========================================
MAX_LEN = 512         # Maksimum token (dizi) uzunluğu
BATCH_SIZE = 32       # Batch boyutu
EPOCHS = 10           # Eğitim döngüsü sayısı
LEARNING_RATE = 1e-3  # Öğrenme oranı
BASE_DIR = "g_gemma_aligned_folds"

# Cihaz ayarı (Ekran kartı varsa kullan, yoksa işlemciye geç)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}\n")

# ==========================================
# 2. VERİ SETİ (DATASET) SINIFI VE PADDING
# ==========================================
class SequenceDataset(Dataset):
    def __init__(self, df, max_len):
        self.max_len = max_len
        self.labels = df['is_correct'].astype(int).values
        self.sequences = df['top1_top2_list'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Boş dizi kontrolü
        if not isinstance(seq, (list, np.ndarray)) or len(seq) == 0:
            seq = [0.0]
            
        # Diziyi numpy dizisine çevir
        seq = np.array(seq, dtype=np.float32)
        
        # PADDING / TRUNCATION İŞLEMİ
        if len(seq) > self.max_len:
            # Dizi çok uzunsa sonundan kırp
            padded_seq = seq[:self.max_len]
        else:
            # Dizi kısaysa sonuna 0'lar (padding) ekle
            pad_length = self.max_len - len(seq)
            padded_seq = np.pad(seq, (0, pad_length), 'constant', constant_values=0.0)
            
        # PyTorch formatına (Tensor) çevir. Boyut: [MAX_LEN, 1]
        x_tensor = torch.tensor(padded_seq, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_tensor, y_tensor

# ==========================================
# 3. DERİN ÖĞRENME MİMARİLERİ
# ==========================================

# A. 1D Evrişimli Sinir Ağı (1D-CNN)
class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        # PyTorch Conv1d girdisi (Batch, Kanal, Uzunluk) şeklindedir
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * (MAX_LEN // 4), 1) # Pooling ile uzunluk 4'e bölündü
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x boyutu: [Batch, Length, Channels] -> [Batch, Channels, Length] yapıyoruz
        x = x.permute(0, 2, 1) 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Düzleştir (Flatten)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze()

# B. Uzun Kısa-Süreli Bellek Ağı (LSTM)
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x boyutu: [Batch, Length, 1]
        out, (hn, cn) = self.lstm(x)
        # Son zaman adımının çıktısını alıyoruz
        last_out = out[:, -1, :] 
        last_out = self.dropout(last_out)
        final_out = self.fc(last_out)
        return self.sigmoid(final_out).squeeze()

# ==========================================
# 4. EĞİTİM VE TEST DÖNGÜSÜ
# ==========================================
print("===== DERİN ÖĞRENME 5-FOLD ÇAPRAZ DOĞRULAMA BAŞLIYOR =====\n")

results = {
    "1D-CNN": {"acc": [], "auc": []},
    "LSTM":   {"acc": [], "auc": []}
}

for fold in range(1, 6):
    fold_dir = os.path.join(BASE_DIR, f"fold_{fold}")
    
    # Ham PKL dosyalarını oku (Bunları daha önce bölmüştük)
    train_df = pd.read_pickle(os.path.join(fold_dir, "train_raw.pkl"))
    test_df = pd.read_pickle(os.path.join(fold_dir, "test_raw.pkl"))
    
    # Dataset ve DataLoader oluştur
    train_dataset = SequenceDataset(train_df, MAX_LEN)
    test_dataset = SequenceDataset(test_df, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"--- FOLD {fold} Eğitimleri ---")
    
    # Her fold için modelleri sıfırdan oluştur
    models = {
        "1D-CNN": Simple1DCNN().to(device),
        "LSTM": SimpleLSTM().to(device)
    }
    
    for model_name, model in models.items():
        criterion = nn.BCELoss() # İkili sınıflandırma kayıp fonksiyonu
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # --- MODEL EĞİTİMİ (TRAIN) ---
        model.train()
        for epoch in range(EPOCHS):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
        # --- MODEL TESTİ (EVALUATION) ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                predictions = model(batch_x)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                
        # Metrikleri hesapla
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        pred_classes = (all_preds >= 0.5).astype(int)
        acc = accuracy_score(all_labels, pred_classes)
        auc = roc_auc_score(all_labels, all_preds)
        
        results[model_name]["acc"].append(acc)
        results[model_name]["auc"].append(auc)
        
        print(f"  > {model_name:<7} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    print()

# ==========================================
# 5. GENEL SONUÇLAR
# ==========================================
print("=====================================================")
print("     DERİN ÖĞRENME 5-FOLD GENEL PERFORMANS ÖZETİ")
print("=====================================================")
print(f"{'Algoritma':<15} | {'Ortalama Accuracy':<20} | {'Ortalama ROC-AUC'}")
print("-" * 55)

for model_name, metrics in results.items():
    mean_acc = np.mean(metrics["acc"])
    std_acc = np.std(metrics["acc"])
    mean_auc = np.mean(metrics["auc"])
    std_auc = np.std(metrics["auc"])
    
    acc_str = f"{mean_acc:.4f} (±{std_acc:.4f})"
    auc_str = f"{mean_auc:.4f} (±{std_auc:.4f})"
    
    print(f"{model_name:<15} | {acc_str:<20} | {auc_str}")
print("=====================================================")