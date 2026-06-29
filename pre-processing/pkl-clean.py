import pickle
import re

# ── Yükle ─────────────────────────────────────────────────────
with open("data\samples_infos_combined.pkl", "rb") as f:
    data = pickle.load(f)

samples = data['samples']

# ── Temiz örnek filtresi ───────────────────────────────────────
def is_clean(s):
    text = s['generated_text']
    if '</think>' in text: return True
    if '####' in text: return True
    if re.search(r'[Cc]evap\s*:\s*\d+', text): return True
    if len(s['top1_top2_list']) < 400: return True
    return False

temiz = [s for s in samples if is_clean(s)]

correct = sum(s['is_correct'] for s in temiz)
wrong   = len(temiz) - correct

print(f"Toplam temiz örnek : {len(temiz)}")
print(f"Doğru              : {correct} ({100*correct/len(temiz):.1f}%)")
print(f"Yanlış             : {wrong}  ({100*wrong/len(temiz):.1f}%)")

# ── Birkaç temiz örnek göster ─────────────────────────────────
print("\n" + "="*65)
print("TEMİZ ÖRNEKLER (İLK 5)")
print("="*65)

for s in temiz[:5]:
    text = s['generated_text']
    
    # Neden temiz sayıldı?
    neden = []
    if '</think>' in text:       neden.append("</think> var")
    if '####' in text:           neden.append("#### var")
    if re.search(r'[Cc]evap\s*:\s*\d+', text): neden.append("Cevap: var")
    if len(s['top1_top2_list']) < 400: neden.append(f"kısa ({len(s['top1_top2_list'])} token)")

    print(f"\nSoru ID     : {s['question_id']}")
    print(f"Temiz neden : {', '.join(neden)}")
    print(f"Token sayısı: {len(s['top1_top2_list'])}")
    print(f"Prediction  : {s['prediction']}")
    print(f"Ground Truth: {s['ground_truth']}")
    print(f"is_correct  : {s['is_correct']}")
    print(f"Metin sonu  : ...{repr(text[-200:])}")

# ── Kaydet ────────────────────────────────────────────────────
output = {
    "meta": {
        "kaynak"    : "samples_infos_part1.pkl",
        "filtre"    : "think_bitmis_veya_kisa",
        "n_total"   : len(temiz),
        "n_correct" : correct,
        "accuracy"  : round(correct / len(temiz), 4),
    },
    "samples": temiz
}

with open("samples_infos_clean.pkl", "wb") as f:
    pickle.dump(output, f)

print("\n✓ samples_infos_clean.pkl kaydedildi.")