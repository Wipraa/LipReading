# 01 — Arsitektur Model

---

## 1. Gambaran Umum Tiga Varian Model

Penelitian ini merancang tiga varian model untuk dibandingkan:

| Varian | Input | Tujuan |
|--------|-------|--------|
| Baseline Conv-LSTM (M01) | RGB `[30, 96, 96, 3]` | Titik acuan perbandingan |
| Single-Stream LK P17 | Optical Flow `[29, 96, 96, 3]` | Menguji optical flow saja |
| Dual-Stream LK P17 + M02 | RGB + Optical Flow | Model utama / terbaik |

---

## 2. Arsitektur Baseline Conv-LSTM

### Deskripsi Naratif

Arsitektur baseline Conv-LSTM memproses sekuens frame RGB melalui dua tahap berturutan. **Tahap 1 — Ekstraksi fitur spasial**: tiga blok Conv2D berjalan secara time-distributed (diterapkan independen pada setiap frame), mengekstraksi fitur dengan filter 32, 64, dan 128 kanal, masing-masing diikuti MaxPooling 2×2 yang mereduksi dimensi spasial dari 96×96 menjadi 12×12. **Tahap 2 — Pemrosesan temporal**: dua layer ConvLSTM memproses sekuens fitur spasial sebagai urutan waktu dengan hidden channel masing-masing 128 dan 64, menangkap dinamika gerak bibir lintas frame. Global Average Pooling mengubah representasi 64×12×12 menjadi vektor 64-dimensi, yang kemudian diproses oleh dua layer fully connected (Linear 64→256 dengan Dropout 0.5, lalu Linear 256→18) untuk menghasilkan logit 18 kelas.

### Diagram Tekstual (per frame → temporal)

```
Input RGB
[B, 3, 30, 96, 96]
        │
        ▼ (time-distributed)
┌─────────────────────┐
│  Conv2D Block 1     │  32 filter, kernel 3×3, ReLU, MaxPool 2×2
│  Output: 32×48×48   │
├─────────────────────┤
│  Conv2D Block 2     │  64 filter, kernel 3×3, ReLU, MaxPool 2×2
│  Output: 64×24×24   │
├─────────────────────┤
│  Conv2D Block 3     │  128 filter, kernel 3×3, ReLU, MaxPool 2×2
│  Output: 128×12×12  │
└─────────────────────┘
        │
        ▼ (sekuens spasial → ConvLSTM)
┌─────────────────────┐
│  ConvLSTM Layer 1   │  128 hidden channel, kernel 3×3
│  Output: [B, T, 128, 12, 12]
├─────────────────────┤
│  ConvLSTM Layer 2   │  64 hidden channel, kernel 3×3
│  Output: [B, T, 64, 12, 12]
└─────────────────────┘
        │
        ▼ (ambil frame terakhir)
┌─────────────────────┐
│  Global Avg Pooling │  64×1×1 → flatten
│  Output: [B, 64]    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  FC Classifier      │  Linear(64→256) → Dropout(0.5) → ReLU
│                     │  Linear(256→18)
│  Output: [B, 18]    │
└─────────────────────┘
        │
        ▼
   Klasifikasi 18 kelas
```

### Tabel Layer — Baseline M01

| No | Layer | Output Shape | Keterangan |
|----|-------|-------------|------------|
| 1 | Input | `[B, 3, 30, 96, 96]` | Sekuens 30 frame RGB |
| 2 | Conv2D Block 1 | `[B×30, 32, 48, 48]` | Time-distributed, MaxPool 2×2 |
| 3 | Conv2D Block 2 | `[B×30, 64, 24, 24]` | Time-distributed, MaxPool 2×2 |
| 4 | Conv2D Block 3 | `[B×30, 128, 12, 12]` | Time-distributed, MaxPool 2×2 |
| 5 | ConvLSTM-1 | `[B, 30, 128, 12, 12]` | Hidden 128, return sequences |
| 6 | ConvLSTM-2 | `[B, 30, 64, 12, 12]` | Hidden 64, return last |
| 7 | Global Avg Pooling | `[B, 64]` | Spatial pooling + flatten |
| 8 | FC (256) + Dropout | `[B, 256]` | Dropout 0.5 |
| 9 | FC (18) / Softmax | `[B, 18]` | Output logit 18 kelas |

---

## 3. Arsitektur Dual-Stream (Model Terbaik: P17 + M02)

Model ini memproses dua jalur input secara paralel melalui dua modul **StreamEncoder** yang independen.

### Deskripsi Naratif

Kedua StreamEncoder berbagi arsitektur identik (Conv2D ×3 + ConvLSTM ×2 + GlobalAvgPool) namun dilatih secara independen. Stream pertama memproses input optical flow Lucas-Kanade P17 berukuran [B, 3, 29, 96, 96] (29 pasang frame), sementara stream kedua memproses input RGB [B, 3, 30, 96, 96]. Masing-masing StreamEncoder menghasilkan representasi vektor 64-dimensi, disebut `f_flow` dan `f_rgb`.

Mekanisme **AttentionFusion** menggabungkan kedua representasi secara adaptif melalui gate sigmoid yang dipelajari selama training. Vektor konkatenasi [f_rgb || f_flow] ∈ R^128 diproyeksikan ke dua vektor gate:

- `g_rgb = sigmoid(W_rgb · [f_rgb || f_flow])` — bobot kontribusi stream RGB
- `g_flow = sigmoid(W_flow · [f_rgb || f_flow])` — bobot kontribusi stream optical flow

Fitur gabungan dihitung sebagai: `f_fused = g_rgb ⊙ f_rgb + g_flow ⊙ f_flow`, di mana ⊙ adalah perkalian elemen demi elemen, dan W_rgb, W_flow ∈ R^{64×128} adalah parameter yang dapat dilatih. Mekanisme ini memungkinkan model secara otomatis menyesuaikan kontribusi setiap stream: ketika kualitas optical flow rendah, model meningkatkan bobot g_rgb, dan sebaliknya. Arsitektur M02 (LSTM hidden 128, 128) dipilih sebagai konfigurasi terbaik berdasarkan hasil OFAT.

### Diagram Arsitektur Dual-Stream

```
Optical Flow Input          RGB Input
[B, 3, 29, 96, 96]         [B, 3, 30, 96, 96]
        │                           │
        ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  StreamEncoder  │         │  StreamEncoder  │
│  (Flow Stream)  │         │  (RGB Stream)   │
│                 │         │                 │
│ Conv2D 32@48x48 │         │ Conv2D 32@48x48 │
│ Conv2D 64@24x24 │         │ Conv2D 64@24x24 │
│ Conv2D128@12x12 │         │ Conv2D128@12x12 │
│ ConvLSTM 128    │         │ ConvLSTM 128    │  ← M02: hidden 128,128
│ ConvLSTM 128    │         │ ConvLSTM 128    │
│ GlobalAvgPool   │         │ GlobalAvgPool   │
│ f_flow: [B, 64] │         │ f_rgb:  [B, 64] │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
          ┌─────────────────────┐
          │   AttentionFusion   │
          │                     │
          │  concat([f_rgb,     │
          │          f_flow])   │  → [B, 128]
          │  g_rgb  = σ(W_rgb · │    concat)
          │  g_flow = σ(W_flow ·│    concat)
          │  f_fused = g_rgb⊙f_rgb │
          │           + g_flow⊙f_flow
          │  Output: [B, 64]    │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │   FC Classifier     │
          │  Linear(64→256)     │
          │  Dropout(0.5)→ReLU  │
          │  Linear(256→18)     │
          └──────────┬──────────┘
                     ▼
              Klasifikasi 18 kelas
```

---

## 4. Variasi Arsitektur OFAT (Skenario 4)

Pengujian dilakukan pada 6 variasi arsitektur (dataset Gabungan 18 kelas):

| ID | CNN Channels | LSTM Hidden | Dropout | FC Hidden | Test Acc | F1 (w) | Best Epoch | Waktu |
|----|-------------|-------------|---------|-----------|----------|--------|------------|-------|
| **M01** | 32, 64, 128 | **128, 64** | 0.5 | 256 | 83.06% | 82.96% | 48 | 98.1 mnt |
| **M02** ✓ | 32, 64, 128 | **128, 128** | 0.5 | 256 | **85.00%** | **84.83%** | 59 | 116.6 mnt |
| M03 | 32, 64, 128 | 256, 128 | 0.5 | 256 | 80.83% | 80.76% | 98 | 181.2 mnt |
| M04 | 32, 64, 128 | 128, 128, 64 | 0.5 | 256 | 85.00% | 85.00% | 99 | 130.5 mnt |
| M05 | 64, 128, 256 | 128, 64 | 0.5 | 256 | 81.67% | 81.62% | 61 | 161.1 mnt |
| M06 | 32, 64, 128 | 128, 64 | **0.3** | 256 | 84.72% | 84.75% | 93 | 98.0 mnt |

✓ M02 dipilih sebagai arsitektur terbaik: konvergensi lebih cepat (epoch 59 vs 99) dan waktu training lebih efisien (116.6 vs 130.5 mnt) dibanding M04 dengan test accuracy setara.

---

## 5. Hyperparameter Pelatihan

| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate awal | 0.001 |
| Weight Decay | 1e-5 |
| Batch Size | 8 |
| Maksimum Epoch | 100 |
| LR Scheduler | ReduceLROnPlateau (faktor 0.5) |
| Patience (baseline) | 5 epoch |
| Patience (dual-stream) | 8 epoch |
| Loss Function | CrossEntropyLoss |
| Gradient Clipping | Ya (dual-stream) |
| Dropout | 0.5 (M01–M05), 0.3 (M06) |

