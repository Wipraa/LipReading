# 04 — Hasil Eksperimen

---

## Skenario 1 — Perbandingan Farneback vs Lucas-Kanade

Dataset: Number saja (10 kelas: angka 1–10). Model Conv-LSTM identik untuk kedua metode.

| Metode | Best Val Acc | Best Epoch | Test Acc | F1 Macro | Waktu/Epoch |
|--------|-------------|------------|----------|----------|-------------|
| Farneback | 61.59% | 80/100 | 57.73% | 57.41% | ~71 detik |
| **Lucas-Kanade** | 60.00% | 96/100 | **62.27%** | **61.15%** | ~77 detik |

**Kesimpulan**: Lucas-Kanade dipilih karena test accuracy lebih tinggi (+4.54%), meskipun best val acc Farneback sedikit lebih tinggi. LK menunjukkan generalisasi lebih baik pada data yang belum dilihat.

---

## Skenario 2 — OFAT Parameter Lucas-Kanade

Dataset: Number (10 kelas). Metrik pemilihan: **training accuracy** (kemampuan ekstraksi fitur optimal).

**Parameter terpilih: P17**

| Parameter | Nilai P17 |
|-----------|-----------|
| maxCorners | 300 |
| qualityLevel | 0.005 |
| minDistance | 5 |
| blockSize | 6 |
| winSize | 15×15 |
| maxLevel | 3 |

**Hasil P17**: Training Accuracy = **62.05%**, Test Accuracy = **60.45%**

Tiga pola penting dari OFAT:
1. Menurunkan `qualityLevel` dari 0.01 → 0.005 meningkatkan kemampuan deteksi titik fitur di area bibir
2. Menaikkan `blockSize` dari 3 → 6 memberikan estimasi gradien lebih stabil
3. `maxCorners=300` menghasilkan flow map lebih kaya informasi dibanding `maxCorners=200`

---

## Skenario 3 — Pelatihan Model pada Dataset Gabungan 18 Kelas

### 3A. Baseline Conv-LSTM (RGB, Arsitektur M01)

| Metrik | Nilai |
|--------|-------|
| Test Loss | 0.8074 |
| **Test Accuracy** | **83.06%** |
| Precision (macro) | 83.41% |
| Recall (macro) | 83.06% |
| F1-score (macro) | 82.96% |
| F1-score (weighted) | 82.96% |
| Best Val Accuracy | 84.58% |
| Best Epoch | 48/100 |
| Waktu Training | 98.1 menit |
| Rata-rata waktu/epoch | ~59 detik |

#### Per-Kelas Baseline (Tabel 4.5)

| Kelas | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1 | 0.947 | 0.900 | 0.923 | 20 |
| 2 | 0.947 | 0.900 | 0.923 | 20 |
| 3 | 0.800 | 0.800 | 0.800 | 20 |
| 8 | 0.905 | 0.950 | 0.927 | 20 |
| 9 | 0.900 | 0.900 | 0.900 | 20 |
| 10 | 0.909 | 1.000 | **0.952** | 20 |
| a | 1.000 | 1.000 | **1.000** | 20 |
| b | 0.950 | 0.950 | **0.950** | 20 |
| c | 0.800 | 0.600 | 0.686 | 20 |
| d | 0.615 | 0.800 | 0.696 | 20 |
| e | 0.944 | 0.850 | 0.895 | 20 |
| f | 0.833 | 1.000 | 0.909 | 20 |
| buku | 0.818 | 0.900 | 0.857 | 20 |
| dia | 0.778 | 0.700 | 0.737 | 20 |
| **saya** | 0.571 | 0.600 | **0.585** | 20 |
| **keliling** | 0.684 | 0.650 | **0.667** | 20 |
| kelompok | 0.833 | 0.750 | 0.789 | 20 |
| sekarang | 0.778 | 0.700 | 0.737 | 20 |

Kelas terbaik: a (F1=1.000), 10 (0.952), b (0.950)
Kelas terburuk: saya (F1=0.585), keliling (0.667)

---

### 3B. LK P17 Single-Stream (Optical Flow, Arsitektur M01)

| Metrik | Nilai |
|--------|-------|
| Test Loss | 2.3354 |
| **Test Accuracy** | **51.25%** |
| Precision (macro) | 50.66% |
| Recall (macro) | 51.26% |
| F1-score (macro) | 50.47% |
| F1-score (weighted) | 50.47% |
| Best Val Accuracy | 52.08% |
| Best Epoch | 66/100 |
| Waktu Training | 95.1 menit |
| Rata-rata waktu/epoch | ~57 detik |

#### Per-Kelas Single-Stream LK P17 (Tabel 4.7)

| Kelas | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1 | 0.565 | 0.650 | 0.605 | 20 |
| 2 | 0.688 | 0.550 | 0.611 | 20 |
| 3 | 0.348 | 0.400 | 0.372 | 20 |
| 8 | 0.722 | 0.650 | 0.684 | 20 |
| 9 | 0.500 | 0.526 | 0.513 | 20 |
| 10 | 0.538 | 0.700 | 0.609 | 20 |
| a | 0.550 | 0.550 | 0.550 | 20 |
| **b** | 0.762 | 0.800 | **0.780** | 20 |
| **c** | 0.125 | 0.050 | **0.071** | 20 |
| **d** | 0.167 | 0.200 | **0.128** | 20 |
| e | 0.350 | 0.350 | 0.350 | 20 |
| f | 0.688 | 0.550 | 0.611 | 20 |
| buku | 0.588 | 0.500 | 0.541 | 20 |
| dia | 0.625 | 0.500 | 0.556 | 20 |
| **saya** | 0.294 | 0.250 | **0.270** | 20 |
| keliling | 0.615 | 0.800 | 0.696 | 20 |
| kelompok | 0.538 | 0.700 | 0.609 | 20 |
| sekarang | 0.455 | 0.500 | 0.476 | 20 |

Kelas terbaik: b (F1=0.780)
Kelas terburuk: c (0.071), d (0.128), saya (0.270)

---

### 3C. Dual-Stream LK P17 (Arsitektur M01)

| Metrik | Nilai |
|--------|-------|
| Test Loss | 0.8598 |
| **Test Accuracy** | **86.07%** |
| Precision (macro) | 86.70% |
| Recall (macro) | 86.04% |
| F1-score (macro) | 85.93% |
| F1-score (weighted) | 85.94% |
| Best Val Accuracy | 86.53% |
| Best Epoch | 84/100 |
| Waktu Training | 192.9 menit |
| Rata-rata waktu/epoch | ~115–117 detik |

#### Per-Kelas Dual-Stream LK P17 M01 (Tabel 4.9)

| Kelas | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1 | 0.870 | 1.000 | **0.930** | 20 |
| 2 | 0.857 | 0.900 | 0.878 | 20 |
| 3 | 0.875 | 0.700 | 0.778 | 20 |
| 8 | 0.826 | 0.950 | 0.884 | 20 |
| 9 | 0.933 | 0.737 | 0.824 | 20 |
| 10 | 0.909 | 1.000 | **0.952** | 20 |
| a | 1.000 | 0.900 | 0.947 | 20 |
| b | 0.833 | 1.000 | **0.909** | 20 |
| c | 0.875 | 0.700 | 0.778 | 20 |
| d | 0.692 | 0.900 | 0.783 | 20 |
| e | 0.944 | 0.850 | 0.895 | 20 |
| f | 0.950 | 0.950 | **0.950** | 20 |
| buku | 0.857 | 0.900 | 0.878 | 20 |
| dia | 0.889 | 0.800 | 0.842 | 20 |
| **saya** | 0.727 | 0.800 | **0.762** | 20 |
| keliling | 0.778 | 0.700 | 0.737 | 20 |
| kelompok | 0.889 | 0.800 | 0.842 | 20 |
| sekarang | 0.900 | 0.900 | **0.900** | 20 |

Kelas terbaik: 1 (F1=0.930), 10 (0.952), f (0.950)
Kelas terburuk: saya (0.762) — namun sudah jauh meningkat dari baseline (0.585)

---

## Skenario 4 — OFAT Arsitektur Model Conv-LSTM

Dataset: Gabungan 18 kelas.

| ID | LSTM Hidden | Dropout | Val Acc | Test Acc | F1 (w) | Best Epoch | Waktu |
|----|-------------|---------|---------|----------|--------|------------|-------|
| M01 | 128, 64 | 0.5 | 84.58% | 83.06% | 82.96% | 48 | 98.1 mnt |
| **M02** ✓ | **128, 128** | 0.5 | **85.97%** | **85.00%** | **84.83%** | **59** | 116.6 mnt |
| M03 | 256, 128 | 0.5 | 79.44% | 80.83% | 80.76% | 98 | 181.2 mnt |
| M04 | 128, 128, 64 | 0.5 | 84.03% | 85.00% | 85.00% | 99 | 130.5 mnt |
| M05 | 128, 64 (CNN besar) | 0.5 | 86.25% | 81.67% | 81.62% | 61 | 161.1 mnt |
| M06 | 128, 64 | **0.3** | 85.97% | 84.72% | 84.75% | 93 | 98.0 mnt |

---

## Skenario 5 — Arsitektur M02 pada LK P17 (Single & Dual-Stream)

### 5A. LK P17 Single-Stream dengan M02

| Konfigurasi | LSTM Hidden | Val Acc | Test Acc | F1 (w) | Best Epoch |
|-------------|-------------|---------|----------|--------|------------|
| P17-M01 | 128, 64 | 52.08% | 51.25% | 50.47% | 66 |
| **P17-M02** | **128, 128** | **51.53%** | **53.48%** | **54.03%** | 97 |
| P17-M04 | 128, 128, 64 | 45.42% | 43.73% | 44.27% | 89 |

Peningkatan M02 hanya +2.23% (51.25% → 53.48%), mengkonfirmasi bahwa bottleneck single-stream bukan pada kapasitas arsitektur melainkan pada keterbatasan representasi optical flow itu sendiri.

### 5B. Dual-Stream P17 + M02 (Model Terbaik Keseluruhan)

| Metrik | Nilai |
|--------|-------|
| Test Loss | 0.8598 |
| **Test Accuracy** | **86.63%** |
| Precision (macro) | 86.85% |
| Recall (macro) | 86.63% |
| F1-score (macro) | 86.61% |
| **F1-score (weighted)** | **86.60%** |
| Best Val Accuracy | 87.22% |
| Best Epoch | 83/100 |
| Waktu Training | 225.4 menit |

---

## Perbandingan Keseluruhan Model Dual-Stream (Tabel 4.13)

| Model | LK Params | Arsitektur | Val Acc | Test Acc | F1 (w) | Best Epoch | Waktu |
|-------|-----------|-----------|---------|----------|--------|------------|-------|
| Dual LK Default | BaseLK | M01 (128,64) | 85.56% | 84.17% | 83.92% | 97 | 192.1 mnt |
| Dual LK P17 | P17 | M01 (128,64) | 86.53% | 86.07% | 85.94% | 84 | 192.9 mnt |
| **Dual P17+M02** ✓ | **P17** | **M02 (128,128)** | **87.22%** | **86.63%** | **86.60%** | **83** | 225.4 mnt |
| Baseline RGB | — | M01 (128,64) | 84.58% | 83.06% | 82.96% | 48 | 98.1 mnt |

Kontribusi terpisah:
- Parameter LK Default → P17: **+1.90%** (84.17% → 86.07%)
- Arsitektur M01 → M02: **+0.56%** (86.07% → 86.63%)
- Total peningkatan dari baseline: **+3.57%**

---

## Rekapitulasi Seluruh Eksperimen (Tabel 4.14)

| No | Konfigurasi | Input | Val Acc | Test Acc | F1 (w) | Best Epoch |
|----|-------------|-------|---------|----------|--------|------------|
| 1 | Baseline RGB (M01) | RGB | 84.58% | 83.06% | 82.96% | 48 |
| 2 | LK P17 Single (M01) | Flow | 52.08% | 51.25% | 50.47% | 66 |
| 3 | Dual-Stream LK Default | RGB+Flow | 85.56% | 84.17% | 83.92% | 97 |
| 4 | Dual-Stream LK P17 (M01) | RGB+Flow | 86.53% | 86.07% | 85.94% | 84 |
| 5 | RGB M02 | RGB | 85.97% | 85.00% | 84.83% | 59 |
| 6 | P17-M02 Single | Flow | 51.53% | 53.48% | 54.03% | 97 |
| 7 | P17-M04 Single | Flow | 45.42% | 43.73% | 44.27% | 89 |
| **8** | **Dual-Stream P17+M02** | **RGB+Flow** | **87.22%** | **86.63%** | **86.60%** | **83** |

---

## Analisis Berdasarkan Jumlah Suku Kata (Tabel 4.15)

Rata-rata Recall per kelompok suku kata:

| Suku Kata | Label | Baseline (M01) | LK P17 Single | Dual P17+M02 |
|-----------|-------|---------------|--------------|--------------|
| 1 suku kata | a, b, c, d, e, f | 86.67% | 41.67% | **89.17%** |
| 2 suku kata | 1, 2, 3, buku, dia, saya | 80.00% | 47.50% | **87.50%** |
| 3 suku kata | 8, 9, 10, keliling, kelompok, sekarang | 82.50% | 64.60% | **86.28%** |
| **Rata-rata keseluruhan** | — | 83.06% | 51.25% | **86.63%** |

**Temuan kunci**:
- LK single-stream sangat buruk pada label 1 suku kata (41.67%) karena gerak bibir singkat menghasilkan perbedaan antar frame minimal
- Label 3 suku kata justru relatif lebih baik ditangani LK single-stream (64.60%) karena gerakan lebih panjang dan distinktif
- Dual-stream P17+M02 unggul merata di semua kelompok, dengan peningkatan terbesar pada 2 suku kata (+7.50% dari baseline) dan 1 suku kata (+2.50%)

---

## Analisis Confusion Matrix

### Baseline
- Kelas `c` sering tertukar dengan `e` atau `d` (bentuk mulut mirip secara visual)
- Kelas `saya` dan `dia` sering tertukar (kata ganti dengan pola gerak singkat dan serupa)
- Kelas `keliling` memiliki recall hanya 65% — pola temporal 3 suku kata kurang distinktif

### Single-Stream LK P17
- Penyebaran kesalahan jauh lebih luas (confusion matrix lebih menyebar)
- Kelas `c` dan `d` hampir selalu salah diklasifikasi (F1 di bawah 0.130)
- Kelas `3` (tiga) dan `saya` juga sangat rendah

### Dual-Stream LK P17 M01
- Pola diagonal confusion matrix lebih tegas dibanding baseline dan single-stream
- Kelas `saya` meningkat signifikan: recall 0.60 (baseline) → 0.80 (dual-stream), peningkatan +20%
- Kelas `sekarang` meningkat dari 0.70 → 0.90 recall (+20%)
- Kelas `1` mencapai recall sempurna 1.000
- Hanya kelas `saya` (F1=0.762) dan `keliling` (F1=0.737) yang masih di bawah rata-rata

### Kelas dengan Performa Rendah Secara Konsisten

| Kelas | Baseline F1 | LK Single F1 | Dual M01 F1 | Kemungkinan Penyebab |
|-------|------------|-------------|-------------|---------------------|
| saya | 0.585 | 0.270 | 0.762 | Mirip dengan "dia", pola gerak singkat, variasi antar subjek |
| keliling | 0.667 | 0.696 | 0.737 | Pola 3 suku kata dengan gerak bibir kurang distinktif |
| c | 0.686 | 0.071 | 0.778 | Bentuk mulut mirip huruf e dan d |
| d | 0.696 | 0.128 | 0.783 | Bentuk mulut mirip huruf c dan e |
