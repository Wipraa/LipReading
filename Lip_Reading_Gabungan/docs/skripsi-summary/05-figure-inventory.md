# 05 — Inventaris Figure

Semua figure berasal dari notebook Jupyter atau di-generate saat training.

---

## Figure dari BAB III (Metodologi)

| No | Nama/Judul Figure | Lokasi | Kategori | Deskripsi |
|----|-------------------|--------|----------|-----------|
| G3.1 | Alur Preprosesing Data | Skripsi hal. 28 | `preprocessing-step` | Diagram alir 4 tahap: Frame Extraction → Lip Detection → Frame Selection → Crop & Resize |
| G3.2 | Potongan Video Menjadi Beberapa Frame | Skripsi hal. 28 | `preprocessing-step` | Ilustrasi dari referensi (Jin Ting et al., 2021), bukan dari dataset penelitian |
| G3.3 | Pendeteksian Bibir Menggunakan Koordinat Landmark | Skripsi hal. 29 | `preprocessing-step` | Ilustrasi dari referensi (Jin Ting et al., 2021) |
| G3.4 | Alur Model Optical Flow Conv-LSTM (Baseline) | Skripsi hal. 31 | `arsitektur` | Diagram blok arsitektur baseline: RGB → CNN 3 blok → ConvLSTM × 2 → GAP → FC → Klasifikasi |
| G3.5 | Alur Model Dual Stream Optical Flow Conv-LSTM | Skripsi hal. 33 | `arsitektur` | Diagram blok dual-stream: dua jalur StreamEncoder paralel → AttentionFusion → FC → Klasifikasi |

---

## Figure dari BAB IV (Hasil)

| No | Nama/Judul Figure | Lokasi (Notebook) | Kategori | Deskripsi | Status Export |
|----|-------------------|--------------------|----------|-----------|---------------|
| G4.1 | Contoh Hasil Crop Bibir Setiap Label | `03_Preprocessing_Gabungan.ipynb` | `sample-frame` | Grid 18 gambar crop bibir 96×96, satu sampel per label, menunjukkan konsistensi preprocessing | Perlu diexport |
| G4.2 | Visualisasi 3 Channel Optical Flow (Frame Asli, Channel u, Channel v, Magnitude) | `04c_OptFlow_LK_P17_Preprocessing.ipynb` | `preprocessing-step` | 4 baris × 4 kolom: grayscale frame, channel u (biru-merah), channel v (biru-merah), magnitude (kuning-merah) | Perlu diexport |
| G4.3 | Distribusi Raw Flow Sebelum Normalisasi dan Setelah Normalisasi | `04c_OptFlow_LK_P17_Preprocessing.ipynb` | `preprocessing-step` | Histogram distribusi nilai u, v, magnitude sebelum dan sesudah normalisasi global z-score P17 | Perlu diexport |
| G4.4 | Hasil OFAT Parameter Lucas-Kanade pada Dataset Number | `04_OptFlow_LK_Preprocessing_Gabungan.ipynb` | `lainnya` | Bar chart akurasi train & test untuk 18 konfigurasi (BaseLK, P1–P18), dengan P17 disorot hijau | Perlu diexport |
| G4.5 | Kurva Training dan Validation Loss/Accuracy Model Baseline | `05_Baseline_Training_Gabungan.ipynb` | `training-curve` | Dua panel: (kiri) loss train vs val sepanjang 100 epoch; (kanan) accuracy train vs val, best epoch=48 ditandai | Perlu diexport |
| G4.6 | Penurunan Learning Rate dan Rata-rata Waktu Training per Epoch Model Baseline | `05_Baseline_Training_Gabungan.ipynb` | `training-curve` | Dua panel: (kiri) grafik LR scheduler; (kanan) bar chart waktu per epoch, avg ~59 detik | Perlu diexport |
| G4.7 | Confusion Matrix Baseline | `05_Baseline_Training_Gabungan.ipynb` | `confusion-matrix` | Dua panel: confusion matrix count (kiri) dan normalized (kanan), 18×18, test acc 83.06% | Perlu diexport |
| G4.8 | Bar Chart Akurasi per Kelas Baseline | `05_Baseline_Training_Gabungan.ipynb` | `confusion-matrix` | Bar chart 18 kelas, threshold 70% ditampilkan, overall 83.06%, kelas 'c', 'saya', 'keliling' di bawah threshold (oranye) | Perlu diexport |
| G4.9 | Kurva Training dan Validation Loss/Accuracy Model Lucas-Kanade Single-Stream | `06_LK_Training_Gabungan.ipynb` | `training-curve` | Dua panel: loss dan accuracy, best val acc=52.08% di epoch 66, overfitting signifikan terlihat jelas | Perlu diexport |
| G4.10 | Penurunan Learning Rate dan Rata-rata Waktu Training per Epoch Model LK Single-Stream | `06_LK_Training_Gabungan.ipynb` | `training-curve` | Dua panel: LR scheduler dan waktu per epoch ~57 detik | Perlu diexport |
| G4.11 | Confusion Matrix Lucas-Kanade Single-Stream | `06_LK_Training_Gabungan.ipynb` | `confusion-matrix` | Confusion matrix 18×18 test acc 51.25%, pola diagonal sangat lemah dibanding baseline | Perlu diexport |
| G4.12 | Bar Chart Akurasi per Kelas LK P17 Single-Stream | `06_LK_Training_Gabungan.ipynb` | `confusion-matrix` | Bar chart 18 kelas, sebagian besar oranye/merah (di bawah 70%), kelas 'c' sangat rendah (5%) | Perlu diexport |
| G4.13 | Kurva Training dan Validation Loss/Accuracy Model Dual-Stream | `07d_DualStream_M02_Gabungan.ipynb` | `training-curve` | Dua panel: konvergensi lebih lambat namun stabil, best val acc=86.53% epoch 84 (M01) atau 87.22% epoch 83 (M02) | Perlu diexport |
| G4.14 | Penurunan Learning Rate dan Rata-rata Waktu Training per Epoch Model Dual-Stream | `07d_DualStream_M02_Gabungan.ipynb` | `training-curve` | Dua panel: LR scheduler dan waktu per epoch ~116 detik (M01) / ~avg 116 detik | Perlu diexport |
| G4.15 | Confusion Matrix Dual-Stream | `07d_DualStream_M02_Gabungan.ipynb` | `confusion-matrix` | Confusion matrix 18×18, pola diagonal lebih tegas, test acc 86.07% (M01) / 86.63% (M02) | Perlu diexport |
| G4.16 | Bar Chart Akurasi per Kelas Dual-Stream | `07d_DualStream_M02_Gabungan.ipynb` | `confusion-matrix` | Bar chart 18 kelas, hampir seluruhnya hijau (di atas 70%), overall 86.07% | Perlu diexport |
| G4.17 | Grafik Batang Perbandingan Model Dual-Stream | `07d_DualStream_M02_Gabungan.ipynb` | `lainnya` | Grouped bar chart membandingkan 4 model (Baseline, Dual Default, Dual P17, Dual P17+M02) dengan metrik Val Acc, Test Acc, F1 Weighted | Perlu diexport |

---

## Figure Prioritas untuk Slide Sidang

Berdasarkan nilai informatif untuk presentasi:

| Prioritas | Figure | Alasan |
|-----------|--------|--------|
| ⭐⭐⭐ | G3.4 + G3.5 | Arsitektur baseline dan dual-stream — wajib ada |
| ⭐⭐⭐ | G4.7 + G4.15 | Perbandingan confusion matrix baseline vs dual-stream |
| ⭐⭐⭐ | G4.17 | Grafik perbandingan ringkasan semua model |
| ⭐⭐ | G4.5 + G4.9 + G4.13 | Kurva training tiga model (menunjukkan overfitting single-stream) |
| ⭐⭐ | G4.2 | Visualisasi optical flow — intuitif untuk penguji |
| ⭐⭐ | G4.8 + G4.12 + G4.16 | Bar chart per kelas perbandingan ketiga model |
| ⭐ | G3.1 | Alur preprocessing |
| ⭐ | G4.4 | OFAT parameter LK |

---

## Kode Export Figure (untuk figure yang belum diexport)

Tambahkan di akhir setiap notebook yang relevan untuk mengexport figure ke folder `docs/skripsi-summary/figures/`:

```python
import matplotlib.pyplot as plt
import os

EXPORT_DIR = "docs/skripsi-summary/figures"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Contoh: simpan figure yang sedang aktif
fig.savefig(
    os.path.join(EXPORT_DIR, "G4_15_confusion_matrix_dualstream.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
print(f"Figure tersimpan di: {EXPORT_DIR}")
```

> **Catatan**: Karena seluruh figure dibuat dalam notebook dan skripsi PDF berisi versi embed, semua figure perlu di-regenerate dari notebook masing-masing lalu di-export menggunakan kode di atas dengan `dpi=300`.
