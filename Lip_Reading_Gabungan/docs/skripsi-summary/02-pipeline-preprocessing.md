# 02 — Pipeline Preprocessing

---

## 1. Alur Lengkap: Video Mentah → Tensor Input Model

Pipeline preprocessing mengubah video mentah SIBI menjadi tensor input model melalui lima tahap berurutan yang diimplementasikan menggunakan Python, MediaPipe, dan OpenCV.

**Tahap 1 — Frame Extraction**: Setiap video MP4 beresolusi 1920×1080 (30 fps, direkam menggunakan Logitech C920) diekstraksi frame demi frame menggunakan OpenCV. Seluruh frame diekstraksi tanpa filtrasi awal; seleksi dilakukan pada tahap berikutnya berdasarkan skor gerak.

**Tahap 2 — Lip Detection**: Setiap frame di-resize ke 224×224 piksel sebelum diproses oleh MediaPipe FaceLandmarker (Tasks API versi 0.10, file model 3.76 GB). Detektor mengenali 478 titik landmark wajah, dari mana dipilih 40 titik bibir (20 OUTER_LIP + 20 INNER_LIP). Bounding box area bibir dihitung dari koordinat minimum dan maksimum titik-titik tersebut, diperbesar ±30% pada semua sisi untuk mencegah area bibir terpotong.

**Tahap 3 — Frame Selection (Motion Score)**: Untuk setiap frame dibuat thumbnail grayscale berukuran 32×32 piksel. Motion score dihitung sebagai selisih absolut piksel antara frame berurutan. Dari seluruh frame dalam satu video, dipilih 30 frame dengan skor gerak tertinggi, kemudian diurutkan kembali secara temporal untuk mempertahankan urutan kronologis gerakan bibir.

**Tahap 4 — Crop & Resize**: Setiap frame yang terpilih di-crop berdasarkan bounding box bibir yang dihitung pada Tahap 2, kemudian di-resize ke dimensi tetap 96×96 piksel dalam format RGB. Hasilnya disimpan sebagai array NumPy float32 dengan shape [30, 96, 96, 3] per video.

**Tahap 5 — Optical Flow Lucas-Kanade (P17)**: Array RGB [30, 96, 96, 3] digunakan sebagai input untuk menghitung optical flow antar-frame berurutan. Titik fitur dideteksi menggunakan `cv2.goodFeaturesToTrack` (maxCorners=300, qualityLevel=0.005, minDistance=5, blockSize=6) dan dilacak menggunakan `cv2.calcOpticalFlowPyrLK` (winSize=15×15, maxLevel=3). Karena LK menghasilkan flow sparse (hanya pada titik fitur), hasil dikonversi ke representasi dense melalui interpolasi linier `scipy.griddata`. Output tiga kanal (u=horizontal, v=vertikal, magnitude=√u²+v²) dinormalisasi menggunakan statistik global dari data training (clip p2–p98 kemudian z-score). Dari 30 frame RGB dihasilkan 29 pasang frame, menghasilkan array [29, 96, 96, 3] per video.

### Diagram Alir

```
Video Mentah (.mp4)
  Resolusi: 1920×1080, 30 fps
  Kamera: Logitech C920 Webcam Pro Full HD 1018p
  Durasi: bervariasi (setiap subjek mengulang 10×)
        │
        ▼ TAHAP 1 — Frame Extraction
  Ekstraksi setiap frame dari video secara berurutan
  (semua frame diekstraksi, kemudian diseleksi di Tahap 3)
        │
        ▼ TAHAP 2 — Lip Detection
  Library : MediaPipe FaceLandmarker (Tasks API, versi 0.10)
  - Setiap frame di-resize ke 224×224 sebelum deteksi
  - Detektor mengenali 478 titik landmark wajah
  - Dipilih 40 titik: 20 OUTER_LIP + 20 INNER_LIP
  - Bounding box bibir = min/max koordinat landmark ±30% margin
        │
        ▼ TAHAP 3 — Frame Selection (Motion Score)
  - Dibuat thumbnail grayscale 32×32 piksel per frame
  - Motion score = selisih piksel antar frame berurutan
  - Dipilih 30 frame dengan skor gerak tertinggi
  - Frame diurutkan kembali secara temporal (preservasi urutan)
        │
        ▼ TAHAP 4 — Crop & Resize
  - Setiap frame di-crop berdasarkan bounding box bibir
  - Di-resize ke ukuran 96×96 piksel (RGB)
        │
        ▼ OUTPUT PREPROCESSING
  numpy array: [30, 96, 96, 3]   ← shape per video
  Format: .npy, dtype float32
  Struktur folder: preprocessed/dependent/raw_lips/{train,val,test}/{kelas}/
        │
        ▼ TAHAP 5 — Optical Flow Lucas-Kanade (P17)
  Input: [30, 96, 96, 3] (RGB array)
  - Titik fitur: cv2.goodFeaturesToTrack
    (maxCorners=300, qualityLevel=0.005, minDistance=5, blockSize=6)
  - Tracking: cv2.calcOpticalFlowPyrLK
    (winSize=15×15, maxLevel=3)
  - Sparse → Dense: scipy.griddata interpolasi linier
  - 3 channel output: u (horizontal), v (vertikal), magnitude (√u²+v²)
  - Normalisasi global: clip p2–p98 → z-score (mean & std dari data training)
        │
        ▼ OUTPUT OPTICAL FLOW
  numpy array: [29, 96, 96, 3]   ← 30 frame RGB → 29 pasang frame
  Format: .npy, dtype float32
  Struktur folder: preprocessed/dependent/optical_flows/{train,val,test}/{kelas}/
        │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  INPUT MODEL BASELINE      INPUT MODEL DUAL-STREAM
  RGB: [B, 3, 30, 96, 96]   RGB: [B, 3, 30, 96, 96]
                             Flow:[B, 3, 29, 96, 96]
```

---

## 2. Dimensi Data di Setiap Tahap

| Tahap | Representasi Data | Shape |
|-------|-------------------|-------|
| Video mentah | MP4, 1920×1080, 30fps | — |
| Setelah frame extraction | Frame RGB per video | variabel × 1920×1080×3 |
| Setelah lip detection | Frame dengan bounding box | variabel × variabel×variabel×3 |
| Setelah frame selection | 30 frame terpilih | `[30, H_raw, W_raw, 3]` |
| Setelah crop & resize | Array numpy | `[30, 96, 96, 3]` |
| Setelah optical flow | Array numpy (normalized) | `[29, 96, 96, 3]` |
| Input baseline | Tensor PyTorch | `[B, 3, 30, 96, 96]` |
| Input dual-stream (flow) | Tensor PyTorch | `[B, 3, 29, 96, 96]` |

---

## 3. Library dan Versi yang Digunakan

| Library | Fungsi | Versi / Keterangan |
|---------|--------|-------------------|
| MediaPipe | Face landmark detection | Tasks API versi 0.10, model `face_landmarker.task` (3.76 GB) |
| OpenCV (`cv2`) | Frame extraction, optical flow | — |
| `scipy.griddata` | Interpolasi sparse-to-dense optical flow | — |
| NumPy | Array processing, .npy I/O | — |
| PyTorch | Tensor, training | 2.11.0+cu128 |
| Python | Runtime | 3.12.3 |

---

## 4. Parameter Kunci

### Preprocessing (Lip ROI)

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Landmark yang dipakai | 40 titik | 20 OUTER_LIP + 20 INNER_LIP |
| Resize sebelum deteksi | 224×224 piksel | Menyeragamkan skala deteksi |
| Margin bounding box | +30% ke semua sisi | Agar bibir tidak terpotong |
| Thumbnail motion score | 32×32 piksel grayscale | Efisiensi komputasi |
| Jumlah frame terpilih | 30 frame | Berdasarkan motion score tertinggi |
| Output crop size | 96×96 piksel | RGB |

### Optical Flow Lucas-Kanade (P17 — Konfigurasi Terpilih)

| Parameter | Nilai | Fungsi |
|-----------|-------|--------|
| `maxCorners` | 300 | Jumlah maksimum titik fitur yang dideteksi |
| `qualityLevel` | 0.005 | Ambang batas kualitas sudut minimum |
| `minDistance` | 5 piksel | Jarak minimum antar titik fitur |
| `blockSize` | 6 | Ukuran blok untuk komputasi turunan |
| `winSize` | 15×15 | Ukuran jendela pencarian aliran |
| `maxLevel` | 3 | Kedalaman piramida gambar |
| Interpolasi | Linier (`scipy.griddata`) | Sparse → dense flow map |
| Normalisasi | Global z-score (clip p2–p98) | Menggunakan statistik data training |

### Perbandingan Farneback vs Lucas-Kanade (Skenario 1, Dataset Number)

| Metode | Best Val Acc | Test Acc | F1 Macro | Waktu/Epoch |
|--------|-------------|----------|----------|-------------|
| Farneback | 61.59% | 57.73% | 57.41% | ~71 detik |
| **Lucas-Kanade** | 60.00% | **62.27%** | **61.15%** | ~77 detik |

Lucas-Kanade dipilih karena test accuracy lebih tinggi (+4.54%), meskipun waktu per epoch sedikit lebih lama.

---

## 5. Hasil Preprocessing

- **Tingkat keberhasilan**: 100% — seluruh 3.600 video berhasil menghasilkan file `.npy` tanpa kegagalan.
- **Distribusi data** (Tabel 4.1):

| Split | Jumlah Video | Jumlah Label | Video per Label |
|-------|-------------|-------------|----------------|
| Train | 2.520 | 18 | 140 |
| Val | 720 | 18 | 40 |
| Test | 360 | 18 | 20 |
| **Total** | **3.600** | **18** | **200** |

