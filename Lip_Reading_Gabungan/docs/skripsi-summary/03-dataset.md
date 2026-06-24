# 03 — Dataset

---

## 1. Sumber dan Pengumpulan Data

- **Jenis data**: Data primer (rekaman langsung)
- **Subjek**: 20 orang — 10 guru SLB + 10 siswa penyandang tunarungu
- **Peralatan rekaman**:
  - Kamera: Logitech C920 Webcam Pro Full HD 1018p
  - Laptop: IdeaPad 14ALC7 (AMD RYZEN 3 5000, AMD RADEON, Samsung SSD)
- **Spesifikasi rekaman**: Resolusi 1920×1080, framerate 30 fps
- **Preprocessing awal video**: mengubah resolusi, menyamakan framerate, menghilangkan suara

---

## 2. Label dan Klasifikasi

### 18 Label yang Digunakan

| Kategori | Label | Pelafalan | Suku Kata | Jumlah Suku Kata |
|----------|-------|-----------|-----------|-----------------|
| Alphabet | a | /a/ | a | 1 |
| Alphabet | b | /be/ | be | 1 |
| Alphabet | c | /ce/ | ce | 1 |
| Alphabet | d | /de/ | de | 1 |
| Alphabet | e | /e/ | e | 1 |
| Alphabet | f | /ef/ | ef | 1 |
| Angka | 1 | satu | sa-tu | 2 |
| Angka | 2 | dua | du-a | 2 |
| Angka | 3 | tiga | ti-ga | 2 |
| Kata | buku | buku | bu-ku | 2 |
| Kata | dia | dia | di-a | 2 |
| Kata | saya | saya | sa-ya | 2 |
| Angka | 8 | delapan | de-la-pan | 3 |
| Angka | 9 | sembilan | sem-bi-lan | 3 |
| Angka | 10 | sepuluh | se-pu-luh | 3 |
| Kata | keliling | keliling | ke-li-ling | 3 |
| Kata | kelompok | kelompok | ke-lom-pok | 3 |
| Kata | sekarang | sekarang | se-ka-rang | 3 |

### Distribusi Berdasarkan Jumlah Suku Kata

| Kelompok | Label | Jumlah Label |
|----------|-------|-------------|
| 1 suku kata | a, b, c, d, e, f | 6 |
| 2 suku kata | 1, 2, 3, buku, dia, saya | 6 |
| 3 suku kata | 8, 9, 10, keliling, kelompok, sekarang | 6 |

Pemilihan 18 label ini dirancang **seimbang berdasarkan jumlah suku kata** untuk memastikan representasi perbedaan durasi dan kompleksitas gerak bibir yang proporsional.

---

## 3. Jumlah Sampel

- Setiap subjek merekam setiap label **10 kali pengulangan**
- Total: 20 subjek × 18 label × 10 ulangan = **3.600 video**
- Dataset ini merupakan dataset **subject-dependent** (data uji berasal dari subjek yang sama dengan data latih, bukan subjek baru)

> **Catatan**: Penelitian ini **tidak menggunakan skema subject-independent**. Seluruh eksperimen dilakukan pada satu dataset gabungan dengan pembagian berdasarkan nomor pengulangan.

---

## 4. Skema Pembagian Data (Split)

### Strategi Split

Data dibagi berdasarkan **nomor pengulangan**, bukan secara acak:
- **Train**: pengulangan ke-1 hingga ke-7 (70%)
- **Val**: pengulangan ke-8 dan ke-9 (20%)
- **Test**: pengulangan ke-10 (10%)

Strategi ini memastikan pembagian data merata tanpa kehilangan data dan tidak ada kebocoran antar split.

### Distribusi Numerik (Tabel 4.1)

| Split | Jumlah Video | Jumlah Label | Video per Label | Persentase |
|-------|-------------|-------------|----------------|-----------|
| Train | 2.520 | 18 | 140 | 70% |
| Val | 720 | 18 | 40 | 20% |
| Test | 360 | 18 | 20 | 10% |
| **Total** | **3.600** | **18** | **200** | **100%** |

---

## 5. Augmentasi Data

Penelitian ini **tidak menggunakan augmentasi data**. Variasi alami dalam data berasal dari:
- Perbedaan antar subjek (10 guru + 10 siswa dengan latar belakang beragam)
- 10 pengulangan per subjek per label dengan variasi pengucapan alami

---

## 6. Karakteristik Dataset

| Aspek | Keterangan |
|-------|------------|
| Bahasa | Sistem Isyarat Bahasa Indonesia (SIBI) |
| Format awal | Video MP4, 1920×1080, 30 fps |
| Format setelah preprocessing | NumPy array `.npy`, shape `[30, 96, 96, 3]` |
| Total penyimpanan mentah | ~1.8 GB |
| Total setelah preprocessing | ~2.8 GB |
| Keseimbangan kelas | Seimbang sempurna (200 sampel per kelas) |
| Skenario evaluasi | Subject-dependent |
| Data bersifat | Rekaman (tidak real-time) |
