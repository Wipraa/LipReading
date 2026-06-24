# Catatan Teknis — Inkonsistensi dan Hal yang Perlu Diverifikasi

> **CATATAN**: File ini TIDAK diupload ke NotebookLM. Berisi catatan inkonsistensi teknis untuk keperluan debugging dan verifikasi saja. Prefix underscore menandai file ini sebagai internal.

---

## Inkonsistensi 1: `input_size: 64` vs Output Aktual 96×96

**Sumber**: `CLAUDE.md` / `config.json` vs hasil aktual preprocessing

- `config.json` mencantumkan `"input_size": 64` di section `training`
- Output aktual preprocessing (Notebook 03) adalah **96×96 piksel** (shape per file: `[30, 96, 96, 3]`)
- Seluruh eksperimen pelatihan menggunakan resolusi **96×96**, bukan 64×64
- Angka 64 di config kemungkinan adalah nilai lama yang belum diperbarui dan tidak digunakan secara aktif oleh kode training

**Dampak pada hasil**: Tidak ada — training berjalan dengan 96×96. Perlu diverifikasi apakah ada bagian kode yang membaca `input_size` dari config secara langsung.

**Untuk slide/presentasi**: Gunakan angka **96×96** — ini adalah ukuran aktual yang digunakan di semua eksperimen.

---

## Inkonsistensi 2: Test Loss Identik di Tabel 4.8 dan Tabel 4.12

- Tabel 4.8 (Dual LK P17 + M01, Skenario 3C): Test Loss = **0.8598**
- Tabel 4.12 (Dual P17+M02, Skenario 5B): Test Loss = **0.8598**

Angka test loss yang identik pada dua eksperimen berbeda (M01 vs M02) perlu diverifikasi: apakah ini kebetulan numerik atau kesalahan ketik dalam skripsi. Metrik utama (test accuracy 86.07% vs 86.63% dan F1 weighted 85.94% vs 86.60%) sudah berbeda secara konsisten, sehingga dampak pada analisis utama sangat minimal. Test Loss bukan metrik presentasi utama.

---

## Inkonsistensi 3: Ambiguitas M01 vs M02 di Figure G4.13/G4.15/G4.16

Di `05-figure-inventory.md`, beberapa deskripsi figure mencampur dua angka:
- G4.13: "best val acc=86.53% epoch 84 (M01) **atau** 87.22% epoch 83 (M02)"
- G4.15: "test acc 86.07% (M01) / 86.63% (M02)"
- G4.16: "overall 86.07%" (merujuk ke M01, bukan model terbaik final)

Saat mengexport figure dari notebook `07d_DualStream_M02_Gabungan.ipynb`, pastikan memilih figure dari **run M02** (hasil terbaik: test acc 86.63%) bukan run M01. Figure G4.16 pada skripsi kemungkinan menampilkan bar chart M01, bukan M02 — perlu dikonfirmasi dari notebook.

---

## Catatan: Perbedaan Nomor Tabel antara File Markdown dan Skripsi

Beberapa nomor tabel di file markdown mengikuti penomoran skripsi:
- "Tabel 4.3" di skripsi = Skenario 1 Farneback vs LK
- "Tabel 4.10" di skripsi = OFAT Arsitektur

Jika ada ketidakcocokan antara data di file markdown dengan data yang diingat dari skripsi, selalu gunakan data numerik di file markdown (yang sudah diverifikasi saat pembuatan) sebagai referensi utama.
