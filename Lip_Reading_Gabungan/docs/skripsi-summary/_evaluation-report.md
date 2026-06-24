# Laporan Evaluasi: Optimasi File Markdown sebagai Sumber NotebookLM

**Tanggal evaluasi**: 2026-05-08  
**Evaluator**: Claude Code (claude-sonnet-4-6)  
**Tujuan**: Menilai apakah 6 file di `docs/skripsi-summary/` sudah optimal untuk digunakan sebagai sumber fitur "Buat slide presentasi AI" di NotebookLM (BETA).

---

## Konteks: Cara Kerja NotebookLM RAG

NotebookLM menggunakan Retrieval-Augmented Generation (RAG) dengan karakteristik berikut yang relevan untuk evaluasi ini:

- **Chunking**: Teks dipotong berdasarkan heading dan paragraf. Tabel markdown dibaca sebagai teks polos dengan pipe (`|`) sebagai pemisah.
- **Buta kode**: Blok ` ``` ` (fenced code blocks) dibaca sebagai teks literal, bukan dirender. Diagram ASCII dalam code block menjadi noise.
- **Buta gambar**: Tidak dapat membaca file gambar. Referensi ke figure hanya berguna jika deskripsinya ada dalam teks.
- **Tidak ada konteks silang dokumen otomatis**: Setiap chunk dinilai secara independen. Kalimat "lihat file 01" tidak akan di-follow-up secara otomatis.
- **Self-containment kritis**: Chunk yang berisi angka kunci tanpa konteks (mis., "83.06%" tanpa menyebut "test accuracy baseline") sulit dimanfaatkan.

---

## Fase 1: Inventaris File

| File | Ukuran (perkiraan) | Jumlah Baris | H2 | H3 | Placeholder `[TBD]`/`[PERLU KONFIRMASI]` |
|------|-------------------|--------------|----|----|------------------------------------------|
| `01-arsitektur-model.md` | ~7.4 KB | 176 | 6 | 3 | Tidak ada |
| `02-pipeline-preprocessing.md` | ~5.7 KB | 143 | 5 | 3 | Tidak ada |
| `03-dataset.md` | ~3.8 KB | 112 | 6 | 2 | Tidak ada |
| `04-hasil-eksperimen.md` | ~12 KB | 303 | 8 | 8 | Tidak ada |
| `05-figure-inventory.md` | ~6.9 KB | 83 | 3 | 0 | Tidak ada |
| `06-poin-pembahasan.md` | ~12 KB | 166 | 3 | 14+ | Tidak ada |

**Pemeriksaan tambahan:**
- File `00-inkonsistensi.md`: **Tidak ada** — aman, tidak ada file konflik terpisah.
- Direktori `figures/`: **Kosong** — tidak ada satu pun file gambar yang sudah diexport.
- Seluruh 6 file bebas dari placeholder yang belum diisi.

---

## Fase 2: Evaluasi Per File

Skala penilaian: 1 (sangat buruk) — 5 (sangat baik).

---

### `01-arsitektur-model.md`

#### A. Kelengkapan Konten: 4/5
Mencakup semua 3 varian model, tabel layer lengkap dengan shape, OFAT arsitektur 6 variasi (M01–M06), dan tabel hyperparameter. Formula AttentionFusion ada di dalam ASCII diagram (code block). **Kurang**: formula matematis AttentionFusion (`g_rgb = σ(W_rgb · [f_rgb||f_flow])`) tidak ada dalam bentuk teks naratif di luar code block.

#### B. Self-Containment untuk RAG: 3/5
- Dua blok ASCII besar (```` ``` ````) mendeskripsikan arsitektur baseline dan dual-stream. RAG akan membaca ini sebagai teks literal penuh simbol `│`, `▼`, `┌`, `└` — tidak bermakna sebagai chunk tersendiri.
- "Sumber utama: BAB III (hal. 30–37)" di baris ke-3 mereferensikan konteks yang tidak dimiliki NotebookLM.
- Section 6 "Catatan Inkonsistensi" menyebutkan: `input_size: 64` (config.json) vs **96×96** (aktual). Informasi ini penting untuk pengembang tetapi **berisiko membingungkan** RAG — dua angka berbeda untuk hal yang sama.

#### C. Struktur untuk Slide: 3/5
Heading H2 cukup jelas. Namun dua arsitektur utama (informasi paling penting untuk slide) hanya tersedia dalam format code block yang tidak akan di-chunk dengan baik.

#### D. Visual Cue: 2/5
Diagram arsitektur disimpan sebagai ASCII art di code block. NotebookLM tidak akan menyarankan slide dengan diagram ini karena kontennya adalah teks noise.

#### E. Konsistensi Bahasa: 5/5
Formal Indonesian akademis, konsisten di seluruh file.

**Skor Total: 3.4/5**  
**Masalah Utama**: ASCII architecture diagram di code block adalah konten terpenting file ini, tetapi paling tidak berguna untuk RAG.

---

### `02-pipeline-preprocessing.md`

#### A. Kelengkapan Konten: 5/5
Mencakup seluruh 5 tahap pipeline, tabel dimensi data per tahap, tabel library + versi, parameter LK P17 lengkap, dan perbandingan Farneback vs LK. Semua angka kunci ada (300 corners, winSize 15×15, dll.).

#### B. Self-Containment untuk RAG: 3/5
- Pipeline utama (tahap 1–5) digambarkan dalam satu blok ASCII besar (±60 baris). Ini adalah masalah struktural utama: informasi terpenting file ini ada di code block.
- "Sumber utama: Section 3.2–3.3 (hal. 28–30)" tidak berguna untuk NotebookLM.
- Tabel parameter P17 dan tabel Farneback vs LK adalah konten terbaik — sudah self-contained.

#### C. Struktur untuk Slide: 3/5
H2 jelas. Tabel parameter dan tabel perbandingan metode sangat layak dijadikan slide. Namun narasi pipeline (informasi paling penting) terkunci di code block.

#### D. Visual Cue: 2/5
Flowchart pipeline dalam ASCII code block. Tidak ada deskripsi tekstual naratif per tahap di luar code block yang dapat di-chunk RAG secara bermakna.

#### E. Konsistensi Bahasa: 5/5
Konsisten.

**Skor Total: 3.6/5**  
**Masalah Utama**: Pipeline flowchart utama dalam code block. Tabel-tabel di bawahnya sudah baik.

---

### `03-dataset.md`

#### A. Kelengkapan Konten: 5/5
Mencakup 18 label lengkap dengan pronunciation dan jumlah suku kata, skema split (nomor pengulangan), distribusi numerik, tidak ada augmentasi, dan karakteristik dataset. Semua angka kunci ada (3.600 video, 20 subjek, 70/20/10%).

#### B. Self-Containment untuk RAG: 5/5
File terbersih dari semua 6 file. Setiap heading mengandung informasi lengkap. Tidak ada code block problematik. Tabel 18 label dengan pronunciation adalah konten yang sangat kaya untuk RAG.

#### C. Struktur untuk Slide: 5/5
H2 langsung mengikuti alur logis: sumber → label → jumlah sampel → split → augmentasi → karakteristik. Setiap section mandiri.

#### D. Visual Cue: 4/5
Tabel-tabel bersih. Catatan `> **Referensi**` di akhir section mereferensikan halaman skripsi yang tidak relevan untuk NotebookLM, tapi tidak mengganggu konten utama.

#### E. Konsistensi Bahasa: 5/5
Konsisten.

**Skor Total: 4.8/5**  
**File terbaik** dalam koleksi ini. Hampir tidak ada yang perlu diperbaiki.

---

### `04-hasil-eksperimen.md`

#### A. Kelengkapan Konten: 5/5
Mencakup semua 5 skenario eksperimen, tiga tabel per-kelas lengkap (18 baris masing-masing), rekapitulasi 8 eksperimen, dan analisis berdasarkan jumlah suku kata. Semua angka kunci ada dan akurat.

#### B. Self-Containment untuk RAG: 3/5
- File terpanjang (303 baris, ~12KB). RAG akan memotong file ini menjadi banyak chunk kecil.
- Tiga tabel per-kelas 18 baris kemungkinan akan dipotong di tengah oleh chunking, memisahkan header dari data.
- Angka-angka kunci (83.06%, 51.25%, 86.63%) tersebar di berbagai section — RAG mungkin perlu beberapa retrieval untuk menemukan semuanya.
- Tidak ada code block problematik, tidak ada ASCII art. Isi murni teks dan tabel.

#### C. Struktur untuk Slide: 3/5
H2 dan H3 sudah baik. Namun section "Skenario 3A Baseline" saja sudah sangat panjang (±60 baris termasuk tabel per-kelas). Untuk slide generation, RAG perlu mengambil banyak chunk untuk merekonstruksi gambar besar.

#### D. Visual Cue: 4/5
Tabel markdown dense tetapi berfungsi. Angka-angka terstruktur dengan baik untuk dikutip dalam slide.

#### E. Konsistensi Bahasa: 5/5
Konsisten.

**Skor Total: 4.0/5**  
**Masalah Utama**: Panjang file dan kepadatan tabel per-kelas. Bukan masalah fatal, tapi chunking kurang ideal.

---

### `05-figure-inventory.md`

#### A. Kelengkapan Konten: 4/5
Mencakup inventaris 22 figure (5 BAB III + 17 BAB IV) dengan lokasi notebook, kategori, deskripsi, dan prioritas sidang. Kode Python export disertakan.

#### B. Self-Containment untuk RAG: 2/5
- Kode Python export (±15 baris) adalah dead weight — RAG membacanya sebagai teks literal.
- **Semua 22 figure berstatus "Perlu diexport"** — tidak ada satu pun gambar yang tersedia. File ini mendeskripsikan figure yang belum ada.
- Kolom "Lokasi (Notebook)" mereferensikan path notebook yang tidak bermakna untuk NotebookLM.
- G4.13/G4.15/G4.16 menyebut "M01 atau M02" secara ambigu — dua angka berbeda dicampur dalam satu baris.

#### C. Struktur untuk Slide: 3/5
Tabel prioritas figure (⭐⭐⭐ / ⭐⭐ / ⭐) adalah konten berguna — RAG dapat menggunakannya untuk menyarankan slide mana yang perlu visual.

#### D. Visual Cue: 1/5
File ini seharusnya mendukung visual slide, tetapi tidak ada satu pun gambar yang tersedia. Semua referensi adalah deskripsi teks tentang gambar yang belum diexport.

#### E. Konsistensi Bahasa: 3/5
Campuran Indonesian dengan Python code block dan istilah teknis path notebook.

**Skor Total: 2.6/5**  
**File terlemah** dalam koleksi. Berguna sebagai checklist internal tetapi nilai RAG-nya sangat rendah selama figure belum diexport.

---

### `06-poin-pembahasan.md`

#### A. Kelengkapan Konten: 5/5
Mencakup 7 poin temuan utama (format Claim/Bukti/Implikasi), 7 pasang Q&A penguji dengan jawaban naratif lengkap, dan tabel 7 limitasi penelitian. Setiap poin memiliki angka spesifik sebagai bukti.

#### B. Self-Containment untuk RAG: 5/5
File paling self-contained. Setiap H3 (mis., "Poin 1 — Lucas-Kanade lebih unggul...") memiliki Claim + Bukti + Implikasi yang lengkap dalam satu section. Tidak ada code block. Tidak ada referensi ke halaman skripsi yang tidak bermakna. Angka kunci disebut eksplisit dalam konteks.

#### C. Struktur untuk Slide: 5/5
Format Claim/Bukti/Implikasi secara natural memetakan ke struktur slide: judul → data → kesimpulan. Setiap poin bisa menjadi satu slide yang berdiri sendiri. Q&A juga berguna untuk slide "Antisipasi Pertanyaan".

#### D. Visual Cue: 5/5
Tidak memerlukan gambar — angka kunci sudah ada dalam teks naratif. Tabel limitasi bersih dan informatif.

#### E. Konsistensi Bahasa: 5/5
Formal Indonesian akademis terbaik di antara semua file.

**Skor Total: 5.0/5**  
**File terbaik kedua** (bersama `03`). Sangat dioptimalkan untuk RAG.

---

## Ringkasan Skor Per File

| File | Konten | Self-Contain | Slide Struct | Visual | Bahasa | **Rata-rata** |
|------|--------|-------------|-------------|--------|--------|--------------|
| `01-arsitektur` | 4 | 3 | 3 | 2 | 5 | **3.4** |
| `02-preprocessing` | 5 | 3 | 3 | 2 | 5 | **3.6** |
| `03-dataset` | 5 | 5 | 5 | 4 | 5 | **4.8** |
| `04-hasil` | 5 | 3 | 3 | 4 | 5 | **4.0** |
| `05-figure-inv` | 4 | 2 | 3 | 1 | 3 | **2.6** |
| `06-pembahasan` | 5 | 5 | 5 | 5 | 5 | **5.0** |
| **Rata-rata koleksi** | **4.7** | **3.5** | **3.7** | **3.0** | **4.7** | **3.9** |

---

## Fase 3: Pemeriksaan Konsistensi Lintas File

### Angka Kunci — Konsistensi ✓/✗

| Metrik | File 01 | File 04 | File 06 | Status |
|--------|---------|---------|---------|--------|
| Baseline test acc | — | 83.06% | 83.06% | ✓ Konsisten |
| Single-stream test acc | — | 51.25% | 51.25% | ✓ Konsisten |
| Dual P17+M02 test acc | — | 86.63% | 86.63% | ✓ Konsisten |
| Dual P17+M02 F1 (w) | — | 86.60% | 86.60% | ✓ Konsisten |
| Dataset total | — | 3.600 | — | — |
| M02 LSTM hidden | 128,128 | — | — | ✓ (hanya di 01) |
| Farneback test acc | — | 57.73% | — | ✓ (hanya di 04) |
| LK test acc (Skenario 1) | — | 62.27% | — | ✓ (hanya di 04) |
| Epoch terbaik P17+M02 | — | — | 83 | — |
| Best val acc P17+M02 | — | 87.22% | 87.22% | ✓ Konsisten |
| Waktu/epoch dual-stream | — | ~116 detik | ~116 detik | ✓ Konsisten |

### Inkonsistensi yang Teridentifikasi

1. **`input_size: 64` vs `96×96`** (`01-arsitektur-model.md` Section 6):  
   File 01 secara eksplisit mencatat inkonsistensi ini (config.json vs aktual). File 02 dan 03 hanya menyebut 96×96 tanpa ambiguitas. Dari perspektif RAG: jika NotebookLM menemukan chunk dari Section 6 file 01, ia mungkin menghasilkan slide dengan angka 64 yang salah.  
   **Rekomendasi**: Hapus atau pindahkan Section 6 ke file catatan terpisah (di luar sumber NotebookLM).

2. **Tabel G4.13/G4.15/G4.16 di `05-figure-inventory.md` — Ambiguitas M01 vs M02**:  
   Contoh: "G4.15 Confusion Matrix Dual-Stream ... test acc 86.07% (M01) / 86.63% (M02)". Ini mencampur dua angka yang berbeda (86.07% dan 86.63%) dalam satu baris. RAG bisa memunculkan angka yang salah untuk slide confusion matrix.  
   **Rekomendasi**: Pisahkan baris untuk M01 dan M02, atau sebutkan hanya M02 (model terbaik final).

3. **Angka "Test Loss: 0.8598" identik di Tabel 4.8 dan Tabel 4.12** (disebutkan di Section 6 file 01):  
   Ini adalah inkonsistensi dalam skripsi asli yang sudah dicatat. Tidak ada di file lain. Dampak RAG minimal karena Test Loss bukan metrik utama yang disajikan.

4. **Referensi `> **Referensi**: hal. XX` di semua file**:  
   Semua 6 file menyertakan footer referensi ke halaman skripsi (mis., "Tabel 4.3 (hal. 43)"). Ini berguna untuk penulis tetapi menjadi dead weight bagi NotebookLM yang tidak memiliki akses ke PDF.

5. **`Sumber utama: BAB III (hal. XX–YY)` di setiap file**:  
   Baris pertama setiap file menyebut sumber dari skripsi. Tidak berbahaya tetapi tidak berguna untuk RAG.

---

## Fase 4: Simulasi Generasi 15 Slide

Berikut adalah simulasi bagaimana NotebookLM akan membangun 15 slide dari koleksi ini, beserta analisis gap:

| No | Judul Slide yang Diharapkan | Sumber Utama | Kualitas Konten yang Tersedia | Gap? |
|----|----------------------------|-------------|-------------------------------|------|
| 1 | Judul & Identitas Penelitian | — | **Tidak ada file yang mencakup latar belakang, judul resmi, atau motivasi penelitian** | ⚠️ GAP BESAR |
| 2 | Latar Belakang & Motivasi | — | Tidak ada narasi "mengapa SIBI lip reading penting" | ⚠️ GAP BESAR |
| 3 | Rumusan Masalah & Tujuan Penelitian | — | Tidak ada file yang mencantumkan rumusan masalah formal | ⚠️ GAP BESAR |
| 4 | Dataset SIBI: 18 Label | `03-dataset.md` | Sangat baik — tabel 18 label dengan suku kata | ✓ |
| 5 | Skema Split Data & Jumlah Sampel | `03-dataset.md` | Sangat baik — tabel split 70/20/10% | ✓ |
| 6 | Pipeline Preprocessing | `02-pipeline-preprocessing.md` | Lemah — pipeline ada di code block ASCII | ⚠️ PARSIAL |
| 7 | Parameter Optical Flow LK P17 | `02-pipeline-preprocessing.md` | Baik — tabel parameter tersedia | ✓ |
| 8 | Arsitektur Baseline Conv-LSTM | `01-arsitektur-model.md` | Lemah — arsitektur ada di code block ASCII | ⚠️ PARSIAL |
| 9 | Arsitektur Dual-Stream & AttentionFusion | `01-arsitektur-model.md` | Lemah — ada di code block; formula tidak dalam teks naratif | ⚠️ PARSIAL |
| 10 | Perbandingan Farneback vs LK | `04-hasil-eksperimen.md` | Baik — tabel tersedia di Section Skenario 1 | ✓ |
| 11 | Hasil Baseline vs Single-Stream vs Dual-Stream | `04-hasil-eksperimen.md` + `06-poin-pembahasan.md` | Sangat baik — angka ada di kedua file dengan konteks | ✓ |
| 12 | Analisis Per Kelas: Kelas Paling Sulit | `04-hasil-eksperimen.md` + `06-poin-pembahasan.md` | Baik — data per kelas ada, dengan penjelasan di file 06 | ✓ |
| 13 | Temuan Utama & Kontribusi | `06-poin-pembahasan.md` | Sangat baik — 7 poin dengan Claim/Bukti/Implikasi | ✓ |
| 14 | Limitasi Penelitian | `06-poin-pembahasan.md` | Sangat baik — tabel 7 limitasi lengkap | ✓ |
| 15 | Kesimpulan & Saran Penelitian Lanjutan | `06-poin-pembahasan.md` | Parsial — Poin 7 ada, tapi tidak ada section "Kesimpulan" formal | ⚠️ PARSIAL |

### Ringkasan Gap:

| Kategori | Jumlah Slide | Keterangan |
|----------|-------------|------------|
| ✓ Konten baik, siap digunakan | 7 slide | Slides 4,5,7,10,11,12,13,14 |
| ⚠️ Konten parsial/terdegradasi | 4 slide | Slides 6,8,9,15 (informasi ada tapi format tidak optimal) |
| ⚠️ GAP — konten tidak ada | 3 slide | Slides 1,2,3 (latar belakang, motivasi, rumusan masalah) |

**Gap terbesar**: Tidak ada file yang mencakup **Bab 1 (Pendahuluan)** — latar belakang, rumusan masalah, tujuan, dan manfaat penelitian. NotebookLM tidak akan bisa membuat slide pembuka yang bermakna.

---

## Fase 5: Rekomendasi Prioritas

### Prioritas 1 — Kritis (perlu diperbaiki sebelum NotebookLM dapat menghasilkan slide lengkap)

**R1. Buat file `00-pendahuluan.md`**  
File ini tidak ada sama sekali. Tanpanya, NotebookLM tidak bisa membuat slide pembuka (3 dari 15 slide kosong). Isi yang diperlukan:
- Latar belakang: prevalensi tunarungu Indonesia, kebutuhan komunikasi SIBI, gap teknologi lip reading SIBI
- Rumusan masalah formal (dari Section 1.2 skripsi)
- Tujuan penelitian (dari Section 1.3)
- Kontribusi penelitian (5 poin dari `06-poin-pembahasan.md` Q7 sudah ada, bisa dikopikan)

**R2. Tambahkan section "Kesimpulan" naratif di `06-poin-pembahasan.md`**  
Saat ini file 06 berisi poin-poin yang sangat baik tetapi tidak ada paragraf kesimpulan ringkas seperti yang biasa muncul di slide penutup. Tambahkan section:
```
## Kesimpulan
Penelitian ini membuktikan bahwa pendekatan dual-stream Conv-LSTM yang menggabungkan...
```

### Prioritas 2 — Penting (meningkatkan kualitas slide yang sudah bisa dibuat)

**R3. Ubah ASCII diagram di `01-arsitektur-model.md` menjadi teks naratif**  
Ganti code block arsitektur dengan deskripsi naratif atau bullet list. Contoh:
```
Arsitektur Baseline terdiri dari: (1) tiga blok Conv2D time-distributed...
```
ASCII diagram boleh dipertahankan sebagai referensi visual bagi manusia, tetapi tambahkan deskripsi teks di atasnya.

**R4. Ubah pipeline di `02-pipeline-preprocessing.md` menjadi teks naratif**  
Sama dengan R3. Tambahkan deskripsi tekstual per tahap di luar code block. Tabel dimensi yang sudah ada sudah baik — lengkapi dengan narasi singkat per tahap.

**R5. Hapus atau pisahkan Section 6 "Catatan Inkonsistensi" dari `01-arsitektur-model.md`**  
Pindahkan ke file `_catatan-teknis.md` (dengan underscore prefix agar tidak diupload ke NotebookLM). Section ini berguna untuk pengembang tetapi berisiko menghasilkan slide dengan angka 64 yang salah.

### Prioritas 3 — Opsional (peningkatan kecil)

**R6. Export figure prioritas ⭐⭐⭐ dari notebook**  
Gambar G4.7 (confusion matrix baseline), G4.15 (confusion matrix dual-stream), G4.17 (bar chart perbandingan), G3.4 dan G3.5 (arsitektur diagram) adalah aset paling penting untuk slide visual. Setelah diexport ke `figures/`, mereka bisa diupload bersama file markdown ke NotebookLM.

**R7. Hapus semua baris `> **Referensi**: hal. XX` dari semua file**  
Atau ganti dengan deskripsi yang self-contained. Contoh: ganti "> Referensi: Tabel 4.3 (hal. 43)" dengan "*(Data dari eksperimen perbandingan metode optical flow pada dataset Number 10 kelas)*".

**R8. Pisahkan `04-hasil-eksperimen.md` menjadi 2 file**  
File ini (303 baris) terlalu panjang untuk chunking yang optimal. Pertimbangkan memisahkan:
- `04a-hasil-skenario-1-2.md`: Farneback vs LK dan OFAT parameter
- `04b-hasil-skenario-3-4-5.md`: baseline, single-stream, dual-stream pada dataset gabungan

**R9. Perbarui `05-figure-inventory.md` setelah export**  
Setelah figure diexport (R6), ubah status "Perlu diexport" menjadi path file aktual. File ini punya nilai rendah selama semua figure belum ada.

---

## Kesimpulan Evaluasi

**Koleksi ini sudah layak digunakan di NotebookLM, tetapi tidak optimal.**

Kekuatan utama:
- `03-dataset.md` dan `06-poin-pembahasan.md` sudah mendekati sempurna untuk RAG.
- Semua angka kunci akurat dan konsisten antar file.
- Tidak ada placeholder atau data yang belum dikonfirmasi.

Kelemahan utama:
- **Gap terbesar**: Tidak ada Bab 1 (Pendahuluan) — 3 dari 15 slide yang diharapkan akan kosong atau dipaksakan dari konteks yang tidak tepat.
- **Format code block**: Dua informasi terpenting (arsitektur model dan pipeline preprocessing) disimpan dalam format yang tidak dapat di-chunk secara bermakna oleh RAG.
- **Figure kosong**: Seluruh direktori `figures/` masih kosong, membatasi kemampuan NotebookLM untuk menyarankan slide dengan visual.

**Jika hanya satu tindakan yang dilakukan**: Buat file `00-pendahuluan.md` (Prioritas R1). Ini akan mengisi gap paling besar dan memungkinkan NotebookLM menghasilkan slide pembuka yang koheren.

**Jika dua tindakan**: Tambahkan R1 + R3/R4 (ubah code block menjadi teks naratif di file 01 dan 02). Ini akan meningkatkan skor rata-rata koleksi dari 3.9 menjadi perkiraan 4.5+.

---

## Status Setelah Perbaikan (2026-05-08)

Semua rekomendasi prioritas 1 dan 2 telah dieksekusi. Berikut penilaian ulang per file:

### Perubahan yang Dilakukan

| Task | Status | Keterangan |
|------|--------|-----------|
| T1 — Buat `00-pendahuluan.md` | ✅ Selesai | File baru ~4.2KB, mencakup latar belakang, rumusan masalah, tujuan, manfaat, batasan, kontribusi |
| T2 — Narasi sebelum ASCII diagram `01` | ✅ Selesai | Ditambah narasi naratif untuk baseline dan dual-stream + formula AttentionFusion inline |
| T2 — Narasi step-by-step `02` | ✅ Selesai | Ditambah 5 paragraf per-tahap sebelum diagram alir ASCII |
| T3 — Pindah Section 6 ke `_catatan-teknis.md` | ✅ Selesai | Section inkonsistensi dihapus dari `01`, file `_catatan-teknis.md` dibuat |
| T4 — Kesimpulan + Saran di `06` | ✅ Selesai | Dua section baru ditambahkan di akhir file |
| T5 — Hapus semua footer `> **Referensi**: hal.XX` | ✅ Selesai | Dihapus dari semua 6 file (01–06); baris "Sumber utama:" juga dihapus |

### Skor Ulang Per File (Estimasi)

| File | Self-Contain | Slide Struct | **Rata-rata Lama** | **Rata-rata Baru** |
|------|-------------|-------------|-------------------|-------------------|
| `00-pendahuluan` | 5 | 5 | — (baru) | **5.0** |
| `01-arsitektur` | 4→4.5 | 3→4.5 | 3.4 | **4.2** |
| `02-preprocessing` | 3→4.5 | 3→4.5 | 3.6 | **4.4** |
| `03-dataset` | 5 | 5 | 4.8 | **4.9** |
| `04-hasil` | 3 | 3 | 4.0 | **4.1** |
| `05-figure-inv` | 2 | 3 | 2.6 | **2.7** |
| `06-pembahasan` | 5 | 5→5 | 5.0 | **5.0** |
| **Rata-rata koleksi** | | | **3.9** | **~4.5** |

### File yang Siap Diupload ke NotebookLM

Semua file **tanpa** prefix underscore (7 file total):

| File | Ukuran Estimasi | Kesiapan |
|------|----------------|---------|
| `00-pendahuluan.md` | ~4.2 KB | ✅ Siap |
| `01-arsitektur-model.md` | ~10 KB | ✅ Siap |
| `02-pipeline-preprocessing.md` | ~8 KB | ✅ Siap |
| `03-dataset.md` | ~3.5 KB | ✅ Siap |
| `04-hasil-eksperimen.md` | ~11 KB | ✅ Siap |
| `05-figure-inventory.md` | ~6.5 KB | ✅ Siap (berguna sebagai checklist; nilai RAG rendah hingga figure diexport) |
| `06-poin-pembahasan.md` | ~14 KB | ✅ Siap |

**Total: 7 file, ~57 KB** — semua siap diupload. File `_evaluation-report.md` dan `_catatan-teknis.md` (prefix underscore) **tidak** diupload ke NotebookLM.
