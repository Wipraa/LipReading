# 06 — Poin Pembahasan & Antisipasi Pertanyaan Penguji

---

## Temuan Utama (untuk Slide Pembahasan)

### Poin 1 — Lucas-Kanade lebih unggul dari Farneback untuk representasi optical flow bibir

**Claim**: Metode Lucas-Kanade menghasilkan generalisasi lebih baik dibanding Farneback pada task pembacaan gerak bibir.

**Bukti**: Test accuracy LK = 62.27% vs Farneback = 57.73% (selisih +4.54%) pada dataset Number 10 kelas (Tabel 4.3). Meskipun best val acc Farneback sedikit lebih tinggi (61.59% vs 60.00%), LK unggul pada data uji yang belum dilihat.

**Implikasi**: Representasi sparse-to-dense dari LK (melalui interpolasi scipy.griddata) lebih mampu melakukan generalisasi pada pergerakan lokal area bibir yang subtle. Untuk SIBI lip reading, gerakan bibir bersifat sparse dan lokal sehingga pendekatan sparse optical flow lebih sesuai.

---

### Poin 2 — Single-stream optical flow tidak dapat menggantikan RGB, justru jauh lebih buruk

**Claim**: Optical flow Lucas-Kanade jika digunakan sendiri (tanpa konteks visual RGB) tidak memberikan representasi yang cukup untuk klasifikasi gerak bibir SIBI.

**Bukti**: Test accuracy single-stream hanya 51.25% (F1=50.47%), jauh di bawah baseline 83.06% (selisih -31.81%). Training accuracy mencapai 97.66% sementara validation accuracy hanya 50.56% — gap besar mengindikasikan overfitting signifikan (Tabel 4.6). Peningkatan kapasitas arsitektur (M02, M04) tidak membantu: M02 hanya naik +2.23% menjadi 53.48%.

**Implikasi**: Bottleneck bukan pada kapasitas model melainkan pada kualitas informasi optical flow itu sendiri. Peta optical flow bibir memiliki kemiripan tinggi antar beberapa kelas (terutama huruf c, d, e yang secara visual memiliki bentuk mulut serupa), sehingga sulit dibedakan tanpa konteks visual RGB.

---

### Poin 3 — Dual-stream dengan AttentionFusion secara konsisten mengalahkan kedua single-stream

**Claim**: Kombinasi informasi visual RGB dan gerak optical flow melalui mekanisme AttentionFusion memberikan representasi yang lebih komprehensif dan menghasilkan akurasi tertinggi.

**Bukti**: Dual-stream P17+M02 mencapai test accuracy 86.63% (F1=86.60%), melampaui baseline (83.06%) sebesar +3.57% dan single-stream (51.25%) sebesar +35.38%. Model ini unggul merata di semua kelompok suku kata: 1 suku kata (89.17%), 2 suku kata (87.50%), 3 suku kata (86.28%) — lihat Tabel 4.15.

**Implikasi**: Mekanisme AttentionFusion yang belajar secara adaptif menentukan bobot kontribusi setiap stream (melalui gate sigmoid) lebih efektif dari penggabungan sederhana. Ketika kualitas optical flow kurang memadai, model secara otomatis meningkatkan bobot RGB, dan sebaliknya.

---

### Poin 4 — Kualitas parameter optical flow lebih kritis dari kapasitas arsitektur model

**Claim**: Optimasi parameter Lucas-Kanade (konfigurasi P17) memberikan dampak lebih besar pada performa akhir dibanding peningkatan kapasitas arsitektur model.

**Bukti**: Penggantian parameter LK Default → P17 meningkatkan test accuracy +1.90% (84.17% → 86.07%), sedangkan penggantian arsitektur M01 → M02 hanya +0.56% (86.07% → 86.63%). Lihat Tabel 4.13 dan Section 4.11.5.

**Implikasi**: Dalam pipeline deep learning berbasis optical flow, kualitas representasi input lebih determinan dari kapasitas model. Investasi lebih besar dalam optimasi preprocessing dan ekstraksi fitur (bukan hanya model architecture search) terbukti lebih efektif.

---

### Poin 5 — Kelas 'saya' dan 'keliling' secara konsisten rendah di semua model

**Claim**: Beberapa kelas memiliki performa rendah secara konsisten lintas semua model, yang disebabkan oleh keterbatasan inheren dalam karakteristik data, bukan keterbatasan model.

**Bukti**:
- Kelas 'saya': F1 baseline=0.585, single-stream=0.270, dual-stream=0.762. Meskipun dual-stream meningkat signifikan, kelas ini tetap yang terendah.
- Kelas 'keliling': F1 baseline=0.667, single-stream=0.696, dual-stream=0.737.
- Kelas 'c' dan 'd': F1 pada single-stream sangat rendah (0.071, 0.128) karena kemiripan bentuk mulut.

**Implikasi**: Kemiripan pola gerak bibir antar kelas tertentu ('saya'↔'dia', 'c'↔'d'↔'e') dan variasi tinggi antar 20 subjek (10 guru, 10 siswa dengan latar belakang berbeda) menjadi batas atas alami sistem. Diperlukan teknik tambahan seperti augmentasi data atau subject-independent training untuk mengatasi ini.

---

### Poin 6 — Label 1 suku kata paling sulit untuk model single-stream

**Claim**: Kompleksitas temporal (jumlah suku kata) mempengaruhi kemampuan model optical flow secara berbeda-beda antara single-stream dan dual-stream.

**Bukti**: Recall LK single-stream untuk label 1 suku kata (a,b,c,d,e,f) hanya 41.67%, sementara label 3 suku kata (8,9,10,keliling,kelompok,sekarang) mencapai 64.60%. Dual-stream membalikkan tren ini: label 1 suku kata justru lebih tinggi (89.17% vs 86.28%) — Tabel 4.15.

**Implikasi**: Label 1 suku kata menghasilkan perbedaan antar frame yang minimal dalam optical flow, sehingga single-stream sulit menangkap informasi yang cukup. Dual-stream mengatasi ini karena stream RGB tetap memberikan konteks visual yang kaya meski gerakan singkat.

---

### Poin 7 — Konfigurasi akhir terbaik: Dual-Stream P17 + M02 mencapai 86.63%

**Claim**: Konfigurasi Dual-Stream LK P17 dengan arsitektur M02 merupakan konfigurasi terbaik dari seluruh 8 eksperimen yang dilakukan.

**Bukti**: Test accuracy 86.63%, F1 weighted 86.60%, best val accuracy 87.22% pada epoch 83 (Tabel 4.12). Total peningkatan dari baseline: +3.57%.

**Implikasi**: Penelitian ini membuktikan bahwa pendekatan dual-stream yang mempertahankan informasi RGB sekaligus menambahkan informasi gerak optical flow — bukan menggantikan RGB dengan optical flow — adalah strategi yang tepat untuk SIBI lip reading.

---

## Limitasi yang Teridentifikasi

| Limitasi | Deskripsi | Dampak |
|----------|-----------|--------|
| Subject-dependent only | Semua data uji berasal dari 20 subjek yang sama dengan data latih | Kemampuan generalisasi ke subjek baru tidak terukur |
| Tidak ada cross-subject validation | Tidak ada pengujian pada subjek di luar dataset | Akurasi nyata di kondisi lapangan bisa lebih rendah |
| Dataset terbatas (18 kelas) | Kosakata SIBI jauh lebih luas | Sistem belum representatif untuk penggunaan praktis |
| Single-stream optical flow sangat rendah | Test acc hanya 51.25% | Menunjukkan optical flow saja tidak cukup untuk task ini |
| Waktu inferensi dual-stream 2× lebih lambat | ~116 detik/epoch vs ~59 detik baseline | Belum cocok untuk aplikasi real-time |
| Tidak menggunakan augmentasi data | Semua variasi hanya dari rekaman alami | Potensi overfitting pada variasi pengucapan tertentu |
| backbone Conv-LSTM berat | Tidak ringan untuk perangkat mobile | Perlu backbone ringan seperti MobileNet untuk deployment |

---

## Pertanyaan Kritis Penguji & Poin Jawaban

### Q1: "Mengapa single-stream optical flow hasilnya sangat buruk (51.25%) padahal optical flow seharusnya membawa informasi gerak yang penting?"

**Jawaban**:
Ada tiga penjelasan utama (Section 4.11.2):
1. **Mismatch arsitektur**: Backbone CNN yang dirancang untuk fitur visual RGB kurang optimal untuk peta optical flow yang memiliki distribusi nilai sangat berbeda (z-score bukan [0,1])
2. **Kemiripan kelas dalam optical flow**: Pola optical flow area bibir beberapa kelas (c, d, e) sangat mirip secara visual karena bentuk mulut serupa, sehingga sulit dibedakan tanpa konteks tekstur RGB
3. **Overfitting**: Training accuracy 97.66% vs validation accuracy 50.56% menunjukkan model menghafal pola training tanpa generalisasi. Meningkatkan kapasitas model (M02, M04) tidak menyelesaikan masalah ini, mengkonfirmasi bahwa bottleneck ada pada informasi input, bukan kapasitas model

---

### Q2: "Mengapa tidak menggunakan Transformer atau ViT yang lebih modern daripada Conv-LSTM?"

**Jawaban**:
- Penelitian ini secara eksplisit membandingkan Conv-LSTM (tanpa optical flow) vs Optical Flow Conv-LSTM sebagai rumusan masalah utama (Section 1.2)
- Transformer memerlukan data yang jauh lebih banyak untuk konvergensi yang baik; dengan hanya 2.520 data training, Conv-LSTM lebih stabil
- Referensi [Aripin & Setiawan, 2024] yang menggunakan dataset bahasa Indonesia serupa menunjukkan Conv-LSTM/LRCN sudah mencapai 90%+ pada kondisi ideal
- Transformer dapat menjadi arah pengembangan ke depan (saran untuk penelitian lanjutan di Section 5.2)
- Fokus penelitian ini adalah kontribusi optical flow + dual-stream, bukan arsitektur transformer per se

---

### Q3: "Akurasi 86.63% — apakah ini cukup baik untuk digunakan secara praktis?"

**Jawaban**:
- Untuk sistem subject-dependent dengan 18 kelas pada dataset SIBI yang pertama kali dibuat penelitian ini, 86.63% adalah hasil yang baik
- Sebagai perbandingan: penelitian lip reading bahasa Indonesia oleh Aripin & Setiawan (2024) menggunakan LRCN mencapai 95.42% namun pada dataset berbeda (8 bentuk bibir, bukan 18 kelas kata/angka/alfabet)
- Untuk penggunaan praktis bagi penyandang tunarungu, diperlukan akurasi di atas 90%+ dan perlu pengujian subject-independent
- Penelitian ini merupakan baseline pertama untuk dataset SIBI gabungan 18 kelas, yang menjadi fondasi penelitian selanjutnya

---

### Q4: "Kenapa tidak diuji dengan skema subject-independent? Bukan itu yang lebih realistis?"

**Jawaban**:
- Penelitian ini menggunakan subject-dependent sebagai langkah awal yang bertujuan memaksimalkan potensi model sebelum meningkatkan kesulitan evaluasi
- Dengan hanya 20 subjek, pembagian subject-independent (misalnya leave-one-out) hanya menghasilkan 19/20 subjek training → sangat sedikit untuk Conv-LSTM yang membutuhkan variasi cukup
- Subject-independent adalah arah saran penelitian lanjutan yang disebutkan di Section 5.2
- [Catatan: jika penguji meminta lebih detail, akui ini sebagai keterbatasan yang disadari dan rencanakan sebagai penelitian lanjutan]

---

### Q5: "Mengapa jumlah suku kata dipilih sebagai kriteria seleksi label?"

**Jawaban** (Section 3.1.3):
- Jumlah suku kata merepresentasikan perbedaan durasi dan kompleksitas gerak bibir secara kuantitatif
- Memastikan dataset tidak bias terhadap satu pola durasi tertentu: label singkat (1 suku kata: alfabet), menengah (2 suku kata: angka kecil + kata pendek), dan panjang (3 suku kata: angka besar + kata kompleks)
- Distribusi seimbang: 6 label × 3 kelompok = 18 label dengan representasi proporsional

---

### Q6: "Mengapa tidak melakukan augmentasi data? Apakah ini tidak membatasi performa?"

**Jawaban**:
- Keputusan tidak melakukan augmentasi bertujuan menjaga kondisi eksperimen yang bersih (controlled experiment) sehingga perbandingan antar model adil
- Variasi alami sudah tersedia dari 20 subjek berbeda dengan pengulangan 10 kali
- Augmentasi pada data video bibir (flip horizontal, time warping) dapat mengintroduksi artefak yang tidak realistis
- Augmentasi dapat menjadi arah peningkatan di penelitian lanjutan

---

### Q7: "Apa kontribusi spesifik penelitian ini dibandingkan penelitian lip reading bahasa Indonesia sebelumnya?"

**Jawaban**:
1. **Dataset baru**: Dataset SIBI gabungan 18 label (alphabet, angka, kata) dari 20 subjek — belum pernah ada sebelumnya untuk SIBI lip reading
2. **Perbandingan sistematis**: Membandingkan tiga varian model (baseline, single-stream, dual-stream) dengan metodologi OFAT yang terkontrol
3. **AttentionFusion**: Implementasi mekanisme attention fusion adaptif untuk menggabungkan RGB dan optical flow pada konteks SIBI
4. **Temuan counter-intuitive**: Membuktikan bahwa optical flow tidak dapat menggantikan RGB secara langsung (single-stream lebih buruk dari baseline), hanya efektif sebagai stream tambahan
5. **Analisis berdasarkan suku kata**: Analisis performa model berdasarkan kompleksitas temporal (jumlah suku kata) memberikan insight linguistik yang belum ada di penelitian sebelumnya

---

## Kesimpulan Penelitian

Penelitian ini berhasil mengembangkan sistem pembacaan gerak bibir untuk video SIBI menggunakan pendekatan dual-stream Optical Flow Conv-LSTM yang mencapai test accuracy 86,63% dengan F1-score 86,60%, melampaui baseline Conv-LSTM RGB sebesar 3,57% (dari 83,06%). Temuan paling signifikan adalah bahwa optical flow Lucas-Kanade tidak dapat menggantikan representasi RGB secara langsung: model single-stream optical flow hanya mencapai 51,25% akibat keterbatasan inheren representasi gerak bibir tanpa konteks visual, dan peningkatan kapasitas arsitektur tidak mampu mengatasi keterbatasan ini. Pendekatan dual-stream dengan AttentionFusion yang mempertahankan informasi RGB sekaligus menambahkan informasi gerak terbukti menjadi strategi yang efektif. Selain itu, optimasi parameter Lucas-Kanade (konfigurasi P17) memberikan dampak peningkatan lebih besar (+1,90%) dibanding optimasi arsitektur model (+0,56%), menunjukkan bahwa kualitas representasi input lebih determinan dari kapasitas model. Keterbatasan utama penelitian ini adalah cakupan subject-dependent dan 18 kelas yang belum merepresentasikan seluruh kosakata SIBI.

---

## Saran Penelitian Lanjutan

1. **Pengujian subject-independent**: Melakukan evaluasi pada subjek baru di luar dataset pelatihan agar kemampuan generalisasi model dapat diukur secara lebih objektif. Dengan hanya 20 subjek, pembagian subject-independent (misalnya leave-one-out) menghasilkan data training yang sangat sedikit untuk Conv-LSTM, sehingga perluasan dataset perlu dilakukan terlebih dahulu.

2. **Perluasan dataset**: Memperluas dataset dari segi jumlah label (mencakup lebih banyak kosakata SIBI) maupun jumlah subjek, agar model dapat belajar dari variasi pengguna yang lebih beragam dan mengurangi risiko overfitting pada pola pengucapan tertentu.

3. **Backbone ringan untuk deployment mobile**: Mengganti backbone Conv-LSTM dengan arsitektur yang lebih ringan seperti MobileNet agar sistem dapat berjalan secara real-time pada perangkat mobile, sehingga applicable untuk penggunaan sehari-hari penyandang tunarungu di Indonesia.

4. **Augmentasi data video bibir**: Menerapkan teknik augmentasi yang sesuai untuk data video bibir (variasi kecepatan temporal, penambahan noise realistis, atau pergeseran posisi frame) untuk meningkatkan variasi data latih tanpa mengintroduksi artefak yang tidak realistis.

5. **Eksplorasi arsitektur modern**: Mengeksplorasi penggunaan Transformer atau Vision Transformer (ViT) dengan dataset yang lebih besar, mengingat arsitektur tersebut berpotensi lebih baik dalam menangkap dependensi jangka panjang pada data video bibir dengan jumlah data yang cukup.
