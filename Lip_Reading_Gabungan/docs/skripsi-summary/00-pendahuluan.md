# Pendahuluan Penelitian SIBI Lip Reading

---

## Latar Belakang

Membaca gerak bibir (*lip reading*) adalah keterampilan memahami kata, kalimat, atau ucapan seseorang hanya melalui pengamatan gerakan bibir tanpa mendengar suara yang dihasilkan. Teknologi pembacaan gerak bibir secara komputasional bekerja dengan menganalisis perubahan bentuk bibir dari sekuens video untuk mengenali informasi yang sedang disampaikan. Meski tampak sederhana, bahkan manusia yang terlatih dalam pembacaan gerak bibir hanya mampu mencapai akurasi 17±12% untuk 30 kata tunggal dan 21±11% untuk kata majemuk — angka yang mencerminkan betapa sulitnya tugas ini secara kognitif maupun komputasional.

Penyandang tunarungu adalah individu yang mengalami gangguan pendengaran sebagian atau total yang mengakibatkan keterbatasan dalam menerima informasi secara auditori. Di Indonesia, komunikasi penyandang tunarungu difasilitasi oleh Sistem Isyarat Bahasa Indonesia (SIBI) — sistem bahasa isyarat resmi yang dirilis Departemen Kebudayaan dan Pendidikan Indonesia pada tahun 1994. SIBI menggunakan kombinasi gerakan tangan, ekspresi wajah, dan gerak bibir sebagai media komunikasi terpadu. Karena gerak bibir merupakan komponen inherent dalam komunikasi SIBI, kemampuan membaca gerak bibir secara otomatis menjadi faktor kritis untuk membangun sistem pengenalan SIBI yang dapat digunakan secara praktis.

Penelitian pembacaan gerak bibir secara komputasional telah dilakukan untuk berbagai bahasa, namun hasilnya tidak dapat langsung diterapkan pada SIBI. Penelitian untuk bahasa Turki menggunakan CNN dan Bi-LSTM mencapai 84,5%–88,55%; bahasa Mandarin dengan CNN 3D dan resBi-LSTM hanya 64,35%; bahasa Indonesia oleh Aripin & Setiawan (2024) menggunakan LRCN dengan 8 bentuk bibir mencapai 95,42%, tetapi tidak menggunakan kosakata SIBI yang sesungguhnya dan tidak mengintegrasikan informasi gerak temporal (*optical flow*). Sementara penelitian Kurniawan & Kaswidjanti (2024) mengenali isyarat SIBI namun tidak menggunakan pembacaan gerak bibir sebagai fitur. Penelitian yang secara khusus menangani pembacaan gerak bibir untuk video SIBI dengan cakupan kosakata gabungan (alfabet, angka, dan kata dasar) serta memanfaatkan informasi gerak temporal belum pernah dilakukan sebelumnya.

Gap inilah yang menjadi motivasi penelitian ini: mengembangkan sistem pembacaan gerak bibir khusus untuk video SIBI menggunakan pendekatan *Optical Flow* Conv-LSTM yang menggabungkan informasi visual RGB dengan informasi gerak Lucas-Kanade, serta membandingkannya secara sistematis dengan model Conv-LSTM standar tanpa *optical flow* sebagai titik acuan (*baseline*).

---

## Rumusan Masalah

Bagaimana perbandingan hasil akurasi yang didapat dengan menggunakan model Conv-LSTM dan *Optical Flow* Conv-LSTM untuk mendeteksi gerakan bibir pada video SIBI?

---

## Tujuan Penelitian

1. Mengembangkan model komputasi pembacaan gerak bibir untuk bahasa SIBI (Sistem Isyarat Bahasa Indonesia) agar lebih efektif dan akurat.
2. Membandingkan efektivitas hasil akurasi dari pendekatan model Conv-LSTM dengan model *Optical Flow* Conv-LSTM dalam konteks pembacaan gerak bibir untuk bahasa SIBI.
3. Menghasilkan rekomendasi untuk pengembangan lebih lanjut dan implementasi teknologi pembacaan gerak bibir dalam aplikasi yang dapat digunakan oleh penyandang tunarungu di Indonesia.

---

## Manfaat Penelitian

### Bagi Praktisi
- Membantu mengembangkan alat bantu yang efektif bagi orang tunarungu untuk memahami percakapan melalui teknologi pengenalan gerak bibir, sehingga mereka dapat berkomunikasi lebih mudah dan efektif dalam berbagai situasi.
- Hasil penelitian dapat digunakan untuk mengembangkan aplikasi mobile atau perangkat lunak lainnya yang dapat diimplementasikan dalam kehidupan sehari-hari orang tunarungu, meningkatkan kualitas hidup mereka.

### Bagi Peneliti
- Menambah pengetahuan di bidang pengenalan gerak bibir dan aplikasi *deep learning* dalam konteks bahasa Isyarat Indonesia, serta memberikan wawasan baru dan dasar untuk penelitian lanjutan.
- Pengembangan dataset yang beragam dan representatif untuk bahasa SIBI dapat digunakan oleh peneliti lain untuk berbagai keperluan penelitian lebih lanjut.

### Bagi Keilmuan
- Memberikan referensi bagi penelitian yang akan datang tentang pengenalan gerak bibir untuk orang tunarungu, khususnya dalam konteks SIBI.

### Bagi Penyandang Tunarungu
- Diharapkan dapat membantu berkomunikasi dengan lebih baik bagi penyandang tunarungu ketika dibuatkan aplikasi khusus yang mengimplementasikan hasil penelitian ini.

---

## Batasan Penelitian

Penelitian ini memiliki delapan batasan yang mendefinisikan ruang lingkup eksperimen:

- Fokus pada pengenalan gerak bibir dalam bahasa SIBI dengan data yang diambil langsung menggunakan gerakan SIBI.
- Menggunakan metode Conv-LSTM dengan *Optical Flow* pada tahap *preprocessing* untuk menangkap detail perubahan gerak yang lebih optimal.
- Metode pembanding utama adalah Conv-LSTM tanpa tahap *preprocessing* *Optical Flow* (*baseline*).
- Data diambil dari guru di SLB (Sekolah Luar Biasa) dan siswa penyandang tunarungu.
- Evaluasi performa berfokus pada metrik akurasi, presisi, *recall*, dan *F1-score*.
- Dataset mencakup *alphabet* (a, b, c, d, e, dan f), bilangan (1, 2, 3, 8, 9, dan 10), dan 6 kata dasar bahasa SIBI (buku, dia, saya, keliling, kelompok, dan sekarang) — total 18 label yang dipilih secara seimbang berdasarkan jumlah suku kata.
- Data bersifat terekam (tidak *real-time*).
- Output program berupa teks.

---

## Kontribusi Penelitian

Penelitian ini memberikan lima kontribusi utama terhadap bidang lip reading bahasa Indonesia:

1. **Dataset SIBI Gabungan 18 Label**: Membangun dataset pertama pembacaan gerak bibir SIBI yang mencakup alfabet (a–f), bilangan (1, 2, 3, 8, 9, 10), dan kata dasar (buku, dia, saya, keliling, kelompok, sekarang) dari 20 subjek — 10 guru SLB dan 10 siswa penyandang tunarungu — dengan total 3.600 video rekaman.

2. **Perbandingan Sistematis Tiga Varian Model**: Membandingkan tiga pendekatan dalam satu kerangka eksperimen yang terkontrol: baseline Conv-LSTM RGB, single-stream optical flow, dan dual-stream dengan AttentionFusion, menggunakan metodologi OFAT pada dataset yang sama.

3. **Implementasi AttentionFusion untuk SIBI**: Menerapkan mekanisme attention fusion adaptif yang menggabungkan representasi visual RGB dan gerak optical flow melalui gate sigmoid yang dipelajari secara otomatis, untuk pertama kalinya pada konteks SIBI lip reading.

4. **Temuan Counter-Intuitive**: Membuktikan bahwa optical flow tidak dapat menggantikan RGB secara langsung — pendekatan single-stream optical flow (51,25% test accuracy) justru jauh lebih buruk dari baseline RGB (83,06%) — dan hanya efektif sebagai stream tambahan dalam konfigurasi dual-stream (86,63%).

5. **Analisis Berbasis Kompleksitas Suku Kata**: Memberikan analisis performa berdasarkan jumlah suku kata label (1, 2, dan 3 suku kata) yang mengungkap pola kesulitan model yang belum pernah dilaporkan sebelumnya untuk lip reading SIBI, dengan temuan bahwa model dual-stream unggul merata di semua kelompok (89,17%, 87,50%, dan 86,28%).
