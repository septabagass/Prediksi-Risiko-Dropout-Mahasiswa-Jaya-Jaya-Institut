# Proyek Akhir: Sistem Prediksi Risiko Dropout Mahasiswa Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang menghadapi permasalahan mahasiswa yang tidak menyelesaikan pendidikan atau dropout.

Proyek ini menggunakan pendekatan data science untuk memahami karakteristik mahasiswa berdasarkan data pendaftaran, demografi, kondisi sosial-ekonomi, dan performa akademik semester pertama serta kedua. Selain analisis data dan dashboard, proyek menyediakan model machine learning yang dapat membantu memprediksi status mahasiswa.

> Catatan metodologis: karena dataset menyediakan performa akademik semester 1 dan 2, model ini memprediksi status berdasarkan data hingga semester kedua. Model tidak diposisikan sebagai early warning sebelum mahasiswa memulai perkuliahan.

### Permasalahan Bisnis

1. Tingginya jumlah mahasiswa yang berstatus dropout perlu dipahami dari perspektif data.
2. Institusi membutuhkan informasi mengenai karakteristik akademik dan non-akademik yang berkaitan dengan status mahasiswa.
3. Institusi membutuhkan dashboard untuk memantau distribusi dan faktor performa mahasiswa.
4. Institusi membutuhkan prototype machine learning untuk membantu mengidentifikasi probabilitas status mahasiswa.

### Cakupan Proyek

1. Data understanding.
2. Data preparation dan preprocessing.
3. Exploratory Data Analysis (EDA).
4. Pembangunan model klasifikasi.
5. Evaluasi menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix.
6. Analisis feature importance.
7. Pembuatan business dashboard.
8. Pembuatan prototype Streamlit.
9. Deployment prototype ke Streamlit Community Cloud.
10. Penyusunan rekomendasi action items.

### Persiapan

**Sumber data:**  
Students' Performance — Dicoding Dataset  
https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

**Environment:**

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Menjalankan aplikasi:

```bash
streamlit run app.py
```

---

## Data Understanding

Dataset terdiri dari 4.424 data mahasiswa dan 37 kolom, termasuk target `Status`.

Target memiliki tiga kelas:

- `Dropout`
- `Enrolled`
- `Graduate`

Variabel dataset mencakup informasi pendaftaran, demografi, sosial-ekonomi, performa semester 1, performa semester 2, dan kondisi ekonomi makro.

---

## Data Preparation / Preprocessing

Tahapan preprocessing:

1. Memeriksa missing value.
2. Memeriksa duplicate.
3. Mempertahankan nilai akademik dalam bentuk desimal.
4. Memisahkan target `Status` dari fitur.
5. Memperlakukan variabel kode kategori sebagai categorical features.
6. Melakukan median imputation dan standardisasi pada fitur numerik.
7. Melakukan most-frequent imputation dan One-Hot Encoding pada fitur kategorikal.
8. Menggabungkan preprocessing dan model dalam satu Pipeline.

---

## Exploratory Data Analysis

EDA digunakan untuk memahami pola data sebelum modeling.

Analisis utama:

- Distribusi status mahasiswa.
- Performa akademik berdasarkan status.
- Perbedaan jumlah mahasiswa yang lulus, masih terdaftar, dan dropout.
- Hubungan indikator akademik dan status mahasiswa.
- Faktor administratif dan sosial-ekonomi yang relevan.

---

## Modeling

Model utama yang digunakan adalah **Random Forest Classifier**.

Konfigurasi:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

Data dibagi dengan rasio 80:20 menggunakan stratified split.

---

## Evaluation

**Isi bagian berikut setelah notebook selesai dijalankan. Jangan mengarang angka.**

| Metric | Result |
|---|---:|
| Accuracy | `[ISI HASIL]` |
| Weighted Precision | `[ISI HASIL]` |
| Weighted Recall | `[ISI HASIL]` |
| Weighted F1-score | `[ISI HASIL]` |
| Dropout Recall | `[ISI HASIL]` |

### Interpretasi

Tuliskan:

- apakah model cukup baik membedakan tiga status;
- bagaimana performa kelas Dropout;
- kelas mana yang paling sering tertukar;
- apakah model cukup layak digunakan sebagai prototype pendukung keputusan.

### Feature Importance

Masukkan 5–10 fitur dengan importance tertinggi dari notebook.

> Feature importance menunjukkan kontribusi prediktif dalam model dan tidak boleh ditafsirkan sebagai hubungan sebab-akibat.

---

## Business Dashboard

Dashboard dibuat untuk membantu monitoring status dan performa mahasiswa.

### Komponen dashboard

**KPI:**

- Total mahasiswa
- Jumlah Dropout
- Jumlah Enrolled
- Jumlah Graduate
- Dropout Rate

**Visualisasi:**

1. Distribusi status mahasiswa.
2. Status berdasarkan program/course.
3. Performa semester 1 berdasarkan status.
4. Performa semester 2 berdasarkan status.
5. Jumlah mata kuliah yang berhasil diselesaikan.
6. Status pembayaran dan status mahasiswa.
7. Status mahasiswa berdasarkan scholarship.
8. Faktor akademik yang relevan berdasarkan hasil EDA.

### Dashboard

**URL / akses dashboard:** `[ISI LINK ATAU INFORMASI AKSES]`

Jika menggunakan Metabase dan requirement meminta kredensial:

- Email: `[ISI EMAIL YANG DIGUNAKAN]`
- Password: `[ISI PASSWORD YANG DIGUNAKAN]`

> Jangan memasukkan password pribadi atau password akun penting ke repository publik. Gunakan akun khusus untuk kebutuhan proyek jika diperlukan.

---

## Menjalankan Sistem Machine Learning

Aplikasi Streamlit menggunakan pipeline model yang telah disimpan di:

```text
model/dropout_prediction_pipeline.pkl
```

Jalankan secara lokal:

```bash
streamlit run app.py
```

Aplikasi menyediakan input karakteristik mahasiswa, performa semester 1 dan 2, serta kondisi ekonomi makro.

Output:

- Predicted Status
- Probabilitas setiap kelas
- Probabilitas Dropout

---

## Deployment

Aplikasi dideploy menggunakan **Streamlit Community Cloud**.

**URL aplikasi:** `[ISI URL STREAMLIT SETELAH DEPLOYMENT]`

---

## Conclusion

Berdasarkan analisis yang dilakukan, proyek ini menghasilkan proses data science dari data understanding hingga deployment prototype machine learning.

Hasil EDA digunakan untuk memahami karakteristik mahasiswa dan mendukung pembuatan dashboard monitoring. Model Random Forest digunakan untuk memprediksi status mahasiswa menjadi Dropout, Enrolled, atau Graduate.

**Kesimpulan berbasis hasil aktual:**

`[TULIS 1–2 PARAGRAF BERDASARKAN HASIL MODEL DAN DASHBOARD]`

---

### Rekomendasi Action Items

1. **Monitoring mahasiswa berdasarkan indikator risiko**  
   Gunakan probabilitas Dropout dari model sebagai salah satu sinyal untuk menentukan mahasiswa yang perlu dipantau lebih lanjut.

2. **Pendampingan akademik**  
   Mahasiswa dengan performa akademik rendah dapat diarahkan ke program konsultasi atau pendampingan akademik.

3. **Monitoring faktor administratif**  
   Jika hasil EDA menunjukkan hubungan yang kuat, indikator seperti status pembayaran dapat dimonitor bersama faktor akademik.

4. **Monitoring program/course**  
   Program dengan proporsi dropout tinggi perlu dianalisis lebih lanjut untuk menemukan penyebab spesifik.

5. **Evaluasi model secara berkala**  
   Model perlu dievaluasi ulang menggunakan data mahasiswa terbaru agar performanya tetap relevan.

6. **Validasi oleh pihak akademik**  
   Hasil prediksi tidak digunakan sebagai keputusan final, tetapi sebagai alat bantu untuk menentukan prioritas intervensi.
