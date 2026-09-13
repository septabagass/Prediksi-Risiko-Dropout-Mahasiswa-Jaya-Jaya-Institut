# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang menghadapi permasalahan mahasiswa yang tidak menyelesaikan pendidikan atau dropout.

Proyek ini bertujuan membantu Jaya Jaya Institut memahami karakteristik mahasiswa dan faktor yang berkaitan dengan status mahasiswa berdasarkan data pendaftaran, demografi, kondisi sosial-ekonomi, serta performa akademik semester 1 dan semester 2.

Hasil proyek mencakup proses data science dari business understanding, data understanding, data preparation, exploratory data analysis (EDA), modeling, evaluation, hingga deployment prototype machine learning. Selain itu, dibuat business dashboard untuk membantu monitoring kondisi mahasiswa.

> **Catatan metodologis:** dataset menyediakan performa akademik semester 1 dan semester 2. Oleh karena itu, model dalam proyek ini memprediksi status mahasiswa berdasarkan informasi hingga semester kedua. Model tidak diposisikan sebagai sistem early warning sebelum mahasiswa memulai perkuliahan.

### Permasalahan Bisnis

1. Institusi perlu memahami karakteristik mahasiswa berdasarkan status Dropout, Enrolled, dan Graduate.
2. Institusi perlu mengetahui indikator akademik dan non-akademik yang paling penting dalam prediksi status mahasiswa.
3. Institusi membutuhkan dashboard untuk memantau distribusi status mahasiswa dan indikator performa yang relevan.
4. Institusi membutuhkan prototype machine learning yang dapat membantu mengidentifikasi probabilitas status mahasiswa.
5. Institusi membutuhkan rekomendasi action items yang dapat digunakan sebagai dasar monitoring dan pendampingan mahasiswa.

### Cakupan Proyek

1. Business understanding dan perumusan permasalahan bisnis.
2. Data understanding.
3. Data preparation dan preprocessing.
4. Exploratory Data Analysis (EDA).
5. Pembangunan model klasifikasi menggunakan Random Forest.
6. Evaluasi model menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix.
7. Analisis feature importance.
8. Pembuatan business dashboard.
9. Pembuatan prototype machine learning menggunakan Streamlit.
10. Deployment prototype ke Streamlit Community Cloud.
11. Penyusunan kesimpulan dan rekomendasi action items.

### Persiapan

**Sumber data:**

Students' Performance — Dicoding Dataset

https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

**Struktur project:**

```text
jaya-jaya-institut/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   └── dropout_prediction_pipeline.pkl
└── data/
    └── dashboard_data.csv
```

**Environment yang digunakan saat development:**

```bash
conda create -n jaya-jaya-institut python=3.11 -y
conda activate jaya-jaya-institut
```

**Install dependency:**

```bash
pip install -r requirements.txt
```

**Menjalankan aplikasi secara lokal:**

```bash
streamlit run app.py
```

## Data Understanding

Dataset asli terdiri dari 4.424 data mahasiswa dan 37 kolom, dengan `Status` sebagai target.

Target memiliki tiga kelas:

- `Dropout`
- `Enrolled`
- `Graduate`

Variabel dataset mencakup informasi pendaftaran, demografi, kondisi sosial-ekonomi, performa akademik semester 1, performa akademik semester 2, serta kondisi ekonomi makro.

Untuk kebutuhan business dashboard, data juga dipersiapkan dalam file `dashboard_data.csv` dengan beberapa kolom label tambahan agar visualisasi lebih mudah dipahami.

## Data Preparation / Preprocessing

Tahapan data preparation dan preprocessing yang dilakukan:

1. Memeriksa struktur data dan tipe data.
2. Memeriksa missing value.
3. Memeriksa duplicate.
4. Memeriksa distribusi target.
5. Mempertahankan nilai akademik dalam bentuk numerik/desimal.
6. Memisahkan target `Status` dari fitur.
7. Memperlakukan variabel kode kategori sebagai categorical features.
8. Membagi data menjadi data training dan testing dengan rasio 80:20 menggunakan stratified split.
9. Melakukan median imputation pada fitur numerik.
10. Melakukan standardisasi pada fitur numerik.
11. Melakukan most-frequent imputation pada fitur kategorikal.
12. Melakukan One-Hot Encoding pada fitur kategorikal.
13. Menggabungkan seluruh preprocessing dan model ke dalam satu Pipeline agar proses preprocessing saat inference konsisten dengan proses training.

## Exploratory Data Analysis

EDA dilakukan untuk memahami karakteristik mahasiswa dan pola yang berkaitan dengan status mahasiswa.

Analisis utama meliputi:

- Distribusi jumlah mahasiswa berdasarkan status.
- Perbandingan performa akademik berdasarkan status.
- Perbandingan jumlah mata kuliah yang diambil dan diselesaikan.
- Analisis indikator semester 1.
- Analisis indikator semester 2.
- Analisis status pembayaran dan status mahasiswa.
- Analisis scholarship holder dan status mahasiswa.
- Analisis karakteristik demografis.
- Analisis program/course berdasarkan status mahasiswa.
- Analisis indikator yang digunakan dalam pemodelan.

Hasil EDA kemudian digunakan sebagai dasar untuk menentukan informasi yang perlu ditampilkan pada business dashboard dan untuk membantu interpretasi hasil model.

## Modeling

Model utama yang digunakan adalah **Random Forest Classifier**.

Konfigurasi model:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

Data dibagi menggunakan stratified split dengan rasio:

- 80% data training
- 20% data testing

Model dibangun dalam satu pipeline bersama preprocessing sehingga data input pada saat prediksi dapat diproses dengan cara yang sama seperti data training.

## Evaluation

Hasil evaluasi pada data testing:

| Metric | Result |
|---|---:|
| Accuracy | 76.50% |
| Weighted Precision | 74.87% |
| Weighted Recall | 76.50% |
| Weighted F1-score | 74.70% |
| Dropout Recall | 75.00% |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Dropout | 0.80 | 0.75 | 0.77 |
| Enrolled | 0.57 | 0.32 | 0.41 |
| Graduate | 0.78 | 0.93 | 0.85 |

### Interpretasi

Model memperoleh accuracy sebesar 76,50% pada data testing. Secara keseluruhan, model sudah dapat membedakan tiga status mahasiswa dengan performa yang cukup baik untuk digunakan sebagai prototype pendukung keputusan.

Kelas **Dropout** memiliki precision 0,80 dan recall 0,75. Artinya, model cukup baik dalam mengenali mahasiswa yang termasuk kelas Dropout, meskipun masih terdapat sebagian mahasiswa Dropout yang diprediksi sebagai kelas lain.

Kelas **Graduate** memiliki performa paling baik dengan recall 0,93 dan F1-score 0,85. Sementara itu, kelas **Enrolled** memiliki performa paling rendah dengan recall 0,32 dan F1-score 0,41. Hal ini menunjukkan bahwa model masih cukup sering kesulitan membedakan mahasiswa yang masih Enrolled dari kelas lainnya.

Dengan demikian, model layak digunakan sebagai **prototype alat bantu analisis dan prioritas monitoring**, tetapi hasil prediksi tidak sebaiknya digunakan sebagai keputusan final tanpa validasi dari pihak akademik.

### Feature Importance

Fitur dengan importance tertinggi dari model antara lain:

| Rank | Feature |
|---:|---|
| 1 | `Curricular_units_2nd_sem_approved` |
| 2 | `Curricular_units_2nd_sem_grade` |
| 3 | `Curricular_units_1st_sem_approved` |
| 4 | `Curricular_units_1st_sem_grade` |
| 5 | `Curricular_units_2nd_sem_evaluations` |
| 6 | `Curricular_units_1st_sem_evaluations` |
| 7 | `Admission_grade` |
| 8 | `Age_at_enrollment` |
| 9 | `Previous_qualification_grade` |

Fitur-fitur tersebut merupakan fitur yang paling penting secara prediktif dalam model. Hasil ini tidak boleh ditafsirkan sebagai hubungan sebab-akibat.

Secara umum, indikator performa akademik semester 1 dan semester 2 memiliki posisi penting dalam model. Hal ini menunjukkan bahwa informasi mengenai jumlah mata kuliah yang berhasil diselesaikan, nilai akademik, dan evaluasi menjadi informasi yang relevan untuk memprediksi status mahasiswa.

## Business Dashboard

Business dashboard dibuat untuk membantu pihak Jaya Jaya Institut melakukan monitoring terhadap distribusi status mahasiswa dan indikator yang relevan.

Dashboard yang digunakan dalam proyek ini adalah **Looker Studio**.

### Komponen Dashboard

Dashboard memuat beberapa komponen utama:

**KPI:**

- Total mahasiswa.
- Jumlah Dropout.
- Jumlah Enrolled.
- Jumlah Graduate.
- Dropout Rate.

**Visualisasi:**

1. Distribusi status mahasiswa.
2. Status mahasiswa berdasarkan course/program.
3. Performa akademik semester 1 berdasarkan status.
4. Performa akademik semester 2 berdasarkan status.
5. Jumlah mata kuliah yang berhasil diselesaikan.
6. Status pembayaran dan status mahasiswa.
7. Status mahasiswa berdasarkan scholarship holder.
8. Indikator akademik yang relevan berdasarkan hasil EDA.

### Dashboard

**URL dashboard:**

`[https://datastudio.google.com/reporting/33de14c0-ca42-4897-89fc-fa8765a0d29d]`

Dashboard sebaiknya dapat diakses oleh reviewer tanpa membutuhkan akun internal yang tidak tersedia bagi reviewer.

## Menjalankan Sistem Machine Learning

Prototype machine learning menggunakan model pipeline yang telah disimpan pada:

```text
model/dropout_prediction_pipeline.pkl
```

Prototype dikembangkan menggunakan Streamlit.

### Menjalankan secara lokal

Pastikan environment sudah aktif:

```bash
conda activate jaya-jaya-institut
```

Kemudian jalankan:

```bash
streamlit run app.py
```

Aplikasi menyediakan dua bagian utama:

1. **Dashboard**
   - Menampilkan ringkasan kondisi mahasiswa.
   - Menampilkan distribusi status.
   - Menampilkan beberapa indikator akademik dan non-akademik.

2. **Prediksi Risiko Dropout**
   - Pengguna memasukkan karakteristik mahasiswa.
   - Model melakukan prediksi status.
   - Output menampilkan predicted status.
   - Output menampilkan probabilitas setiap kelas.
   - Output menampilkan probabilitas Dropout.

Model menggunakan pipeline preprocessing dan Random Forest yang telah dilatih sebelumnya.

## Deployment

Aplikasi dideploy menggunakan **Streamlit Community Cloud**.

**URL aplikasi:**

`[http://localhost:8501]`

Deployment menggunakan file utama:

```text
app.py
requirements.txt
model/dropout_prediction_pipeline.pkl
data/dashboard_data.csv
```

File `requirements.txt` berisi dependency yang dibutuhkan aplikasi, termasuk versi library yang kompatibel dengan model:

```text
streamlit
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.1
joblib==1.6.0
matplotlib
```

## Conclusion

Proyek ini menghasilkan rangkaian proses data science untuk membantu Jaya Jaya Institut memahami dan memprediksi status mahasiswa. Proses dilakukan mulai dari business understanding, data understanding, data preparation, EDA, modeling, evaluation, analisis feature importance, pembuatan business dashboard, hingga deployment prototype machine learning.

Berdasarkan hasil evaluasi, model Random Forest memperoleh accuracy sebesar **76,50%**, weighted precision **74,87%**, weighted recall **76,50%**, dan weighted F1-score **74,70%**. Performa kelas Dropout cukup baik dengan precision **0,80**, recall **0,75**, dan F1-score **0,77**. Kelas Graduate memiliki performa paling tinggi, sedangkan kelas Enrolled masih menjadi kelas yang paling sulit diprediksi.

Feature importance menunjukkan bahwa indikator performa akademik, terutama jumlah mata kuliah yang disetujui dan nilai pada semester 1 dan semester 2, termasuk informasi yang paling penting secara prediktif. Hasil tersebut dapat digunakan sebagai dasar untuk monitoring mahasiswa, namun tidak dapat langsung dianggap sebagai hubungan sebab-akibat.

Prototype Streamlit dapat digunakan sebagai alat bantu untuk melihat kondisi data dan memperoleh prediksi status mahasiswa beserta probabilitasnya. Prediksi sebaiknya digunakan sebagai salah satu informasi pendukung dan tetap divalidasi oleh pihak akademik sebelum digunakan untuk menentukan intervensi kepada mahasiswa.

### Rekomendasi Action Items

1. **Monitoring mahasiswa berdasarkan indikator risiko**

   Gunakan probabilitas Dropout dari model sebagai salah satu sinyal untuk menentukan mahasiswa yang perlu mendapatkan monitoring lebih lanjut.

2. **Prioritaskan monitoring performa akademik**

   Perhatikan indikator seperti jumlah mata kuliah yang berhasil diselesaikan, nilai semester, dan jumlah evaluasi. Mahasiswa dengan performa akademik rendah dapat diarahkan ke program konsultasi atau pendampingan akademik.

3. **Pendampingan akademik**

   Bangun mekanisme pendampingan bagi mahasiswa yang menunjukkan performa akademik rendah atau pola yang konsisten dengan kelompok mahasiswa berisiko.

4. **Monitoring faktor administratif**

   Jika dashboard menunjukkan pola yang relevan, indikator seperti status pembayaran dan faktor administratif dapat dimonitor bersama indikator akademik.

5. **Monitoring program/course**

   Program atau course dengan proporsi Dropout tinggi perlu dianalisis lebih lanjut untuk menemukan faktor spesifik yang mungkin berkaitan dengan kondisi tersebut.

6. **Evaluasi model secara berkala**

   Model perlu dievaluasi menggunakan data mahasiswa terbaru agar performanya tetap relevan dan dapat disesuaikan apabila terdapat perubahan karakteristik mahasiswa.

7. **Validasi oleh pihak akademik**

   Hasil prediksi tidak digunakan sebagai keputusan final. Pihak akademik tetap perlu melakukan validasi dan mempertimbangkan kondisi mahasiswa secara langsung sebelum menentukan tindakan.

8. **Gunakan dashboard sebagai alat monitoring**

   Dashboard dapat digunakan secara berkala untuk memantau perubahan distribusi status mahasiswa dan indikator akademik sehingga institusi dapat lebih cepat menentukan area yang perlu dianalisis.
