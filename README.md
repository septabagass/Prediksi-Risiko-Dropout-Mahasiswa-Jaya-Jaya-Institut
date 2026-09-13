# Proyek Akhir: Sistem Prediksi Risiko Dropout Mahasiswa Jaya Jaya Institut

## Business Understanding

### Latar Belakang

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang menghadapi permasalahan mahasiswa yang tidak menyelesaikan pendidikan atau mengalami dropout. Kondisi ini dapat berdampak pada keberhasilan akademik mahasiswa maupun kinerja institusi.

Untuk membantu memahami permasalahan tersebut, proyek ini membangun sistem analisis dan prediksi status mahasiswa berdasarkan data demografis, informasi pendidikan, kondisi sosial-ekonomi, serta performa akademik mahasiswa.

Solusi yang dibangun terdiri dari:

1. Exploratory Data Analysis (EDA) untuk memahami karakteristik mahasiswa.
2. Business dashboard untuk membantu monitoring kondisi mahasiswa.
3. Model machine learning untuk memprediksi apakah mahasiswa memiliki status **Dropout** atau **Graduate**.
4. Prototype aplikasi berbasis Streamlit yang dapat digunakan untuk melakukan prediksi.

### Permasalahan Bisnis

Berdasarkan permasalahan tersebut, proyek ini berfokus pada beberapa pertanyaan:

1. Bagaimana distribusi status mahasiswa di Jaya Jaya Institut?
2. Karakteristik apa saja yang berkaitan dengan status Dropout dan Graduate?
3. Bagaimana hubungan performa akademik, usia, gender, kondisi pembayaran, beasiswa, dan atribut lainnya dengan status mahasiswa?
4. Bagaimana membangun model machine learning yang dapat membantu mengidentifikasi mahasiswa yang memiliki pola lebih dekat dengan kelompok Dropout atau Graduate?
5. Bagaimana menyajikan hasil analisis dan prediksi agar dapat digunakan oleh pihak institusi sebagai alat bantu monitoring?

### Tujuan Proyek

Proyek ini bertujuan untuk:

1. Memahami karakteristik mahasiswa berdasarkan status Dropout, Enrolled, dan Graduate.
2. Mengidentifikasi faktor-faktor yang berkaitan dengan status Dropout.
3. Membandingkan karakteristik mahasiswa Dropout dan Graduate.
4. Membangun model klasifikasi binary untuk memprediksi Dropout atau Graduate.
5. Menyediakan dashboard interaktif untuk membantu monitoring mahasiswa.
6. Menyediakan prototype machine learning yang dapat digunakan untuk melakukan prediksi berdasarkan data mahasiswa.

### Cakupan Proyek

Tahapan yang dilakukan dalam proyek ini meliputi:

1. Business Understanding.
2. Data Understanding.
3. Data Preparation.
4. Exploratory Data Analysis (EDA).
5. Preprocessing data.
6. Pembangunan model machine learning.
7. Evaluasi model.
8. Analisis feature importance.
9. Pembuatan business dashboard.
10. Pembuatan prototype machine learning menggunakan Streamlit.
11. Penyimpanan model untuk digunakan pada aplikasi.
12. Deployment aplikasi menggunakan Streamlit Community Cloud.

> **Catatan metodologis:** Dataset menyediakan informasi performa akademik hingga semester 2. Oleh karena itu, model menggunakan informasi yang tersedia hingga semester 2. Model ini tidak diposisikan sebagai sistem early warning sebelum mahasiswa memulai perkuliahan.

---

## Data Understanding

Dataset yang digunakan adalah **Students' Performance** yang disediakan melalui Dicoding Dataset.

Sumber dataset:

https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

Dataset terdiri dari:

- **4.424 baris**
- **37 kolom**
- Target awal berupa tiga status mahasiswa:
  - `Dropout`
  - `Enrolled`
  - `Graduate`

Data mencakup beberapa kelompok informasi, antara lain:

- Informasi demografis mahasiswa.
- Informasi pendaftaran.
- Pendidikan sebelumnya.
- Informasi orang tua.
- Kondisi sosial dan ekonomi.
- Status pembayaran tuition fees.
- Status beasiswa.
- Performa akademik semester 1.
- Performa akademik semester 2.

### Status Mahasiswa

Dalam dataset terdapat tiga status:

| Status | Keterangan |
|---|---|
| Dropout | Mahasiswa yang tidak menyelesaikan pendidikan |
| Enrolled | Mahasiswa yang masih berstatus enrolled dan belum memiliki label akhir |
| Graduate | Mahasiswa yang menyelesaikan pendidikan |

### Data untuk Modeling

Sesuai tujuan prediksi, model machine learning hanya menggunakan data dengan status:

- `Dropout`
- `Graduate`

Data `Enrolled` tidak digunakan dalam proses training karena status tersebut belum merupakan label akhir.

Hasil filtering:

| Data | Jumlah |
|---|---:|
| Dropout | 1.421 |
| Graduate | 2.209 |
| Total data modeling | 3.630 |
| Enrolled yang dipisahkan | 794 |

Data `Enrolled` disimpan secara terpisah dalam:

```text
data/enrolled_for_future_prediction.csv
