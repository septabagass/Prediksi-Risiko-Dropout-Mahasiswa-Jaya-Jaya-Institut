# Proyek Akhir: Sistem Prediksi Risiko Dropout Mahasiswa Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang menghadapi tantangan dalam mempertahankan mahasiswa hingga menyelesaikan pendidikan. Mahasiswa yang tidak menyelesaikan pendidikan (dropout) dapat berdampak pada keberhasilan akademik mahasiswa dan kinerja institusi.

Proyek ini menggunakan data mahasiswa yang mencakup informasi demografis, pendaftaran, pendidikan sebelumnya, kondisi sosial-ekonomi, pembayaran biaya pendidikan, serta performa akademik semester 1 dan semester 2. Analisis dilakukan untuk memahami karakteristik mahasiswa berdasarkan status akhirnya, membangun dashboard monitoring, dan mengembangkan prototype machine learning untuk membedakan status **Dropout** dan **Graduate**.

### Permasalahan Bisnis

1. Institusi perlu mengetahui distribusi mahasiswa berdasarkan status Dropout, Enrolled, dan Graduate.
2. Institusi perlu memahami karakteristik yang berkaitan dengan status Dropout, terutama dari sisi usia, performa akademik, dan kondisi finansial.
3. Institusi membutuhkan sarana visualisasi untuk memantau pola status mahasiswa dan mengeksplorasi hubungan antaratribut.
4. Institusi membutuhkan prototype model yang dapat membantu mengidentifikasi pola mahasiswa yang lebih dekat dengan kelompok Dropout atau Graduate.

### Cakupan Proyek

Cakupan pekerjaan dalam proyek ini meliputi:

1. Memahami struktur, tipe data, dan distribusi status pada dataset.
2. Melakukan pembersihan dan persiapan data.
3. Melakukan Exploratory Data Analysis (EDA) terhadap status mahasiswa, usia, gender, performa akademik, kondisi finansial, dan korelasi fitur numerik.
4. Memisahkan data `Enrolled` dari data pelatihan karena statusnya belum merupakan hasil akhir.
5. Membangun model klasifikasi binary menggunakan data berstatus `Dropout` dan `Graduate`.
6. Mengevaluasi model menggunakan Accuracy, Precision, Recall, F1-Score, Classification Report, dan Confusion Matrix.
7. Menyajikan hasil analisis dalam business dashboard.
8. Menyediakan prototype prediksi menggunakan Streamlit.

Model menggunakan fitur yang tersedia hingga semester 2. Oleh karena itu, model ini tidak diklaim sebagai sistem early warning sebelum mahasiswa memulai perkuliahan. Prediksi juga perlu dipakai sebagai alat bantu, bukan sebagai keputusan otomatis terhadap mahasiswa.

---

### Persiapan

**Sumber data:** Students' Performance dari Dicoding Dataset.

https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

Dataset terdiri dari 4.424 baris dan 37 kolom dengan tiga status akhir/tercatat, yaitu `Dropout`, `Enrolled`, dan `Graduate`.

Distribusi status pada dataset:

| Status | Jumlah | Persentase |
|---|---:|---:|
| Graduate | 2.209 | 49,93% |
| Dropout | 1.421 | 32,12% |
| Enrolled | 794 | 17,95% |
| **Total** | **4.424** | **100%** |

Untuk pemodelan, hanya data `Dropout` dan `Graduate` yang digunakan (3.630 baris). Data `Enrolled` (794 baris) dipisahkan karena belum memiliki label akhir.

**Setup environment**

Disarankan menggunakan Python 3.10 atau versi yang kompatibel dengan library pada proyek. Buat virtual environment, aktifkan, lalu pasang library berikut:

```bash
py -3.10 -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --only-binary=:all: streamlit pandas==2.2.3 numpy==2.1.3 scikit-learn==1.6.1 joblib==1.6.0 matplotlib==3.10.9 seaborn
```

-- instal dependensi menggunakan:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pastikan file dataset ditempatkan sesuai path yang digunakan di notebook. Contoh jika file berada di folder `data`:

```python
df = pd.read_csv("data/data.csv", delimiter=";")
```

Sesuaikan nama file dan lokasi tersebut dengan struktur folder proyek yang sebenarnya.

## Business Dashboard

Business dashboard dibuat untuk membantu pihak institusi mengeksplorasi data mahasiswa berdasarkan status, usia, gender, performa akademik, dan kondisi finansial. Dashboard ditujukan untuk menyajikan pola data secara visual agar pihak institusi dapat melakukan monitoring dan menentukan area yang perlu mendapat perhatian lebih lanjut.

Link dashboard:
https://datastudio.google.com/reporting/33de14c0-ca42-4897-89fc-fa8765a0d29d

## Menjalankan Sistem Machine Learning

Prototype machine learning dibuat menggunakan Streamlit. Model melakukan klasifikasi binary dengan pemetaan target:

- `0` = Graduate
- `1` = Dropout

Data `Enrolled` tidak digunakan saat training model karena belum memiliki status akhir. Model dilatih menggunakan 36 fitur dari data Dropout dan Graduate.

**Menjalankan aplikasi secara lokal**

Pastikan struktur file proyek mencakup file aplikasi, model hasil training, dan dependensi. Contoh perintah untuk menjalankan aplikasi dari direktori proyek:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Setelah perintah dijalankan, buka alamat lokal yang ditampilkan Streamlit pada terminal (umumnya `http://localhost:8501`).

**Prototype yang telah di-deploy:**

https://prediksi-risiko-dropout.streamlit.app/

## Conclusion

Berdasarkan hasil analisis terhadap 4.424 data mahasiswa, terdapat 2.209 mahasiswa berstatus Graduate, 1.421 Dropout, dan 794 Enrolled. Untuk membangun model machine learning, data Enrolled dipisahkan karena belum memiliki label akhir. Model klasifikasi binary dilatih menggunakan 3.630 data yang terdiri dari Dropout dan Graduate, dengan target 0 = Graduate dan 1 = Dropout.

Hasil Exploratory Data Analysis menunjukkan beberapa karakteristik yang berbeda antara kelompok Graduate dan Dropout:

1. **Usia:** Rata-rata usia saat pendaftaran kelompok Dropout adalah 26,07 tahun, lebih tinggi dibandingkan Graduate sebesar 21,78 tahun.
2. **Performa akademik:** Rata-rata nilai kelompok Dropout lebih rendah, yaitu 7,26 pada semester 1 dan 5,90 pada semester 2. Sementara itu, kelompok Graduate memiliki rata-rata 12,64 dan 12,70.
3. **Mata kuliah yang disetujui:** Kelompok Dropout rata-rata menyelesaikan 2,55 mata kuliah pada semester 1 dan 1,94 pada semester 2. Kelompok Graduate memiliki rata-rata 6,23 dan 6,18.
4. **Kondisi finansial:** Proporsi mahasiswa Dropout yang tuition fees-nya belum lunas adalah 32,16%, sedangkan pada kelompok Graduate sebesar 1,31%. Proporsi mahasiswa dengan status debitur juga lebih tinggi pada kelompok Dropout.

Model Random Forest menghasilkan Accuracy sebesar 92,84%, Precision Dropout 93,61%, Recall Dropout 87,68%, dan F1-Score Dropout 90,55% pada data pengujian.

Dengan demikian, mahasiswa dalam kelompok Dropout pada dataset ini cenderung memiliki usia pendaftaran lebih tinggi, performa akademik lebih rendah, jumlah mata kuliah yang disetujui lebih sedikit, serta proporsi masalah pembayaran yang lebih besar.

Hasil analisis tersebut merupakan hubungan berdasarkan data historis dan tidak membuktikan hubungan sebab-akibat. Model dapat digunakan sebagai alat bantu identifikasi pola risiko, tetapi hasil prediksi tetap perlu ditinjau oleh pihak akademik.

Model ini menggunakan informasi hingga semester 2 sehingga belum dapat diklaim sebagai sistem peringatan dini sebelum mahasiswa memulai perkuliahan.


### Rekomendasi Action Items

- **Monitoring performa akademik:** lakukan pemantauan berkala terhadap jumlah mata kuliah yang diambil, dievaluasi, dan disetujui serta nilai mahasiswa. Prioritaskan tindak lanjut akademik bagi mahasiswa yang menunjukkan penurunan performa.
- **Dukungan finansial:** identifikasi mahasiswa yang memiliki tunggakan tuition fees atau status debitur, kemudian tawarkan konsultasi, informasi bantuan, atau skema pembayaran sesuai kebijakan institusi.
- **Pendampingan yang sesuai kebutuhan:** gunakan informasi usia dan kondisi mahasiswa sebagai konteks untuk menyediakan layanan konsultasi akademik atau pendampingan yang relevan, bukan sebagai dasar untuk memberi label negatif.
- **Gunakan prediksi secara bertanggung jawab:** jadikan skor model sebagai sinyal untuk pemeriksaan lebih lanjut oleh tim akademik. Jangan menjadikan prediksi sebagai satu-satunya dasar keputusan terhadap mahasiswa.
- **Evaluasi model secara berkala:** uji model pada data periode terbaru, pantau terutama Recall kelas Dropout, dan lakukan pelatihan ulang jika pola data atau performa model berubah.
- **Kembangkan peringatan lebih dini:** karena model saat ini menggunakan informasi hingga semester 2, siapkan evaluasi terpisah jika institusi ingin membuat sistem peringatan pada semester yang lebih awal dengan hanya menggunakan fitur yang tersedia pada waktu tersebut.
