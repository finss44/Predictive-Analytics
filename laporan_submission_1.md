# Laporan Proyek Machine Learning - Elfin Darmawan

## Domain Proyek
### Latar Belakang

Diabetes mellitus adalah gangguan metabolik kronis yang ditandai dengan tingginya kadar glukosa dalam darah (hiperglikemia). Penyakit ini menjadi salah satu penyebab utama kematian di dunia dan merupakan beban besar bagi sistem kesehatan global. Menurut World Health Organization (WHO), sekitar 422 juta orang di seluruh dunia hidup dengan diabetes, dan hampir 1,5 juta kematian setiap tahun disebabkan langsung oleh penyakit ini. Jumlah penderita diperkirakan akan terus meningkat secara signifikan dalam beberapa dekade mendatang, terutama di negara berkembang yang sedang mengalami transisi gaya hidup.

### Kenapa Masalah Ini Penting?

Kebanyakan deteksi diabetes masih dilakukan melalui pemeriksaan langsung seperti tes darah yang bisa mahal dan memakan waktu. Dengan meningkatnya jumlah pasien, rumah sakit atau klinik membutuhkan sistem pendukung keputusan yang cepat dan efisien dalam mendeteksi risiko diabetes hanya dari data kesehatan dasar.

Dengan kemajuan teknologi data dan machine learning, kini dimungkinkan untuk membangun sistem yang dapat memprediksi kemungkinan seseorang menderita diabetes hanya dari data kesehatan dasar seperti kadar glukosa, tekanan darah, indeks massa tubuh (BMI), jumlah kehamilan, usia, dan riwayat keluarga.

### Referensi:
World Health Organization. (2021). Diabetes. https://www.who.int/news-room/fact-sheets/detail/diabetes

American Diabetes Association. (2020). Standards of Medical Care in Diabetes. https://care.diabetesjournals.org

Centers for Disease Control and Prevention (CDC).
National Diabetes Statistics Report (2020).
https://www.cdc.gov/diabetes/data/statistics-report/index.html
## Business Understanding
### Problem Statements

Diabetes merupakan penyakit kronis yang dapat menyebabkan komplikasi serius jika tidak terdeteksi sejak dini. Pemeriksaan laboratorium sebagai metode diagnosis memerlukan biaya dan waktu, serta tidak selalu tersedia di daerah dengan fasilitas kesehatan terbatas. Oleh karena itu, dibutuhkan suatu sistem prediksi berbasis machine learning yang mampu mengidentifikasi kemungkinan seseorang menderita diabetes hanya berdasarkan data kesehatan dasar seperti kadar glukosa, tekanan darah, BMI, usia, dan riwayat kehamilan.

### Goals

- Mengembangkan model prediksi diabetes berbasis machine learning menggunakan data kesehatan dasar.
- Meningkatkan kemampuan deteksi dini diabetes melalui prediksi berbasis data.
- Membandingkan performa dari beberapa model (Logistic Regression, Random Forest, dan KNN).
- Menentukan model terbaik berdasarkan metrik evaluasi klasifikasi.

### Solution statements
Untuk mencapai tujuan tersebut, berikut solusi yang dirancang:

1. Membangun beberapa model machine learning:
   - **Logistic Regression:** Sebagai baseline model karena sederhana dan interpretatif.
   - **Random Forest:** Untuk menangani hubungan non-linear dan interaksi antar fitur.
   - **K-Nearest Neighbors (KNN):** Sebagai metode berbasis jarak yang mudah digunakan.
2. Melakukan training dan evaluasi model menggunakan data yang telah dibersihkan dan diproses.
3. Menggunakan metrik evaluasi klasifikasi yang relevan untuk menilai kinerja model:
   - **Accuracy:** Seberapa sering model membuat prediksi yang benar.
   - **Precision:** Seberapa banyak dari prediksi positif yang benar-benar positif.
   - **Recall:** Seberapa banyak kasus positif yang berhasil ditangkap model.
   - **F1-score:** Harmoni antara precision dan recall.
4. Memilih model terbaik berdasarkan performa evaluasi dan stabilitasnya pada data uji.

## Data Understanding
Dataset ini merupakan dataset kesehatan yang digunakan untuk memprediksi apakah seseorang menderita diabetes berdasarkan parameter diagnostik medis dasar. Dataset ini berasal dari penelitian National Institute of Diabetes and Digestive and Kidney Diseases, dan digunakan dalam berbagai proyek pembelajaran mesin dan statistik.

Sumber dataset:
https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

### Variabel-variabel pada Diabetes dataset adalah sebagai berikut:
1. **Pregnancies**               : Jumlah kehamilan pasien
2. **Glucose**                   : Kadar glukosa dalam darah (mg/dL)
3. **BloodPressure**             : Tekanan darah diastolik (mm Hg)
4. **SkinThickness**             : Ketebalan lipatan kulit triceps (mm)
5. **Insulin**                   : Kadar insulin serum (mu U/ml)
6. **BMI**                       : Body Mass Index (berat/tinggi² dalam kg/m²)
7. **DiabetesPedigreeFunction**  : Indeks silsilah diabetes (riwayat keluarga)
8. **Age**                       : Usia pasien (tahun)
9. **Outcome**                   : Label target (1 = diabetes, 0 = tidak diabetes)

### Exploratory Data Analysis (EDA)
- Menampilkan dan mengecek informasi umum dataset menggunakan `.info`
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
```
Pada dataset ini, untuk setiap kolom tertera 768 non-null. Hal ini menunjukan bahwa tidak ada nilai yang hilang (missing value) pada dataset. Kolom-kolom pada dataset ini mayoritas tipe datanya adalah integer, kecuali kolom `BMI` dan `DiabetesPedigreeFunction` tipe datanya float.
- Melihat statistik deskriptif dari dataset menggunakan `.describe`
```
        Pregnancies	Glucose	        BloodPressure	SkinThickness	Insulin	        BMI	   DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
```
Terlihat nilai tengah setiap kolom, contohnya: Rata-rata jumlah kehamilan adalah sekitar 3.85. Rata-rata kadar glukosa adalah sekitar 120.9. Selain itu, mengukur seberapa tersebar nilai-nilai dalam setiap kolom dari nilai rata-ratanya. Standar deviasi yang lebih tinggi menunjukkan penyebaran data yang lebih luas. Contohnya, standar deviasi untuk Glucose (31.97) lebih tinggi daripada Pregnancies (3.37), menunjukkan variasi kadar glukosa yang lebih besar dibandingkan dengan jumlah kehamilan. Perhatikan juga standar deviasi yang sangat tinggi pada kolom Insulin (115.24), mengindikasikan variasi nilai insulin yang signifikan. Beberapa kolom memiliki nilai minimum 0, sehingga memerlukan penanganan khusus pada kolom tersebut.

- Mengecek Missing Value dan data duplikat menggunakan `.isnull().sum()` dan `.duplicated().sum()`
```
Pregnancies	                0
Glucose	                        0
BloodPressure	                0
SkinThickness	                0
Insulin	                        0
BMI	                        0
DiabetesPedigreeFunction	0
Age	                        0
Outcome	                        0
dtype: int64
```
Hasil dari pengecekan missing value diatas menunjukan bahwa tidak ada nilai null atau hilang dalam dataset untuk semua kolom yang diperiksa. Sehingga kita tidak perlu melakukan penanganan untuk missing value.
```
Jumlah data duplikat: 0
```
Pada tahap ini adalah pengecekan jumlah data duplikat, bisa kita lihat bahwa tidak ada data duplikat sehingga bisa melanjutkan ke tahap selanjutnya yaitu mengecek distribusi fitur dan outliers.

- 

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
