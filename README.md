# Car Insurance Fraud Detection Model

## 📌 Project Type
Personal Project
## 📊 Overview

Project ini bertujuan untuk membangun model machine learning yang mampu mendeteksi potensi **fraud pada klaim asuransi kendaraan** berdasarkan informasi polis, karakteristik pelanggan, detail kendaraan, informasi kecelakaan, serta data klaim yang diajukan.

Pipeline dikembangkan secara **end-to-end**, mulai dari *exploratory data analysis* (EDA), *preprocessing data*, penanganan *class imbalance*, pembangunan model baseline, *hyperparameter tuning*, evaluasi model, *threshold optimization*, hingga analisis *feature importance*. Model yang dihasilkan diharapkan dapat membantu perusahaan asuransi mengidentifikasi klaim yang berpotensi fraud sehingga proses investigasi menjadi lebih efektif, efisien, dan berbasis data.


## 📂 Dataset Overview

Dataset yang digunakan merupakan data historis klaim asuransi kendaraan yang terdiri dari **30.000 data** dengan **24 variabel**, mencakup informasi polis, karakteristik pelanggan, detail kendaraan, informasi kecelakaan, serta data klaim.

Variabel target pada dataset adalah **`fraud_reported`**, dengan:

- **Y** → Klaim Fraud
- **N** → Klaim Normal

Hasil eksplorasi data menunjukkan bahwa dataset:

- Tidak memiliki data duplikat.
- Tidak memiliki *missing values*. 
- Mengalami **class imbalance**, di mana sekitar **87,62%** data merupakan klaim normal dan sekitar **12,38%** merupakan klaim fraud.

Oleh karena itu, proses pengembangan model menerapkan teknik **SMOTETomek** untuk meningkatkan kemampuan model dalam mengenali kasus fraud.


## 🎯 Objectives

- Memahami karakteristik data melalui *exploratory data analysis* (EDA).
- Melakukan *preprocessing* dan *feature encoding*.
- Menangani permasalahan *class imbalance* menggunakan SMOTETomek.
- Mengembangkan serta membandingkan beberapa model klasifikasi.
- Melakukan *hyperparameter tuning* menggunakan RandomizedSearchCV.
- Mengevaluasi performa model menggunakan berbagai metrik klasifikasi.
- Mengoptimalkan threshold klasifikasi berdasarkan F1-Score.
- Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap prediksi fraud menggunakan Gain Feature Importance.


## ⚙️ Machine Learning Pipeline

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Encoding
5. Train-Test Split
6. Class Imbalance Handling (SMOTETomek)
7. Baseline Model Development
8. Hyperparameter Tuning menggunakan RandomizedSearchCV
9. Model Comparison and Selection
10. Threshold Optimization
11. Final Model Evaluation
12. Gain Feature Importance Analysis


## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- LightGBM
- Imbalanced-learn


## 🤖 Machine Learning Models

Project ini membandingkan empat model klasifikasi, yaitu:

- Logistic Regression
- LightGBM
- Tuned Logistic Regression
- Tuned LightGBM

Hyperparameter tuning dilakukan menggunakan **RandomizedSearchCV** untuk memperoleh kombinasi parameter terbaik pada masing-masing model.


## 📈 Model Performance

### Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
|---------|---------:|----------:|--------:|----------:|--------:|------------------:|
| Logistic Regression | 0.8943 | **0.6708** | 0.2880 | 0.4030 | 0.8568 | 0.5221 |
| LightGBM | **0.9022** | 0.6612 | 0.4307 | 0.5216 | 0.8779 | 0.5671 |
| Tuned Logistic Regression | 0.8695 | 0.4672 | 0.3836 | 0.4213 | 0.8376 | 0.4549 |
| **Tuned LightGBM** | 0.8980 | 0.5903 | **0.5760** | **0.5831** | **0.8799** | **0.5793** |

Berdasarkan hasil evaluasi, **Tuned LightGBM** dipilih sebagai model terbaik karena menghasilkan **F1-Score**, **ROC-AUC**, dan **Average Precision** tertinggi dibandingkan model lainnya.

Walaupun model **LightGBM** memiliki **Accuracy** yang sedikit lebih tinggi (90,22%), metrik tersebut kurang representatif pada dataset yang mengalami *class imbalance*. Oleh karena itu, pemilihan model dilakukan berdasarkan **F1-Score**, karena mampu memberikan keseimbangan terbaik antara **Precision** dan **Recall** dalam mendeteksi klaim fraud.


## 🎯 Threshold Optimization

Alih-alih menggunakan threshold default sebesar **0,50**, project ini melakukan evaluasi terhadap beberapa nilai threshold untuk menentukan titik klasifikasi yang paling optimal.

Pemilihan threshold dilakukan berdasarkan **F1-Score**, karena metrik ini memberikan keseimbangan terbaik antara **Precision** dan **Recall**. Pendekatan ini dipilih karena dataset tidak menyediakan informasi mengenai biaya bisnis (*business cost*) dari kesalahan prediksi (*false positive* dan *false negative*), sehingga F1-Score menjadi metrik yang paling representatif.

### Hasil Evaluasi Threshold

| Threshold | Precision | Recall | F1-Score |
|----------:|----------:|--------:|----------:|
| 0.30 | 0.3596 | **0.8102** | 0.4981 |
| 0.35 | 0.5010 | 0.6837 | 0.5783 |
| 0.40 | 0.5460 | 0.6474 | 0.5924 |
| 0.45 | 0.5842 | 0.6258 | **0.6043** |
| 0.50 | 0.5903 | 0.5760 | 0.5831 |
| 0.55 | 0.6203 | 0.5101 | 0.5598 |
| 0.60 | 0.6556 | 0.4536 | 0.5362 |
| 0.65 | 0.7136 | 0.3890 | 0.5035 |
| 0.70 | **0.7547** | 0.3230 | 0.4524 |

Hasil evaluasi menunjukkan bahwa **threshold 0,45** menghasilkan **F1-Score tertinggi sebesar 0,6043**, lebih baik dibandingkan threshold default **0,50** yang hanya menghasilkan **F1-Score sebesar 0,5831**. Oleh karena itu, **threshold 0,45** dipilih sebagai threshold akhir pada proses klasifikasi.

Selain menggunakan F1-Score sebagai rekomendasi utama, threshold juga dapat disesuaikan berdasarkan kebutuhan bisnis.

| Prioritas | Kapan Digunakan | Kelebihan | Kekurangan |
|-----------|-----------------|-----------|------------|
| **Precision Tinggi** | Ketika biaya investigasi mahal dan hanya klaim yang sangat mencurigakan yang ingin diperiksa. | Mengurangi investigasi yang tidak diperlukan. | Lebih banyak kasus fraud berpotensi lolos. |
| **Recall Tinggi** | Ketika kehilangan kasus fraud jauh lebih merugikan dibanding biaya investigasi. | Menangkap lebih banyak kasus fraud. | Meningkatkan jumlah *false positive*. |
| **F1-Score Tinggi (Dipilih)** | Ketika kemampuan deteksi fraud dan efisiensi investigasi sama-sama penting. | Memberikan keseimbangan terbaik antara Precision dan Recall. | Tidak memaksimalkan Precision maupun Recall secara individual. |


## 🔍 Top 15 Gain Feature Importance

Berdasarkan hasil pelatihan **Tuned LightGBM**, lima belas fitur berikut memberikan kontribusi terbesar terhadap proses identifikasi klaim fraud berdasarkan **Gain Feature Importance**.

| Rank | Feature | Gain Importance |
|-----:|---------|----------------:|
| 1 | `claim_amount` | **479.134,72** |
| 2 | `police_report_available_YES` | **349.659,47** |
| 3 | `police_report_available_NO` | 89.668,40 |
| 4 | `collision_type_Rear Collision` | 86.005,80 |
| 5 | `collision_type_Unknown` | 65.840,77 |
| 6 | `incident_type_Vehicle Theft` | 54.446,21 |
| 7 | `witnesses` | 44.120,76 |
| 8 | `incident_type_Single Vehicle Collision` | 41.966,95 |
| 9 | `total_claim_amount` | 37.715,84 |
| 10 | `collision_type_Side Collision` | 36.113,13 |
| 11 | `collision_type_Front Collision` | 35.463,20 |
| 12 | `authorities_contacted_Police` | 29.182,86 |
| 13 | `number_of_vehicles_involved` | 23.909,56 |
| 14 | `authorities_contacted_Ambulance` | 23.369,78 |
| 15 | `authorities_contacted_Other` | 22.334,47 |

Hasil tersebut menunjukkan bahwa **nilai klaim (`claim_amount`)** merupakan faktor yang paling berpengaruh dalam mengidentifikasi potensi fraud. Selain itu, **ketersediaan laporan polisi**, **jenis tabrakan**, **jenis insiden**, **jumlah saksi**, **total nilai klaim**, serta **otoritas yang dihubungi saat kejadian** juga memberikan kontribusi yang signifikan terhadap proses prediksi model.

Secara umum, model lebih banyak memanfaatkan informasi yang berkaitan dengan **besarnya nilai klaim**, **kondisi dan karakteristik kecelakaan**, serta **bukti pendukung setelah insiden** dibandingkan karakteristik demografis pemegang polis. Hal ini menunjukkan bahwa informasi yang diperoleh selama proses pelaporan klaim memiliki peran penting dalam membedakan klaim fraud dan klaim yang sah.


## 💡 Business Recommendations

- Mengintegrasikan model **Tuned LightGBM** ke dalam proses *claim screening* sebagai alat bantu dalam mendeteksi potensi fraud.
- Menggunakan **threshold 0,45** sebagai konfigurasi awal karena memberikan keseimbangan terbaik antara kemampuan mendeteksi fraud dan efisiensi investigasi.
- Menyesuaikan threshold prediksi sesuai kebutuhan operasional perusahaan, misalnya meningkatkan **Precision** untuk mengurangi biaya investigasi atau meningkatkan **Recall** untuk meminimalkan fraud yang lolos.
- Memprioritaskan investigasi terhadap klaim dengan probabilitas fraud yang tinggi sehingga penggunaan sumber daya investigasi menjadi lebih efektif.
- Memprioritaskan kualitas data pada fitur-fitur yang memiliki kontribusi tinggi terhadap hasil prediksi berdasarkan analisis Gain Feature Importance.
- Melakukan *retraining* model secara berkala menggunakan data klaim terbaru agar performa model tetap optimal.
