# Student Academic Stress Analysis  

Proyek ini bertujuan untuk menganalisis dan memprediksi tingkat stres akademik pelajar menggunakan dataset **Student Academic Stress**. Analisis dilakukan mulai dari *Exploratory Data Analysis (EDA)*, preprocessing data, hingga penerapan berbagai algoritma *Machine Learning* untuk klasifikasi.  

## 📂 Dataset  
Dataset yang digunakan adalah **Student Academic Stress Dataset (real world dataset)** yang mencakup informasi kondisi pelajar seperti:  
- Gender  
- Usia  
- Kegiatan akademik  
- Faktor sosial & keluarga  
- Tingkat stres  

Dataset ini diolah melalui:  
- Encoding kategori → Label Encoding & One-Hot Encoding  
- Normalisasi & Standarisasi  
- Penyeimbangan kelas → SMOTE & ADASYN  

## 🛠 Tools & Libraries  
Proyek ini menggunakan beberapa tools & library Python:  
- **Data Manipulation**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Preprocessing**: `sklearn.preprocessing` (LabelEncoder, OneHotEncoder, StandardScaler)  
- **Modeling**:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Classifier (SVC)  
- **Model Evaluation**:  
  - `accuracy_score`, `classification_report`, `confusion_matrix`  
  - `f1_score`, `precision_score`, `recall_score`  
- **Balancing Data**: `imblearn` (SMOTE, ADASYN)  

## 🔎 Exploratory Data Analysis (EDA)  
- Visualisasi distribusi nilai tiap kolom (histogram, boxplot)  
- Heatmap korelasi antar fitur  
- Analisis penyebaran kelas target  

## 🤖 Machine Learning Models  
Beberapa model dicoba dengan *GridSearchCV* untuk mencari *hyperparameter terbaik*. Evaluasi dilakukan pada data *train* dan *test*.  

### Hasil Evaluasi  
- Performa model diukur berdasarkan **Accuracy, F1-score, Precision, Recall**.  
- Hasil ditampilkan dalam bentuk **Confusion Matrix** untuk tiap model.  
- Model terbaik ditentukan berdasarkan **Test Accuracy tertinggi**.  

📊 Ringkasan hasil (urut berdasarkan akurasi test) ditampilkan di notebook dalam bentuk tabel `results_df`.  

## 🚀 Cara Menjalankan  
1. Clone repositori / unduh notebook ini  
2. Install dependensi yang diperlukan:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
