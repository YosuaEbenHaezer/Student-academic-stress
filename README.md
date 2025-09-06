# Student Academic Stress Analysis  

Proyek ini bertujuan untuk menganalisis dan memprediksi tingkat stres akademik pelajar menggunakan dataset **Student Academic Stress**. Analisis dilakukan mulai dari *Exploratory Data Analysis (EDA)*, preprocessing data, hingga penerapan berbagai algoritma *Machine Learning* untuk klasifikasi.  

## 📂 Dataset  
Dataset yang digunakan adalah **Student Academic Stress Dataset (real world dataset)** yang mencakup informasi kondisi pelajar seperti:  
 0   Timestamp                                                            
 1   Your Academic Stage                                                 
 2   Peer pressure                                                        
 3   Academic pressure from your home                                     
 4   Study Environment                                                    
 5   What coping strategy you use as a student?                           
 6   Do you have any bad habits like smoking, drinking on a daily basis?  
 7   What would you rate the academic  competition in your student life   
 8   Rate your academic stress index  
 
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

          import os
       import numpy as np
       import pandas as pd
       import seaborn as sns
       import matplotlib.pyplot as plt
       import itertools
       from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
       from sklearn.model_selection import train_test_split, GridSearchCV
       from sklearn.metrics import (
           mean_absolute_error, mean_squared_error,
           accuracy_score, classification_report,
           confusion_matrix, f1_score, precision_score, recall_score
       )
       
       from sklearn.linear_model import LogisticRegression
       from sklearn.tree import DecisionTreeClassifier
       from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.svm import SVC
       from imblearn.over_sampling import SMOTE

3. Jalankan Jupyter Notebook:
   jupyter notebook Stress_Pelajar_(1).ipynb
   
  ### 🔗 Cara Mengakses Dataset
Gunakan `kagglehub` untuk mengunduh dataset:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("poushal02/student-academic-stress-real-world-dataset")

print("Path to dataset files:", path)

