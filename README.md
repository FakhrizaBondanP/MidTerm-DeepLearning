# MidTerm-DeepLearning

Nama : Fakhriza Bondan P.
NIM : 1103223146

- `clusteringmidterm.csv` → untuk `Clustering.ipynb`  
- `train_transaction.csv`, `test_transaction.csv` → untuk `Fraud_Detection.ipynb`  
- Dataset regresi (mis. `midterm_regresi.csv`) → untuk `Midterm-Regresi.ipynb`  

## Struktur Proyek

| File/Folder              | Deskripsi                                                                 |
|--------------------------|---------------------------------------------------------------------------|
| `Clustering.ipynb`       | EDA, preprocessing, PCA, dan clustering (K-Means, Agglomerative, DBSCAN). [file:1] |
| `Fraud_Detection.ipynb`  | Pipeline deteksi fraud dengan encoding, SMOTE, scaling, dan NN PyTorch. [file:3]   |
| `Midterm-Regresi.ipynb`  | Pipeline regresi dengan Ridge + GridSearchCV dan NN Keras/TensorFlow. [file:2]     |
| `README.md`              | Dokumentasi proyek dan panduan menjalankan notebook.                                |

## Ringkasan Tiap Notebook

### Clustering.ipynb

- Load dataset kartu kredit dengan 18 fitur numerik dan ID nasabah. 
- Preprocessing: imputasi median, IQR capping outlier, feature engineering, dan scaling `RobustScaler`.
- PCA 2D dan pencarian jumlah cluster optimal menggunakan Elbow & Silhouette Score. 
- Clustering dengan K-Means, Agglomerative Clustering, dan DBSCAN + evaluasi (Silhouette & Davies–Bouldin).

### Fraud_Detection.ipynb

- Load data `train_transaction.csv` dan `test_transaction.csv` dengan ratusan ribu baris dan ±394 fitur, lalu hitung fraud rate dari label `isFraud`.
- Preprocessing numerik & kategorik, termasuk imputasi, encoding biner fitur-fitur seperti `M1–M9` dan `card6`, serta scaling fitur numerik.  
- Penanganan class imbalance menggunakan SMOTE sebelum training model.[file:3]  
- Definisi model neural network PyTorch, training loop dengan `DataLoader`, dan evaluasi (confusion matrix, classification report, ROC–AUC, kurva ROC).  

### Midterm-Regresi.ipynb

- Load dataset dengan kolom `tahun` dan `fitur_1`–`fitur_90` (total sekitar 91 kolom).  
- Preprocessing: imputasi dengan `SimpleImputer`, scaling menggunakan `StandardScaler`, dan pemisahan data train/test.  
- Ridge Regression dengan `GridSearchCV` untuk mencari hyperparameter `alpha` terbaik, dievaluasi dengan RMSE, MAE, dan R².[file:2]  
- Model regresi neural network menggunakan Keras/TensorFlow dengan beberapa dense layer dan EarlyStopping, serta visualisasi training history.[file:2]  
