"""
Klasifikasi Bunga Iris dengan Machine Learning (scikit-learn)

Alur:
1. Muat dataset
2. Bagi data latih dan data uji
3. Bandingkan beberapa model dengan cross-validation
4. Latih model terbaik dan evaluasi
5. Simpan model dan coba prediksi data baru

Instalasi: pip install scikit-learn joblib
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Muat dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"Jumlah data  : {X.shape[0]}")
    print(f"Jumlah fitur : {X.shape[1]} -> {iris.feature_names}")
    print(f"Kelas        : {list(iris.target_names)}\n")

    # 2. Bagi data: 80% latih, 20% uji (stratify menjaga proporsi kelas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Bandingkan beberapa model dengan 5-fold cross-validation
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    print("=== Perbandingan Model (5-fold CV) ===")
    best_name, best_score = None, 0.0
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name:22s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    print(f"\nModel terbaik: {best_name}\n")

    # 4. Latih model terbaik pada seluruh data latih, lalu uji
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("=== Evaluasi pada Data Uji ===")
    print(f"Akurasi: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 5. Simpan model, lalu muat kembali untuk prediksi data baru
    joblib.dump(best_model, "model_iris.joblib")
    print("Model disimpan ke model_iris.joblib")

    loaded = joblib.load("model_iris.joblib")
    sampel_baru = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    hasil = loaded.predict(sampel_baru)
    print("\n=== Prediksi Data Baru ===")
    for fitur, kelas in zip(sampel_baru, hasil):
        print(f"{fitur} -> {iris.target_names[kelas]}")


if __name__ == "__main__":
    main()
