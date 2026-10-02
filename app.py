import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Bunga Iris")


@st.cache_resource
def latih_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    akurasi = accuracy_score(y_test, model.predict(X_test))
    return model, iris, akurasi


model, iris, akurasi = latih_model()

st.title("Prediksi Jenis Bunga Iris")
st.write("Geser slider di sebelah kiri untuk mengatur ukuran bunga, lalu lihat tebakan model.")
st.caption(f"Model: Random Forest | Akurasi pada data uji: {akurasi:.1%}")

st.sidebar.header("Ukuran bunga (cm)")
nilai = []
for i, nama in enumerate(iris.feature_names):
    kolom = iris.data[:, i]
    nilai.append(
        st.sidebar.slider(
            nama,
            min_value=float(kolom.min()),
            max_value=float(kolom.max()),
            value=float(kolom.mean()),
            step=0.1,
        )
    )

prediksi = model.predict([nilai])[0]
probabilitas = model.predict_proba([nilai])[0]

st.subheader("Hasil tebakan")
st.success(f"Jenis bunga: **{iris.target_names[prediksi]}**")

st.subheader("Tingkat keyakinan model")
df_prob = pd.DataFrame({"Jenis": iris.target_names, "Probabilitas": probabilitas}).set_index("Jenis")
st.bar_chart(df_prob)

with st.expander("Lihat data input"):
    st.write(pd.DataFrame([nilai], columns=iris.feature_names))
