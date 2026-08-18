import streamlit as st
import pandas as pd
import joblib
import base64
from preprocessing import preprocess

# CONFIG
st.set_page_config(
    page_title="Prediksi Probabilitas Kelulusan SNBP",
    layout="wide"
)

# CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# HEADER dengan logo dan judul
# HEADER
with open("logo MA2.png", "rb") as f:
    logo = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo}" width="150">
        <h1 style="margin-top: 0.2rem;">
            Prediksi Probabilitas Kelulusan SNBP
        </h1>
        <p style="
            color: #2e7d32;
            font-weight: 500;
            margin-top: -0.5rem;
        ">
            MA Manaratul Islam
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
# LOAD MODEL
model = joblib.load("snbp_proba_model.pkl")

# UPLOAD
uploaded_file = st.file_uploader(
    "Upload File Excel (format .xlsx)",
    type=["xlsx"]
)

# PREDIKSI
if uploaded_file is not None:
    raw_df = pd.read_excel(uploaded_file)
    st.subheader("Preview Dataset")
    st.dataframe(raw_df.head())
    # simpan nama sebelum preprocessing
    nama_siswa = raw_df["nama siswa"].copy()
    # preprocessing
    X = preprocess(raw_df.copy())
    # prediksi
    pred = model.predict(X)
    prob = model.predict_proba(X)
    # dataframe hasil
    hasil = pd.DataFrame({
        "Nama": nama_siswa,
        "Status": [
            "Lolos" if p == 1 else "Tidak Lolos"
            for p in pred
        ],
        "Probabilitas": prob[:,1]
    })

    st.success("Prediksi berhasil dilakukan.")
    st.subheader("Hasil Prediksi")
    st.dataframe(hasil)
    st.divider()

    # PILIH SISWA
    st.subheader("Detail Prediksi per Siswa")
    nama = st.selectbox(
        "Pilih Nama Siswa",
        hasil["Nama"]
    )

    data = hasil[
        hasil["Nama"] == nama
    ].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Status",
            data["Status"]
        )
    with col2:
        st.metric(
            "Probabilitas Kelulusan",
            f"{data['Probabilitas']*100:.2f}%"
        )

    st.progress(float(data["Probabilitas"]))

    # Footer
    st.write("---")
    st.markdown("<p style='text-align: center; color: #94a3b8;'>© 2026 MA Manaratul Islam - Prediksi SNBP</p>", unsafe_allow_html=True)