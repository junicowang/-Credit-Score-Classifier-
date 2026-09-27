"""
app.py — Streamlit web app untuk deployment Credit Score Classification.

Jalankan dengan:
    streamlit run app.py

App ini:
- Menyediakan form input 12 fitur terpenting (sisanya diimputasi otomatis)
- Memuat model via CreditScorePredictor (predictor.py)
- Menampilkan prediksi + bar chart probabilitas per kelas
- Punya 3 tombol contoh (Good/Poor/Standard) untuk memudahkan testing/screenshot
"""

import streamlit as st
import pandas as pd
from predictor import CreditScorePredictor

# ===== Konfigurasi halaman =====
st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="centered")


# ===== Load model (cache supaya tidak reload tiap interaksi) =====
@st.cache_resource
def load_predictor():
    return CreditScorePredictor("model.pkl")


predictor = load_predictor()


# ===== 3 profil contoh untuk auto-fill (memudahkan screenshot test case) =====
EXAMPLES = {
    "Good": {
        'Age': 40, 'Annual_Income': 120000, 'Outstanding_Debt': 100,
        'Interest_Rate': 4, 'Delay_from_due_date': 1, 'Num_of_Delayed_Payment': 0,
        'Num_Credit_Card': 2, 'Num_Credit_Inquiries': 0, 'Changed_Credit_Limit': 3,
        'Monthly_Balance': 800, 'Credit_Mix': 'Good', 'Credit_History_Age': 25,
    },
    "Standard": {
        'Age': 32, 'Annual_Income': 45000, 'Outstanding_Debt': 1200,
        'Interest_Rate': 14, 'Delay_from_due_date': 12, 'Num_of_Delayed_Payment': 8,
        'Num_Credit_Card': 5, 'Num_Credit_Inquiries': 5, 'Changed_Credit_Limit': 10,
        'Monthly_Balance': 330, 'Credit_Mix': 'Standard', 'Credit_History_Age': 14,
    },
    "Poor": {
        'Age': 25, 'Annual_Income': 15000, 'Outstanding_Debt': 4500,
        'Interest_Rate': 34, 'Delay_from_due_date': 45, 'Num_of_Delayed_Payment': 20,
        'Num_Credit_Card': 9, 'Num_Credit_Inquiries': 12, 'Changed_Credit_Limit': 25,
        'Monthly_Balance': 50, 'Credit_Mix': 'Bad', 'Credit_History_Age': 2,
    },
}

# Inisialisasi session_state dengan nilai default (profil Standard)
if "form_values" not in st.session_state:
    st.session_state.form_values = EXAMPLES["Standard"].copy()


def fill_example(label):
    """Callback tombol contoh: isi form dengan profil terpilih."""
    st.session_state.form_values = EXAMPLES[label].copy()


# ===== Header =====
st.title("💳 Credit Score Classifier")
st.markdown(
    f"Prediksi performa kredit nasabah menggunakan model **{predictor.model_name}**. "
    "Isi 12 fitur utama di bawah; fitur lain akan diisi otomatis dengan nilai median."
)

# ===== Tombol contoh =====
st.subheader("Isi otomatis contoh test case")
c1, c2, c3 = st.columns(3)
c1.button("🟢 Contoh Good", on_click=fill_example, args=("Good",), use_container_width=True)
c2.button("🟡 Contoh Standard", on_click=fill_example, args=("Standard",), use_container_width=True)
c3.button("🔴 Contoh Poor", on_click=fill_example, args=("Poor",), use_container_width=True)

st.divider()

# ===== Form input =====
fv = st.session_state.form_values
st.subheader("Data Nasabah")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur (tahun)", min_value=18, max_value=100,
                          value=int(fv['Age']), step=1)
    annual_income = st.number_input("Penghasilan Tahunan ($)", min_value=0.0,
                                    value=float(fv['Annual_Income']), step=1000.0)
    outstanding_debt = st.number_input("Sisa Utang ($)", min_value=0.0,
                                       value=float(fv['Outstanding_Debt']), step=50.0)
    interest_rate = st.number_input("Suku Bunga (%)", min_value=0, max_value=50,
                                    value=int(fv['Interest_Rate']), step=1)
    delay_due = st.number_input("Keterlambatan dari Jatuh Tempo (hari)", min_value=0,
                                value=int(fv['Delay_from_due_date']), step=1)
    num_delayed = st.number_input("Jumlah Pembayaran Terlambat", min_value=0,
                                  value=int(fv['Num_of_Delayed_Payment']), step=1)

with col2:
    num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=20,
                                      value=int(fv['Num_Credit_Card']), step=1)
    num_inquiries = st.number_input("Jumlah Inquiry Kredit", min_value=0, max_value=50,
                                    value=int(fv['Num_Credit_Inquiries']), step=1)
    changed_limit = st.number_input("Perubahan Limit Kredit", value=float(fv['Changed_Credit_Limit']),
                                    step=1.0)
    monthly_balance = st.number_input("Saldo Bulanan ($)", value=float(fv['Monthly_Balance']),
                                      step=50.0)
    credit_mix = st.selectbox("Credit Mix", options=["Bad", "Standard", "Good"],
                              index=["Bad", "Standard", "Good"].index(fv['Credit_Mix']))
    credit_hist = st.number_input("Lama Riwayat Kredit (tahun)", min_value=0, max_value=50,
                                  value=int(fv['Credit_History_Age']), step=1)

st.divider()

# ===== Tombol prediksi =====
if st.button("🔍 Prediksi Credit Score", type="primary", use_container_width=True):
    user_input = {
        'Age': age, 'Annual_Income': annual_income, 'Outstanding_Debt': outstanding_debt,
        'Interest_Rate': interest_rate, 'Delay_from_due_date': delay_due,
        'Num_of_Delayed_Payment': num_delayed, 'Num_Credit_Card': num_credit_card,
        'Num_Credit_Inquiries': num_inquiries, 'Changed_Credit_Limit': changed_limit,
        'Monthly_Balance': monthly_balance, 'Credit_Mix': credit_mix,
        'Credit_History_Age': credit_hist,
    }

    result = predictor.predict(user_input)
    prediction = result['prediction']
    probs = result['probabilities']

    # Tampilkan prediksi dengan warna
    color_map = {"Good": "🟢", "Standard": "🟡", "Poor": "🔴"}
    st.subheader("Hasil Prediksi")
    st.markdown(f"## {color_map.get(prediction, '')} **{prediction}**")

    # Bar chart probabilitas
    st.markdown("**Probabilitas per kelas:**")
    prob_df = pd.DataFrame({
        "Kelas": list(probs.keys()),
        "Probabilitas": list(probs.values())
    }).set_index("Kelas")
    st.bar_chart(prob_df, height=300)

    # Detail angka
    cols = st.columns(len(probs))
    for i, (cls, p) in enumerate(probs.items()):
        cols[i].metric(label=cls, value=f"{p:.1%}")

# ===== Footer =====
st.divider()
st.caption("Model: RandomForest | Final Project Model Deployment | Dataset A")
