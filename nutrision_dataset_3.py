# nutrision_dataset_3.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Food Calorie Predictor - Prediksi Kalori Makanan",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center;">Food Calorie Predictor</h1>
    <p style="color: white; text-align: center; font-size: 1.2rem;">
        Prediksi Kalori Makanan Berdasarkan Kandungan Makronutrien dan Usia
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Input Data Makanan")
    st.markdown("---")
    
    # Input fields
    proteins = st.number_input(
        "Protein (gram)", 
        min_value=0.0, 
        max_value=150.0, 
        value=10.0,
        step=0.5,
        help="Kandungan protein dalam gram (4 kkal/gram)"
    )
    
    fat = st.number_input(
        "Lemak (gram)", 
        min_value=0.0, 
        max_value=150.0, 
        value=15.0,
        step=0.5,
        help="Kandungan lemak dalam gram (9 kkal/gram)"
    )
    
    carbohydrate = st.number_input(
        "Karbohidrat (gram)", 
        min_value=0.0, 
        max_value=300.0, 
        value=30.0,
        step=0.5,
        help="Kandungan karbohidrat dalam gram (4 kkal/gram)"
    )
    
    age = st.number_input(
        "Usia (tahun)", 
        min_value=18, 
        max_value=100, 
        value=35,
        step=1,
        help="Usia pengguna"
    )
    
    st.markdown("---")
    st.markdown("### Informasi")
    st.info("""
    **Konversi Kalori Standar:**
    - Protein: 4 kkal/gram
    - Karbohidrat: 4 kkal/gram  
    - Lemak: 9 kkal/gram
    """)

# Load model
@st.cache_resource
def load_models():
    """Load model dan scaler yang sudah dilatih"""
    try:
        model = joblib.load('best_calorie_nutrision_food_model.pkl')
        scaler = joblib.load('scaler_food.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model tidak ditemukan!")
        st.info("Pastikan file 'best_calorie_nutrision_food_model.pkl' dan 'scaler_food.pkl' berada di folder yang sama dengan aplikasi.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Fungsi prediksi
def predict_calories(proteins, fat, carbohydrate, age, model, scaler):
    """Melakukan prediksi kalori berdasarkan input"""
    # Feature engineering
    protein_fat = proteins * fat
    carb_age = carbohydrate * age
    
    # Buat array fitur (urutan harus sama dengan training)
    features = np.array([[proteins, fat, carbohydrate, age, protein_fat, carb_age]])
    
    # Normalisasi
    features_scaled = scaler.transform(features)
    
    # Prediksi
    prediction = model.predict(features_scaled)[0]
    
    return prediction, protein_fat, carb_age

# Load models
model, scaler = load_models()

# Tampilkan input summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Protein", f"{proteins} g", delta=None)
with col2:
    st.metric("Lemak", f"{fat} g", delta=None)
with col3:
    st.metric("Karbohidrat", f"{carbohydrate} g", delta=None)
with col4:
    st.metric("Usia", f"{age} tahun", delta=None)

# Tombol prediksi
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
with col_button2:
    predict_button = st.button("PREDIKSI KALORI", type="primary", use_container_width=True)

if predict_button:
    if model is not None and scaler is not None:
        # Lakukan prediksi
        prediction, protein_fat, carb_age = predict_calories(
            proteins, fat, carbohydrate, age, model, scaler
        )
        
        # Tampilkan hasil prediksi
        st.markdown("---")
        st.markdown("## Hasil Prediksi")
        
        # Card prediksi
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        with col_result2:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color: white; margin: 0;">Total Kalori</h2>
                <h1 style="color: white; font-size: 80px; margin: 0;">{prediction:.0f}</h1>
                <p style="color: white; font-size: 20px;">kkal</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Kategori makanan
        if prediction <= 100:
            category = "Rendah Kalori"
            category_desc = "Cocok untuk diet atau camilan ringan"
            color = "#4CAF50"
        elif prediction <= 300:
            category = "Sedang"
            category_desc = "Porsi normal untuk makanan sehari-hari"
            color = "#FFC107"
        elif prediction <= 500:
            category = "Tinggi Kalori"
            category_desc = "Makanan berat, cukup untuk makan siang/malam"
            color = "#FF9800"
        else:
            category = "Sangat Tinggi Kalori"
            category_desc = "Makanan cepat saji atau porsi besar"
            color = "#F44336"
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h3 style="color: white; margin: 0;">{category}</h3>
            <p style="color: white; margin: 0;">{category_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Kontribusi kalori per makronutrien
        st.markdown("---")
        st.markdown("## Analisis Kontribusi Kalori")
        
        # Hitung kontribusi manual
        calorie_proteins = proteins * 4
        calorie_fat = fat * 9
        calorie_carb = carbohydrate * 4
        total_manual = calorie_proteins + calorie_fat + calorie_carb
        
        # Data untuk visualisasi
        contribution_data = pd.DataFrame({
            'Makronutrien': ['Protein', 'Lemak', 'Karbohidrat'],
            'Kalori (kkal)': [calorie_proteins, calorie_fat, calorie_carb],
            'Persentase (%)': [
                (calorie_proteins / total_manual * 100) if total_manual > 0 else 0,
                (calorie_fat / total_manual * 100) if total_manual > 0 else 0,
                (calorie_carb / total_manual * 100) if total_manual > 0 else 0
            ]
        })
        
        # Bar chart
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_bar = px.bar(
                contribution_data, 
                x='Makronutrien', 
                y='Kalori (kkal)',
                color='Makronutrien',
                title='Kontribusi Kalori per Makronutrien',
                text='Kalori (kkal)',
                color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3']
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_chart2:
            fig_pie = px.pie(
                contribution_data,
                values='Kalori (kkal)',
                names='Makronutrien',
                title='Persentase Kontribusi Kalori',
                color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Metrik tambahan
        st.markdown("---")
        st.markdown("## Detail Perhitungan")
        
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        with col_metric1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Kalori dari Protein</h4>
                <h2>{calorie_proteins:.0f} kkal</h2>
                <p>{contribution_data.iloc[0]['Persentase (%)']:.1f}% dari total</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_metric2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Kalori dari Lemak</h4>
                <h2>{calorie_fat:.0f} kkal</h2>
                <p>{contribution_data.iloc[1]['Persentase (%)']:.1f}% dari total</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_metric3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Kalori dari Karbohidrat</h4>
                <h2>{calorie_carb:.0f} kkal</h2>
                <p>{contribution_data.iloc[2]['Persentase (%)']:.1f}% dari total</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_metric4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Kalori</h4>
                <h2>{total_manual:.0f} kkal</h2>
                <p>Dari perhitungan manual</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediksi vs Manual
        st.markdown("---")
        st.markdown("## Prediksi Model vs Perhitungan Manual")
        
        comparison_df = pd.DataFrame({
            'Metode': ['Perhitungan Manual', 'Prediksi Model ML'],
            'Kalori (kkal)': [total_manual, prediction]
        })
        
        fig_compare = px.bar(
            comparison_df,
            x='Metode',
            y='Kalori (kkal)',
            color='Metode',
            title='Perbandingan Hasil',
            text='Kalori (kkal)',
            color_discrete_sequence=['#607D8B', '#9C27B0']
        )
        fig_compare.update_traces(textposition='outside')
        fig_compare.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Selisih
        diff = abs(prediction - total_manual)
        diff_percent = (diff / total_manual * 100) if total_manual > 0 else 0
        st.info(f"Selisih antara prediksi model dan perhitungan manual: {diff:.0f} kkal ({diff_percent:.1f}%)")

# Informasi model
with st.expander("Informasi Model Machine Learning", expanded=False):
    st.markdown("""
    ### Tentang Model
    
    Model yang digunakan adalah **Random Forest Regressor** yang telah di-tuning untuk prediksi kalori makanan.
    
    #### Parameter Model Terbaik:
    | Parameter | Nilai |
    |-----------|-------|
    | n_estimators | 300 |
    | min_samples_split | 5 |
    | min_samples_leaf | 2 |
    | max_depth | None (unlimited) |
    
    #### Metrik Kinerja Model:
    | Metrik | Nilai | Keterangan |
    |--------|-------|-------------|
    | MAE | 20.71 kkal | Rata-rata error absolut |
    | RMSE | 60.17 kkal | Akar rata-rata kuadrat error |
    | R-squared | 0.86 | Koefisien determinasi |
    | MAPE | 14.65% | Rata-rata persentase error absolut |
    
    #### Fitur yang Digunakan:
    1. **Proteins** - Kandungan protein (gram)
    2. **Fat** - Kandungan lemak (gram)
    3. **Carbohydrate** - Kandungan karbohidrat (gram)
    4. **Age** - Usia pengguna (tahun)
    5. **protein_fat** - Interaksi protein x lemak
    6. **carb_age** - Interaksi karbohidrat x usia
    
    #### Feature Importance:
    - Lemak (fat): 52.3% - Paling dominan
    - Karbohidrat (carbohydrate): 34.2%
    - Interaksi protein x lemak: 9.7%
    - Protein: 2.5%
    - Lainnya: 1.3%
    
    ### Batasan Model
    - Model dilatih dengan rentang data: 0-940 kkal
    - Prediksi di luar rentang ini mungkin kurang akurat
    - Model lebih akurat untuk makanan dengan makronutrien seimbang
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: gray;">
        Copyright 2024 Food Calorie Predictor | Dibangun dengan Streamlit dan Machine Learning
    </p>
    <p style="color: gray; font-size: 0.8rem;">
        Aplikasi ini menggunakan model Random Forest dengan akurasi R-squared = 0.86
    </p>
</div>
""", unsafe_allow_html=True)
