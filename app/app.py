import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# -----------------------------------------------------------------------------
# pagina
st.set_page_config(
    page_title="Quantum Data Force | UPTC",
    page_icon="⚡",
    layout="wide"
)

st.title(" Monitor y Simulador Energético - UPTC")
st.markdown("**Equipo:** Quantum Data Force | IA Minds 2026")
st.markdown("---")

# -----------------------------------------------------------------------------
# cargar datos
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '../datos/consumos_uptc.zip')
    try:
        df = pd.read_csv(file_path, compression='zip')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Limpieza 
        df['co2_kg'] = df['co2_kg'].fillna(0)
        df['ocupacion_pct'] = df['ocupacion_pct'].fillna(df['ocupacion_pct'].mean())
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

@st.cache_resource
def load_trained_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, '../modelos/modelo_energia_v3.pkl')
    try:
        return joblib.load(model_path)
    except:
        return None

df = load_data()
model = load_trained_model()

if df is not None:
    # -----------------------------------------------------------------------------
    # visualización
    st.sidebar.header(" Filtros Históricos")
    sede_selec = st.sidebar.selectbox(" Selecciona Sede:", df['sede'].unique())
    
    # Filtrar por sede
    df_sede = df[df['sede'] == sede_selec]
    
    # KPIs 
    c1, c2, c3 = st.columns(3)
    c1.metric("Consumo Total Histórico", f"{df_sede['energia_total_kwh'].sum():,.0f} kWh")
    c2.metric("Huella de CO2 Total", f"{df_sede['co2_kg'].sum():,.0f} kg")
    c3.metric("Ocupación Promedio", f"{df_sede['ocupacion_pct'].mean():.1f}%")

    # Gráfica de tendencia
    st.subheader(f" Tendencia de Consumo en {sede_selec}")
    fig_line = px.line(df_sede.tail(1000), x='timestamp', y='energia_total_kwh', 
                       title="Últimos registros detectados", color_discrete_sequence=['#2E86C1'])
    st.plotly_chart(fig_line, use_container_width=True)

    # -----------------------------------------------------------------------------
    # predicción
    st.markdown("---")
    st.header(" Simulador de Proyección IA")
    st.info("Ajusta los parámetros para predecir el comportamiento energético por sede y sector.")

    if model is not None:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Configuración de Escenario")
            sede_p = st.selectbox(" Sede para Predicción", df['sede'].unique(), key="s_pred")
            
            sector_p = st.selectbox("Sector", ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"])
            
        with col_p2:
            st.subheader("Variables Ambientales")
            occ_f = st.slider("Nivel de Ocupación (%)", 0, 100, 50)
            temp_f = st.slider("Temperatura Exterior (°C)", 5, 35, 18)

        # procesamiento
        sede_idx = list(df['sede'].unique()).index(sede_p)
        sector_idx = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"].index(sector_p)
        
        # Predicción para un ciclo de 24 horas
        horas = list(range(24))
        preds_24h = []
        
        for h in horas:
           
            input_data = pd.DataFrame([[h, 1, 10, sede_idx, sector_idx, occ_f, temp_f]], 
                                     columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
            preds_24h.append(model.predict(input_data)[0])

        # resultados
        st.subheader(f" Curva de Carga Predicha para {sector_p} ({sede_p})")
        
        fig_pred = px.area(x=horas, y=preds_24h, 
                          labels={'x': 'Hora del día', 'y': 'Consumo Predicho (kWh)'},
                          color_discrete_sequence=['#F39C12'],
                          template="plotly_white")
        
        
        promedio_pred = sum(preds_24h) / len(preds_24h)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.write(f" **Insight IA:** El consumo promedio esperado para este escenario es de **{promedio_pred:.2f} kWh**.")

    else:
        st.warning(" El archivo 'modelo_energia_v3.pkl' no se encuentra en la carpeta /modelos. Súbelo para activar las predicciones.")

else:
    st.error("No se pudo cargar la base de datos.")
