import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 1. CONFIGURACIÓN DE LA PÁGINA 
st.set_page_config(page_title="Quantum Data Force", layout="wide", page_icon="⚡")

st.title("⚡ Monitor Energético UPTC - IA Minds 2026")
st.markdown("**Estado del Sistema:** En línea | **Fuente de Datos:** Repositorio Seguro")

# cargar zip a csv
@st.cache_data
def cargar_datos():
    ruta_zip = os.path.join(os.path.dirname(__file__), '../datos/consumos_uptc.zip')

    # limpieza de datos
    
    try:
        df = pd.read_csv(ruta_zip, compression='zip')
        
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Texto a Fecha
        df = df[df['energia_total_kwh'] >= 0]             # Borrar negativos
        df['co2_kg'] = df['co2_kg'].fillna(0)             # Rellenar vacíos
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# Cargar los datos
df = cargar_datos()

if not df.empty:
    
    # Filtros
    st.sidebar.header("Filtros")
    sede = st.sidebar.selectbox("Selecciona Sede:", df['sede'].unique())
    
    # Filtrar datos
    df_sede = df[df['sede'] == sede]
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Energía (Histórico)", f"{df_sede['energia_total_kwh'].sum():,.0f} kWh")
    kpi2.metric("Pico Máximo Detectado", f"{df_sede['energia_total_kwh'].max():.2f} kWh")
    kpi3.metric("Registros Analizados", f"{len(df_sede)}")


    
    ## Gráfica Principal
    #st.subheader(f"Comportamiento Energético: {sede}")
    ## Tse toma el ultimo mes
    #ultimo_mes = df_sede[df_sede['timestamp'] > df_sede['timestamp'].max() - pd.Timedelta(days=30)]
    
    #fig = px.line(ultimo_mes, x='timestamp', y='energia_total_kwh', 
    #              title="Últimos 30 días de consumo", color_discrete_sequence=['#00CC96'])
    #st.plotly_chart(fig, use_container_width=True)

    #st.success("¡Conexión exitosa! El archivo ZIP se leyó correctamente.")

#else:
    #st.warning("Esperando conexión con los datos...")

    # GRÁFICA app
    st.subheader(f"Análisis Temporal: {sede}")

    # para seleccionar fechas
    with st.sidebar:
        st.markdown("---")
        st.header("📅 Rango de Análisis")
        fecha_inicio = st.date_input("Fecha Inicio", df_sede['timestamp'].min())
        fecha_fin = st.date_input("Fecha Fin", df_sede['timestamp'].max())

    # filtro de datos
    mask = (df_sede['timestamp'].dt.date >= fecha_inicio) & (df_sede['timestamp'].dt.date <= fecha_fin)
    df_seleccionado = df_sede.loc[mask]

    # gráfica datos filtrados
    if not df_seleccionado.empty:
        fig = px.line(df_seleccionado, x='timestamp', y='energia_total_kwh', 
                      title=f"Consumo desde {fecha_inicio} hasta {fecha_fin}",
                      color_discrete_sequence=['#00CC96'],
                      labels={'energia_total_kwh': 'Consumo (kWh)', 'timestamp': 'Fecha'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos para el rango seleccionado.")
