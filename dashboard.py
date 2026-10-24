"""
dashboard.py
Panel telemétrico interactivo y Asistente IA para el Vivero CECAR.
Diseño ejecutivo profesional sin emojis y sincronización de zona horaria para Colombia (America/Bogota).
"""

import os
import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Cargar variables de entorno (compatible con Local y Streamlit Cloud)
load_dotenv()


def obtener_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)


DATABASE_URL = obtener_secret("DATABASE_URL")
GROQ_API_KEY = obtener_secret("GROQ_API_KEY")

# Configuración inicial de la página
st.set_page_config(
    page_title="Vivero CECAR - Sistema de Riego",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS Personalizados
CSS_PROFESIONAL = """
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main-header {
        background: linear-gradient(135deg, #0f382c 0%, #1a5c48 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .main-header p {
        color: #a7f3d0 !important;
        font-size: 0.95rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.1rem 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .status-card-active {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        color: #991b1b;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1.2rem 0;
    }
    .status-card-inactive {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        color: #166534;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1.2rem 0;
    }
    .status-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }
    .status-desc {
        font-size: 0.95rem;
        margin: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        color: #1a5c48 !important;
        border-bottom: 3px solid #1a5c48 !important;
    }
</style>
"""

st.markdown(CSS_PROFESIONAL, unsafe_allow_html=True)


@st.cache_data(ttl=1)
def cargar_datos():
    """Consulta Neon y ajusta la hora de UTC a Colombia (America/Bogota)."""
    if not DATABASE_URL:
        st.error("DATABASE_URL no está configurada.")
        return pd.DataFrame()

    try:
        conn = psycopg2.connect(DATABASE_URL)
        query = "SELECT * FROM telemetria_diaria ORDER BY fecha_registro DESC LIMIT 100;"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty and "fecha_registro" in df.columns:
            df["fecha_registro"] = pd.to_datetime(df["fecha_registro"])

            if df["fecha_registro"].dt.tz is None:
                df["fecha_registro"] = (
                    df["fecha_registro"]
                    .dt.tz_localize("UTC")
                    .dt.tz_convert("America/Bogota")
                )
            else:
                df["fecha_registro"] = df["fecha_registro"].dt.tz_convert(
                    "America/Bogota"
                )

            df["fecha_registro"] = df["fecha_registro"].dt.tz_localize(None)

        return df
    except Exception as e:
        st.error(f"Error al conectar con la base de datos Neon: {e}")
        return pd.DataFrame()


# --- BARRA LATERAL ---
st.sidebar.markdown("### Configuración del Asistente IA")

MODELOS_DISPONIBLES = {
    "Llama 3.3 70B (Alta Precisión)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Respuesta Rápida)": "llama-3.1-8b-instant",
    "Mixtral 8x7B (Análisis Contextual)": "mixtral-8x7b-32768",
    "Gemma 2 9B (Google)": "gemma2-9b-it",
}

nombre_modelo_sel = st.sidebar.selectbox(
    "Modelo de Lenguaje Activo:", list(MODELOS_DISPONIBLES.keys())
)
modelo_groq_id = MODELOS_DISPONIBLES[nombre_modelo_sel]

# --- ENCABEZADO ---
st.markdown(
    """
    <div class="main-header">
        <h1>Sistema de Riego Inteligente</h1>
        <p>Estación Telemétrica y Motor de Reglas Agrícolas — Vivero Universidad CECAR</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = cargar_datos()

# Pestañas principales
tab_monitoreo, tab_asistente = st.tabs(
    ["Monitoreo Telemétrico", "Asistente IA Agrónomo"]
)

# ---------------------------------------------------------
# TAB 1: MONITOREO TELEMÉTRICO
# ---------------------------------------------------------
with tab_monitoreo:
    if df.empty:
        st.info("Esperando recepción de datos telemétricos desde Neon...")
    else:
        ultima = df.iloc[0]
        fecha_str = ultima["fecha_registro"].strftime("%Y-%m-%d %H:%M:%S")

        st.caption(
            f"Última actualización de estación (Hora Colombia): **{fecha_str}**"
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Temperatura", f"{ultima['temperatura_c']} °C")
        col2.metric("Precipitación", f"{ultima['precipitacion_mm']} mm")
        col3.metric("Humedad Relativa", f"{ultima['humedad_relativa_pct']}%")
        col4.metric("Radiación Solar", f"{ultima['radiacion_solar_wm2']} W/m²")
        col5.metric("Velocidad Viento", f"{ultima['rafaga_viento_kmh']} km/h")

        if ultima["orden_riego"]:
            st.markdown(
                f"""
                <div class="status-card-active">
                    <div class="status-title">ORDEN DE RIEGO ACTIVA</div>
                    <div class="status-desc">Regla Aplicada: <b>{ultima['regla_aplicada']}</b> — {ultima['mensaje']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-card-inactive">
                    <div class="status-title">SISTEMA EN REPOSO — RIEGO DESACTIVADO</div>
                    <div class="status-desc">Regla Aplicada: <b>{ultima['regla_aplicada']}</b> — {ultima['mensaje']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Histórico de Telemetría")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_temp = px.line(
                df,
                x="fecha_registro",
                y="temperatura_c",
                title="Temperatura (°C)",
                labels={
                    "fecha_registro": "Fecha / Hora (Colombia)",
                    "temperatura_c": "Temp (°C)",
                },
                template="plotly_white",
            )
            fig_temp.update_traces(
                line_color="#1a5c48", line_width=2.5, mode="lines+markers"
            )
            fig_temp.update_layout(
                margin=dict(l=20, r=20, t=40, b=20), height=320
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with col_chart2:
            fig_precip = px.bar(
                df,
                x="fecha_registro",
                y="precipitacion_mm",
                title="Precipitación Acumulada (mm)",
                labels={
                    "fecha_registro": "Fecha / Hora (Colombia)",
                    "precipitacion_mm": "Precipitación (mm)",
                },
                template="plotly_white",
            )
            fig_precip.update_traces(marker_color="#2563eb")
            fig_precip.update_layout(
                margin=dict(l=20, r=20, t=40, b=20), height=320
            )
            st.plotly_chart(fig_precip, use_container_width=True)

        with st.expander("Consultar registros históricos detallados"):
            st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: ASISTENTE IA AGRÓNOMO (MULTI-MODELO)
# ---------------------------------------------------------
with tab_asistente:
    st.subheader("Asistente Virtual Agrónomo")
    st.caption(f"Modelo de IA activo: **{modelo_groq_id}**")

    if not GROQ_API_KEY:
        st.warning(
            "La variable GROQ_API_KEY no se encuentra configurada en el entorno."
        )
    else:
        client = Groq(api_key=GROQ_API_KEY)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Escriba su consulta agrónoma..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            contexto_clima = "Sin datos disponibles."
            if not df.empty:
                u = df.iloc[0]
                contexto_clima = (
                    f"Lectura ({u['fecha_registro']}): "
                    f"Temperatura: {u['temperatura_c']}°C, Precipitación: {u['precipitacion_mm']}mm, "
                    f"Humedad Relativa: {u['humedad_relativa_pct']}%, Radiación Solar: {u['radiacion_solar_wm2']}W/m², "
                    f"Estado Riego: {u['orden_riego']}, Regla: {u['regla_aplicada']}, Diagnóstico: {u['mensaje']}."
                )

            system_prompt = (
                "Eres un ingeniero agrónomo consultor experto del Vivero CECAR. "
                f"Información telemétrica actual del vivero en hora local de Colombia: {contexto_clima}. "
                "Responde en un lenguaje claro, técnico, profesional y estrictamente sin utilizar ningún emoji en tu texto."
            )

            messages_for_api = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            with st.chat_message("assistant"):
                try:
                    completion = client.chat.completions.create(
                        model=modelo_groq_id,
                        messages=messages_for_api,
                        temperature=0.6,
                        max_tokens=800,
                    )
                    respuesta = completion.choices[0].message.content
                    st.markdown(respuesta)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": respuesta}
                    )
                except Exception as e:
                    st.error(f"Error al procesar la solicitud con el modelo: {e}")