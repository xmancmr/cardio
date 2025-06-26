import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(layout="wide", page_title="Analyse Cardiaque")
st.title("📊 Dashboard d'Analyse Médicale")

# Chargement des données (à adapter)
df = pd.read_csv('./data/clean_data.csv')

# Variables disponibles
num_vars = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
cat_vars = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

# Charger le fichier CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
# Appliquer le CSS
load_css("./css/dash.css")


with st.sidebar:
    st.header("**⚙️ Paramètres**")
    selected_num = st.multiselect("Variables numériques", num_vars, default=num_vars[:2])
    selected_cat = st.selectbox("Variable catégorielle", cat_vars)
    st.download_button("📥 Exporter les données", df.to_csv(), "heartdisease.csv")


plotly_template = "plotly_dark"

tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Corrélations", "🔄 Comparaisons"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme interactif
        st.subheader("Distribution des Variables Numériques")
        num_var = st.selectbox("Choisir une variable", selected_num, key="hist_var")
        fig = px.histogram(
            df, x=num_var, color=selected_cat, nbins=30,
            template=plotly_template,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={num_var: f"{num_var} (unités)" if num_var is not None
                    else "Choisissez une variable dans la liste"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie Chart des catégories
        st.subheader(f"Répartition de {selected_cat}")
        fig = px.pie(
            df, names=selected_cat, template=plotly_template,
            hole=0.3, color_discrete_sequence=px.colors.sequential.Agsunset
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Matrice de corrélation
    st.subheader("Heatmap des Corrélations")
    corr = df[selected_num].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale='RdBu_r',
        template=plotly_template,
        labels=dict(x="Variable X", y="Variable Y", color="Corrélation")
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Boxplots comparatifs
    st.subheader("Analyse Comparative")
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Axe X", cat_vars, key="x_var")
    with col2:
        y_var = st.selectbox("Axe Y", selected_num, key="y_var")
    
    fig = px.box(
        df, x=x_var, y=y_var, color=x_var,
        template=plotly_template,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={y_var: f"{y_var} (unités)" if y_var is not None else "Choisissez une variable dans la liste"}
    )
    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.header("Analyses Spécifiques")

# Relation entre Âge et Tension Artérielle
expander1 = st.expander("Âge vs Pression Artérielle")
with expander1:
    fig = px.scatter(
        df, x='age', y='resting bp s', trendline="lowess",
        color='chest pain type', 
        labels={"age": "Âge (années)", "resting bp s": "Pression (mmHg)"},
        template=plotly_template
    )
    st.plotly_chart(fig)

# Radar Chart pour Profil Patient
expander2 = st.expander("Profil Médical Multidimensionnel")
with expander2:
    radar_vars = st.multiselect("Variables radar", num_vars, default=num_vars[:3])
    if radar_vars:
        fig = px.line_polar(
            df.sample(1), r=radar_vars, theta=radar_vars,
            line_close=True, template=plotly_template
        )
        st.plotly_chart(fig)

# ===================