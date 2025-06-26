import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from PIL import Image
from scipy import stats

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de risque cardiaque - Multi-Modèles",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css('./css/pre.css')

# ---- 1. Chargement des modèles (version robuste) ----
@st.cache_resource
def load_models():
    """Charge tous les modèles avec vérifications"""
    try:
        # Vérification des fichiers avant chargement
        required_files = [
            './models/heart_model_pipeline_decision_tree.joblib',
            './models/heart_ann_model.h5',
            './models/ann_scaler.joblib',
            './models/feature_names_decision_tree.joblib'
        ]
        
        for file in required_files:
            try:
                with open(file, 'rb'):
                    pass
            except FileNotFoundError:
                st.error(f"Fichier manquant : {file}")
                return None

        # Chargement effectif
        dt_model = joblib.load(required_files[0])
        ann_model = tf.keras.models.load_model(required_files[1])
        scaler = joblib.load(required_files[2])
        feature_names = joblib.load(required_files[3])

        # Validation des dimensions
        if len(feature_names) != len(scaler.mean_):
            st.error("Incompatibilité : Nombre de features différent entre le scaler et les noms de colonnes")
            return None

        return {
            'dt': dt_model,
            'ann': ann_model,
            'scaler': scaler,
            'feature_names': feature_names
        }

    except Exception as e:
        st.error(f"ERREUR de chargement : {str(e)}")
        return None

# ---- 2. Interface Utilisateur améliorée ----
def show_input_form(feature_names):
    """Formulaire avec gestion correcte des onglets"""
    tab1, tab2, tab3 = st.tabs(["📋 Prédiction", "📊 Analyse", "ℹ️ Aide"])
    input_df = None  # Initialisation

    # Tab1 - Formulaire de prédiction
    with tab1:
        with st.form("cardiac_risk_form", clear_on_submit=False):
            cols = st.columns(2)
            
            with cols[0]:
                st.subheader("Informations démographiques")
                age = st.slider("Âge", 1, 120, 50, help="Âge en années")
                sex = st.radio("Sexe", ["Femme", "Homme"], index=1)
                
                st.subheader("Paramètres cliniques")
                trestbps = st.slider("Pression artérielle (mm Hg)", 90, 200, 120)
                chol = st.slider("Cholestérol (mg/dl)", 100, 600, 200)
                thalach = st.slider("Fréquence cardiaque maximale", 60, 220, 150, 
                                   help="Fréquence cardiaque maximale atteinte (bpm)")

            with cols[1]:
                st.subheader("Symptômes et tests")
                cp_options = {
                    0: "Typique angine",
                    1: "Douleur atypique", 
                    2: "Douleur non-angineuse",
                    3: "Asymptomatique"
                }
                cp = st.selectbox("Type de douleur thoracique", 
                                 options=list(cp_options.keys()), 
                                 format_func=lambda x: cp_options[x])
                
                fbs = st.checkbox("Glycémie à jeun > 120 mg/dl")
                exang = st.checkbox("Angine induite par l'effort")
                oldpeak = st.slider("Dépression ST (mm)", 0.0, 10.0, 1.0, 0.1)

            # Deuxième ligne de paramètres
            cols2 = st.columns(2)
            with cols2[0]:
                st.subheader("Résultats ECG")
                restecg_options = {
                    0: "Normal", 
                    1: "Anomalie onde ST-T",
                    2: "Hypertrophie ventriculaire"
                }
                restecg = st.selectbox("ECG au repos", 
                                      options=list(restecg_options.keys()),
                                      format_func=lambda x: restecg_options[x])

            with cols2[1]:
                st.subheader("Paramètres avancés")
                slope_options = {
                    0: "Descendante",
                    1: "Plate",
                    2: "Ascendante"
                }
                slope = st.selectbox("Pente du segment ST", 
                                   options=list(slope_options.keys()),
                                   format_func=lambda x: slope_options[x])

            submitted = st.form_submit_button("🤖 Analyser le risque")

            if submitted:
                # Validation des entrées
                if chol < 100 or chol > 600:
                    st.error("Valeur de cholestérol invalide (doit être entre 100 et 600)")
                else:        
                    # Construction du DataFrame avec ordre garanti
                    input_data = {
                        'age': age,
                        'sex': 1 if sex == "Homme" else 0,
                        'chest pain type': cp,
                        'resting bp s': trestbps,
                        'cholesterol': chol,
                        'max heart rate': thalach,
                        'fasting blood sugar': 1 if fbs else 0,
                        'resting ecg': restecg,
                        'exercise angina': 1 if exang else 0,
                        'oldpeak': oldpeak,
                        'ST slope': slope
                    }
                    input_df = pd.DataFrame([input_data], columns=feature_names)
                    st.session_state.input_data = input_df  # Sauvegarde pour debug

    # Tab2 - Analyse (toujours visible)
    with tab2:
        st.subheader("Analyse exploratoire des données")
        try:
            data = pd.read_csv("./data/clean_data.csv")
            
            with st.expander("Statistiques descriptives"):
                st.dataframe(data.describe())

            # Visualisations
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(8, 4))
                sns.histplot(data['age'], bins=20, kde=True)
                plt.title('Distribution des âges')
                st.pyplot(fig)
                
            with col2:
                fig = plt.figure(figsize=(8, 4))
                sns.countplot(x='sex', data=data)
                plt.title('Répartition par sexe (0=Femme, 1=Homme)')
                st.pyplot(fig)
            
            # Corrélations
            st.markdown("### 🔗 Matrice de corrélation")
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, fmt=".1f", cmap="coolwarm")
            plt.title('Corrélation entre les variables')
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Impossible de charger les données d'analyse : {e}")

    # Tab3 - Aide (toujours visible)
    with tab3:
        st.subheader("Guide d'utilisation")
        st.markdown("""
        **Comment utiliser l'application ?**
        1. Remplissez tous les champs du formulaire de prédiction
        2. Cliquez sur le bouton "Analyser le risque cardiaque"
        3. Consultez les résultats et recommandations
        """)
        
        st.markdown("---")
        st.subheader("Explications des paramètres")
        with st.expander("Détails des variables médicales"):
            st.markdown("""
            - **Âge**: Âge du patient en années
            - **Sexe**: 0 = Femme, 1 = Homme
            - **Type de douleur thoracique**: 
              - 0: Angine typique
              - 1: Douleur atypique
              - 2: Douleur non-angineuse
              - 3: Asymptomatique
            - **Pression artérielle**: Pression au repos en mm Hg
            - **Cholestérol**: Niveau en mg/dl
            - **Fréquence cardiaque max**: Fréquence cardiaque maximale atteinte (bpm)
            """)

    return input_df  # Retourne None si non soumis, sinon le DataFrame

# ---- 3. Affichage des résultats (version robuste) ----
def display_results(prediction, proba, model_type):
    """Affiche les résultats avec gestion des erreurs"""
    try:
        # Validation des probabilités
        if not isinstance(proba, (list, np.ndarray)) or len(proba) != 2:
            raise ValueError("Format de probabilité invalide")
        
        risk_percent = proba[1] * 100
        if np.isnan(risk_percent):
            raise ValueError("Calcul de risque invalide (NaN)")

        # Définition du niveau de risque
        if risk_percent < 30:
            risk_class = "low"
            risk_label = "Faible"
            risk_color = "#28a745"
        elif risk_percent < 70:
            risk_class = "medium"
            risk_label = "Modéré"
            risk_color = "#ffc107"
        else:
            risk_class = "high"
            risk_label = "Élevé"
            risk_color = "#dc3545"

        # Affichage principal
        st.markdown("---")
        st.subheader(f"📊 Résultats ({model_type})")
        
        cols = st.columns([1, 2])
        with cols[0]:
            # Jauge visuelle
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid {risk_color}; 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2 class="risk-{risk_class}" style="margin: 0;">{risk_label}</h2>
                <h1 style="color: {risk_color}; margin: 5px 0;">{risk_percent:.1f}%</h1>
                <p>de risque cardiaque</p>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            # Graphique détaillé
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=['Sain', 'À risque'], y=proba, 
                        palette=['#28a745', '#dc3545'], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Probabilités de prédiction')
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1%}", 
                           (p.get_x() + p.get_width()/2, p.get_height()),
                           ha='center', va='center', xytext=(0, 10), 
                           textcoords='offset points')
            st.pyplot(fig)

        # Conclusion et recommandations
        if prediction[0] == 1:
            st.error("""
            **⚠️ Conclusion:** Risque élevé de maladie cardiaque détecté
            """)
        else:
            st.success("""
            **✅ Conclusion:** Risque faible de maladie cardiaque détecté
            """)
            
        show_recommendations(prediction)

    except Exception as e:
        st.error(f"Erreur d'affichage des résultats : {str(e)}")

# ---- 4. Fonctions auxiliaires ----
def show_recommendations(prediction):
    """Affiche les recommandations personnalisées"""
    st.markdown("---")
    st.subheader("📌 Recommandations personnalisées")
    
    if prediction[0] == 1:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            **🏥 Consultation urgente**
            - Rendez-vous cardiologue sous 7 jours
            - Bilan complet: ECG + échocardiogramme
            - Surveillance tensionnelle quotidienne
            """)
        with cols[1]:
            st.markdown("""
            **💊 Traitement**
            - Évaluation médicamenteuse
            - Bilan lipidique approfondi
            - Test d'effort supervisé
            """)
    else:
        st.markdown("""
        **🩺 Prévention active**
        - Contrôles annuels après 40 ans
        - Activité physique 150 min/semaine
        - Régime méditerranéen recommandé
        """)

def compare_models(models, input_df):
    """Affiche une comparaison des deux modèles"""
    st.markdown("---")
    st.subheader("🆚 Comparaison des modèles")
    
    try:
        # Prédictions DT
        dt_pred = models['dt'].predict(input_df)[0]
        dt_proba = models['dt'].predict_proba(input_df)[0][1]
        
        # Prédictions ANN
        scaled_input = models['scaler'].transform(input_df)
        ann_proba = models['ann'].predict(scaled_input)[0][0]
        ann_pred = 1 if ann_proba >= 0.5 else 0
        
        # Métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Decision Tree", 
                "À risque" if dt_pred == 1 else "Sain",
                f"{dt_proba*100:.1f}% de confiance"
            )
        
        with col2:
            st.metric(
                "Réseau de Neurones", 
                "À risque" if ann_pred == 1 else "Sain",
                f"{ann_proba*100:.1f}% de confiance"
            )

        # Graphique comparatif
        fig, ax = plt.subplots(figsize=(10, 4))
        models_data = {
            'Decision Tree': dt_proba,
            'Réseau Neuronal': ann_proba
        }
        sns.barplot(x=list(models_data.keys()), y=list(models_data.values()),
                    palette="viridis", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilité de risque')
        ax.set_title('Comparaison des prédictions')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), 
                       textcoords='offset points')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur de comparaison : {str(e)}")

# ---- 5. Fonction principale ----
def main():
    """Point d'entrée de l'application"""
    st.title("🫀 **Analyse de Risque Cardiaque**")

    # Chargement des modèles
    with st.spinner("Chargement des modèles..."):
        models = load_models()
    
    if not models:
        st.stop()  # Arrêt si échec de chargement

    # Sélecteur de modèle
    model_type = st.radio(
        "Sélectionnez le modèle:",
        ("Decision Tree", "Réseau de Neurones (ANN)"),
        horizontal=True,
        index=0
    )

    # Formulaire et onglets
    input_df = show_input_form(models['feature_names'])

    # Prédiction si données soumises
    if input_df is not None:
        try:
            # Debug: Afficher les données d'entrée
            st.write("## Données analysées")
            st.dataframe(input_df.style.highlight_max(axis=0))

            # Prédiction
            with st.spinner("Analyse en cours..."):
                if model_type == "Decision Tree":
                    prediction = models['dt'].predict(input_df)
                    proba = models['dt'].predict_proba(input_df)[0]
                else:
                    scaled_input = models['scaler'].transform(input_df)
                    proba_ann = float(models['ann'].predict(scaled_input)[0][0])
                    proba = [1 - proba_ann, proba_ann]
                    prediction = [1] if proba_ann >= 0.5 else [0]

                # Affichage des résultats
                display_results(prediction, proba, model_type)
                
                # Option de comparaison
                if st.checkbox("Afficher la comparaison des modèles"):
                    compare_models(models, input_df)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            st.error("Veuillez vérifier vos données et réessayer.")

if __name__ == "__main__":
    main()
