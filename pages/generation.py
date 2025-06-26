import os
import streamlit as st
from huggingface_hub import InferenceClient
from groq import Groq
import sounddevice as sd
import soundfile as sf
import tempfile



# Configuration des clés API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Initialisation des clients
hf_client = InferenceClient(
    model="microsoft/Phi-3.5-mini-instruct",
    token=HF_TOKEN
)
groq_client = Groq(api_key=GROQ_API_KEY)

# Fonction pour générer une réponse textuelle avec Phi-3.5-mini-instruct
def generate_text_response(prompt):
    try:
        response = hf_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Vous êtes un assistant utile. Répondez de manière claire et concise en français."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur lors de la génération de texte : {str(e)}")
        return None

# Fonction pour générer un fichier audio à partir du texte avec Groq
def generate_speech(text):
    try:
        response = groq_client.audio.speech.create(
            model="whisper-large-v3-turbo",  # Modèle Groq pour la synthèse vocale
            input=text,
            voice="alloy"  # Voix par défaut, ajustable selon les options de Groq
        )
        # Sauvegarde du fichier audio temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'audio : {str(e)}")
        return None

# Interface Streamlit
st.title("Chatbot avec Phi-3.5-mini-instruct et Groq TTS")
st.write("Posez une question, et le chatbot répondra en texte et en audio.")

# Zone de saisie utilisateur
user_input = st.text_input("Votre question :", "")

if st.button("Envoyer"):
    if user_input:
        with st.spinner("Génération de la réponse..."):
            # Générer la réponse textuelle
            text_response = generate_text_response(user_input)
            if text_response:
                st.write("**Réponse :**")
                st.write(text_response)

                # Générer et jouer l'audio
                audio_file = generate_speech(text_response)
                if audio_file:
                    st.write("**Audio :**")
                    audio_data, sample_rate = sf.read(audio_file)
                    st.audio(audio_file, format="audio/wav")
                    sd.play(audio_data, sample_rate)
                    sd.wait()  # Attendre que l'audio soit joué
                    os.remove(audio_file)  # Supprimer le fichier temporaire
    else:
        st.warning("Veuillez entrer une question.")