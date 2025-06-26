
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

# Configuration de la page
st.set_page_config(
    page_title="Chatbot IA - Génération de Texte",
    page_icon="🤖",
    layout="wide"
)

# CSS pour le style du chat
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
    align-items: flex-end;
}
.chat-message.bot {
    background-color: #475063;
    align-items: flex-start;
}
.chat-message .avatar {
    width: 3rem;
    height: 3rem;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    max-width: 80%;
    word-wrap: break-word;
}
.user-message {
    background-color: #1f4e79;
    color: white;
    margin-left: auto;
}
.bot-message {
    background-color: #2d3748;
    color: white;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot_model():
    """Charge le modèle Phi-3.5-mini-instruct"""
    try:
        with st.spinner("Chargement du modèle Phi-3.5-mini-instruct..."):
            model_name = "microsoft/Phi-3.5-mini-instruct"
            
            # Chargement du tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Chargement du modèle
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Création du pipeline
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            st.success("✅ Modèle chargé avec succès!")
            return text_generator, tokenizer
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        return None, None

def generate_response(generator, tokenizer, prompt, max_length=512, temperature=0.7):
    """Génère une réponse avec le modèle"""
    try:
        # Format du prompt pour Phi-3.5
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Génération
        outputs = generator(
            formatted_prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Extraction de la réponse
        response = outputs[0]['generated_text'].strip()
        
        # Nettoyage de la réponse
        if "<|end|>" in response:
            response = response.split("<|end|>")[0].strip()
            
        return response
        
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

def display_chat_message(message, is_user=False):
    """Affiche un message de chat"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message user-message">
                👤 {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="message bot-message">
                🤖 {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("🤖 Chatbot IA - Génération de Texte")
    st.markdown("**Modèle :** Microsoft Phi-3.5-mini-instruct")
    
    # Chargement du modèle
    generator, tokenizer = load_chatbot_model()
    
    if generator is None:
        st.error("Impossible de charger le modèle. Veuillez réessayer.")
        return
    
    # Initialisation de l'historique des conversations
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar avec paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres de génération")
        
        max_length = st.slider(
            "Longueur maximale", 
            min_value=50, 
            max_value=1000, 
            value=300,
            help="Nombre maximum de tokens à générer"
        )
        
        temperature = st.slider(
            "Température", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.7, 
            step=0.1,
            help="Contrôle la créativité (plus élevé = plus créatif)"
        )
        
        st.markdown("---")
        
        # Suggestions de prompts
        st.subheader("💡 Suggestions")
        suggestions = [
            "Explique-moi les maladies cardiovasculaires",
            "Raconte-moi une histoire courte",
            "Comment améliorer ma santé cardiaque ?",
            "Écris un poème sur la technologie",
            "Donne-moi des conseils nutritionnels"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
                st.session_state.current_input = suggestion
        
        st.markdown("---")
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Zone de chat principale
    st.markdown("### 💬 Conversation")
    
    # Conteneur pour l'historique des messages
    chat_container = st.container()
    
    with chat_container:
        # Affichage de l'historique
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])
    
    # Zone de saisie
    st.markdown("---")
    
    # Utilisation du prompt suggéré s'il existe
    default_input = st.session_state.get("current_input", "")
    if default_input:
        del st.session_state.current_input
    
    # Formulaire de saisie
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Votre message :",
                value=default_input,
                placeholder="Tapez votre message ici...",
                height=100,
                key="user_input"
            )
        
        with col2:
            st.write("")  # Espacement
            st.write("")  # Espacement
            send_button = st.form_submit_button("📤 Envoyer", use_container_width=True)
    
    # Traitement du message
    if send_button and user_input.strip():
        # Ajout du message utilisateur à l'historique
        st.session_state.chat_history.append({
            "content": user_input,
            "is_user": True
        })
        
        # Affichage du message utilisateur
        display_chat_message(user_input, is_user=True)
        
        # Génération de la réponse
        with st.spinner("🤖 Génération en cours..."):
            response = generate_response(
                generator, 
                tokenizer, 
                user_input, 
                max_length=max_length, 
                temperature=temperature
            )
        
        # Ajout de la réponse à l'historique
        st.session_state.chat_history.append({
            "content": response,
            "is_user": False
        })
        
        # Affichage de la réponse
        display_chat_message(response, is_user=False)
        
        # Rechargement pour afficher les nouveaux messages
        st.rerun()
    
    # Informations sur le modèle
    with st.expander("ℹ️ À propos du modèle"):
        st.markdown("""
        **Microsoft Phi-3.5-mini-instruct** est un modèle de langage compact mais puissant :
        
        - 🔹 **Taille :** 3.8 milliards de paramètres
        - 🔹 **Optimisé pour :** Conversations et instructions
        - 🔹 **Langues :** Multilingue (français, anglais, etc.)
        - 🔹 **Usage :** Génération de texte, Q&A, assistance
        
        Ce modèle offre un excellent équilibre entre performance et efficacité computationnelle.
        """)
    
    # Statistiques de la session
    if st.session_state.chat_history:
        st.markdown("---")
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m["is_user"]])
        bot_messages = total_messages - user_messages
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages total", total_messages)
        with col2:
            st.metric("Vos messages", user_messages)
        with col3:
            st.metric("Réponses IA", bot_messages)

if __name__ == "__main__":
    main()
