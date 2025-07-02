import streamlit as st
from RAGchatbot.data_ingestion import load_data
from RAGchatbot.embedding import download_gemini_embedding
from google.api_core.exceptions import GoogleAPICallError, RetryError, ServiceUnavailable
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from src.helper import voice_input, llm_model_object, text_to_speech
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
doc_path = os.getenv("DOC_PATH")
model_name = os.getenv("MODEL_NAME")

# Initialize the conversation history and query engine
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def initialize_query_engine():
    try:
        genai.configure(api_key=google_api_key)

        with open(doc_path, "rb") as f:
            document = load_data(f)

        if not document:
            st.error("Failed to load the document.")
            return None

        model = Gemini(model_name=model_name, api_key=google_api_key)
        gemini_embed_model = GeminiEmbedding(model_name=model_name)

        query_engine = download_gemini_embedding(model, document)

        return query_engine

    except (GoogleAPICallError, RetryError, ServiceUnavailable) as e:
        st.error(f"API error: {e.message}")
    except Exception as e:
        st.error(f"Error initializing engine: {str(e)}")
    return None

def main():
    st.set_page_config(page_title="RAG-chatbot")
    st.header("Chat with me")

    if not all([google_api_key, doc_path, model_name]):
        st.error("Missing required environment variables: GOOGLE_API_KEY, DOC_PATH, MODEL_NAME")
        return

    if "query_engine" not in st.session_state:
        with st.spinner("Initializing..."):
            query_engine = initialize_query_engine()
            if query_engine:
                st.session_state.query_engine = query_engine
                st.success("Document processed and query engine ready.")
            else:
                st.stop()

    if st.session_state.conversation:
        st.write("### Conversation History")
        for chat in st.session_state.conversation:
            st.write(f"👤:** {chat['question']}")
            st.write(f"🤖:** {chat['response']}")
            st.write("---")

    if st.button("🎙️"):
        with st.spinner("Listening..."):
            text = voice_input()
            response = llm_model_object(text)
            text_to_speech(response)
            with open("speech.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.text_area(label="Response:", value=response, height=350)
            st.audio(audio_bytes)
            st.download_button(label="Download Speech", data=audio_bytes, file_name="speech.mp3", mime="audio/mp3")

    if "query_engine" in st.session_state:
        user_question = st.text_input("Ask your question", key="user_question_input")
        if st.button("Send", key="send_button"):
            if user_question:
                with st.spinner("Processing..."):
                    try:
                        response = st.session_state.query_engine.query(user_question)
                        st.session_state.conversation.append({"question": user_question, "response": response.response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
