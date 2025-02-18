import streamlit as st
from RAGchatbot.data_ingestion import load_data
from RAGchatbot.embedding import download_gemini_embedding
from google.api_core.exceptions import GoogleAPICallError, RetryError, ServiceUnavailable
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Initialize the conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def main():
    st.set_page_config(page_title="RAG-chatbot")

    st.header("Chat with me")

    # File uploader for the document
    doc = st.file_uploader("Upload your document", type=['pdf', 'txt'], key="doc_uploader")

    # Display available models in the sidebar
    st.sidebar.subheader("Available Models")
    google_api_key = st.sidebar.text_input("Google API Key", type="password", key="api_key_input")

    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            models = genai.list_models()
            text_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
            selected_model = st.sidebar.selectbox("Select a text model", text_models, key="model_select")
        except Exception as e:
            st.sidebar.error(f"Failed to retrieve models: {str(e)}")
            text_models = []
            selected_model = None

        if st.button("Submit Document"):
            if doc is not None and selected_model:
                with st.spinner("Processing..."):
                    try:
                        # Load the document
                        document = load_data(doc)
                        if not document:
                            st.error("Failed to load the document. Please check the file and try again.")
                            return

                        # Initialize the Gemini model with the selected text model
                        model = Gemini(model_name=selected_model, api_key=google_api_key)
                        gemini_embed_model = GeminiEmbedding(model_name=selected_model)

                        if not model:
                            st.error("Failed to load the model. Please check the model configuration and try again.")
                            return

                        # Create the query engine with the embedding model and document
                        query_engine = download_gemini_embedding(model, document)

                        if not query_engine:
                            st.error("Failed to create the query engine. Please check the embedding configuration and try again.")
                            return

                        st.session_state.query_engine = query_engine
                        st.success("Document processed successfully.")

                    except GoogleAPICallError as e:
                        st.error(f"API call error: {e.message}")
                    except RetryError as e:
                        st.error(f"Retry error: {e.message}")
                    except ServiceUnavailable as e:
                        st.error(f"Service unavailable: {e.message}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please upload a document and select a model.")
    else:
        st.warning("Please enter your Google API Key to continue.")

    # Display conversation history
    if st.session_state.conversation:
        st.write("### Conversation History")
        for chat in st.session_state.conversation:
            st.write(f"**👤:** {chat['question']}")
            st.write(f"**🤖:** {chat['response']}")
            st.write("---")

    if "query_engine" in st.session_state:
        user_question = st.text_input("Ask your question", key="user_question_input")

        if st.button("Send", key="send_button"):
            if user_question:
                with st.spinner("Processing..."):
                    try:
                        query_engine = st.session_state.query_engine
                        response = query_engine.query(user_question)
                        st.session_state.conversation.append({"question": user_question, "response": response.response})
                        # Clear the text input field by resetting the state
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
