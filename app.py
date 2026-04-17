import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

@st.cache_resource
def load_index():
    # LLM
    Settings.llm = Ollama(
        model="mistral:7b",
        base_url="http://host.docker.internal:11434",
        temperature=0.5,
        request_timeout=300
    )

    # Embeddings
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://host.docker.internal:11434"
    )

    # Chunking
    Settings.node_parser = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True) 

    return index

index = load_index()
query_engine = index.as_query_engine(similarity_top_k=5)


# -------------------------------
# UI
# -------------------------------
st.title("📄 FAQ Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask your question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(user_input)
            answer = str(response)

            st.markdown(answer)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})