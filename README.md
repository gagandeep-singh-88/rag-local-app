# rag-local-app
This project is a simple Retrieval-Augmented Generation (RAG) based FAQ chatbot that runs entirely on a local machine. It leverages open-source LLMs, specifically the Mistral 7B model, to provide accurate answers based on a predefined FAQ dataset.

The application combines modern frameworks such as LlamaIndex for data indexing and retrieval, Ollama for running the LLM locally, and Streamlit for an interactive chat-based user interface.

# Prerequisites 
Install Ollama on your system

Pull the required model - 
    
    ollama pull mistral:7b

Ensure Ollama is running before starting the application - 
    
    ollama serve

# Build and Run
docker build -t rag-local-app .

docker run -p 8501:8501 rag-local-app

# Explore Chatbot
Run app on http://localhost:8501 and ask questions based on given FAQ - ./data/faq.txt. This application uses a local LLM via Ollama.
As a result, responses may take a few seconds, especially on the first query, since the model runs on local machine resources.

