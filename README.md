# rag-local-app
This is a Simple RAG based FAQ Chat application which can run locally and using open source LLM model (mistral 7b) and frameworks.

# Prerequisites 
Install Ollama on your system
Pull the required model:
    ollama pull mistral:7b
Ensure Ollama is running before starting the application
    ollama serve

# Build and Run
docker build -t rag-local-app .
docker run -p 8501:8501 rag-local-app

# Explore Chatbot
Run app on http://localhost:8501 and ask questions based on given FAQ - ./data/faq.txt. This application uses a local LLM via Ollama.
As a result, responses may take a few seconds, especially on the first query, since the model runs on local machine resources.

