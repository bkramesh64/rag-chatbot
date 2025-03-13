# rag-chatbot
This is a demo RAG based document Chatbot.  Document PDF used is DSPy.  
It will answer the questions around this document and any other context will not be answered
To use this
1. Build Docker Image using DockerFile
   docker build -t my-streamlit-app .

2.RUN Docker
   docker run -p 8501:8501 my-streamlit-app
