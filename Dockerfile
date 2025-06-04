
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY indexing_app.py .
COPY query_with_rag_llm_dock1.py .
COPY data_source.py .
COPY domain_knowledge.py .  
COPY enhanced_prompt.py .
COPY domain_knowledge_updated.py .  
COPY enhanced_prompt_updated.py .  
COPY emission_prompted_qa_full.jsonl .
COPY generated_prompted_qa_from_domain_knowledge.jsonl .
COPY verify_domain_intelligence.py .


RUN pip install --no-cache-dir -r requirements.txt

ENV STREAMLIT_PORT=8501
EXPOSE 8501

CMD ["streamlit", "run", "query_with_rag_llm_dock1.py"]


