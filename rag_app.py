#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:09:13 2025

@author: rameshbk
"""

import os
import base64
import tempfile
import uuid
import gc
import streamlit as st


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# LlamaIndex imports
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "index" not in st.session_state:
    st.session_state.index = None

session_id = st.session_state.id

# Sidebar: Upload Document
with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")
                
                if file_key not in st.session_state.get('file_cache', {}):
                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True
                    )
                    docs = loader.load_data()
                    st.write(f"Loaded {len(docs)} documents.")
                    
                    # Setup LLM & embedding model
                    llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
                    Settings.llm = llm
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embed_model

                    # Create an index over loaded documents
                    st.write("Creating index (this may take a while)...")
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                    st.session_state.index = index

                    # Define a custom QA prompt template
                    qa_prompt_tmpl_str = (
                        "Context information is below:\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, answer the query in a crisp and specific manner. "
                        "If you don't have enough information, respond with \"I don't know!\".\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    
                    # Create a query engine from the index (streaming enabled)
                    query_engine = index.as_query_engine(streaming=True)
                    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
                    st.session_state.query_engine = query_engine
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]
                
                st.sidebar.success("Ready to Chat!")
                st.markdown("### PDF Preview")
                base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>"""
                st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")
            st.stop()

# Main Chat Interface
col1, col2 = st.columns([6, 1])
with col1:
    st.header("Chat with Docs using Llama-3.2")
with col2:
    if st.button("Clear ↺"):
        st.session_state.messages = []
        gc.collect()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.query_engine is None:
        st.error("No document indexed. Please upload a document first.")
    else:
        try:
            response = st.session_state.query_engine.query(prompt)
            response_text = str(response)
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"An error occurred: {e}")
