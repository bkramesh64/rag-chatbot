#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 19:12:19 2025

@author: rameshbk
"""

import streamlit as st
import os
from datetime import datetime
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
import subprocess
import re

# --- CONFIG ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CACHE_DIR = os.environ.get("CACHE_DIR", "./.cache")

# Product domain
SESSION_ID = "45214b21-multi-2bd9a2894c"
MODEL_NAME = "llama3:8b"

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """You are a technical product assistant. 
Provide precise, actionable answers based strictly on the information in the context.
DO NOT reference document names, page numbers, or source material in your answer.

CRITICAL: When asked for technical specifications, focus ONLY on hardware specifications like:
- Network interfaces and connectivity options (ports, protocols supported)
- Physical specifications (dimensions, weight, enclosure type)
- Electrical specifications (power requirements, voltage, current)
- Performance parameters (speed, throughput, resolution, memory)
- Operating conditions (temperature range, environmental ratings)

DO NOT include information about technical support, customer service, manuals, or return policies when asked for technical specifications.

Question: {query_str}

Context Information:
{context_str}

Answer:"""

# --- INITIALIZE SESSION STATE ---
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None

# --- TECH SPEC DETECTION ---
def is_tech_spec_query(query):
    """Detect if a query is asking for technical specifications"""
    query_lower = query.lower()
    
    # Direct spec keywords
    spec_keywords = [
        "technical spec", "specs", "specification", "dimensions", "parameters",
        "measurements", "electrical spec", "performance spec", "power requirement",
        "interface spec", "physical spec", "operating spec", "how big", "how much power",
        "what are the spec", "data sheet", "datasheet", "technical data"
    ]
    
    # More specific product spec categories
    specific_spec_categories = [
        "network interface", "connectivity", "protocol", "dimensions",
        "weight", "size", "form factor", "power input", "voltage", "current",
        "temperature range", "operating temperature", "resolution", "accuracy",
        "memory", "storage", "processor", "throughput", "bandwidth", "ports"
    ]
    
    # Check for direct spec keywords
    for keyword in spec_keywords:
        if keyword in query_lower:
            return True
            
    # Check for specific spec categories
    for category in specific_spec_categories:
        if category in query_lower:
            return True
            
    # Check for typical spec patterns
    if ("what" in query_lower and (
            "technical" in query_lower or 
            "hardware" in query_lower or 
            "spec" in query_lower)):
        return True
        
    return False

# --- LOAD MODELS ---
@st.cache_resource
def get_llm_and_embed():
    llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_URL, request_timeout=120.0, temperature=0.1)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

# --- LOAD INDEX ---
@st.cache_resource
def load_vector_index():
    folders = [f for f in os.listdir(CACHE_DIR) if f.startswith(SESSION_ID)]
    if not folders:
        st.error(f"❌ No index found for Session ID: `{SESSION_ID}`")
        return None
    storage_path = os.path.join(CACHE_DIR, folders[0])
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    return load_index_from_storage(storage_context)

# --- MAIN APP ---
def main():
    st.set_page_config(
        page_title="Product Technical Assistant",
        page_icon="🧠",
        layout="wide"
    )
    
    init_session_state()
    
    # App title
    st.title("Product Technical Assistant")
    st.markdown("Your AI-powered guide for product technical information")
    
    # Load resources
    with st.spinner("Loading models and data..."):
        llm, embed_model = get_llm_and_embed()
        index = load_vector_index()
        if not index:
            st.stop()
        
        # Set up memory 
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    
    # Set up columns for chat and context
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                st.markdown(f"**You:** {entry['content']}")
            else:
                st.markdown(f"**Assistant:** {entry['content']}")
            st.markdown("---")
        
        # User input form
        with st.form(key="query_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask about product specifications or features:",
                placeholder="e.g., What are the technical specifications of the DPA XL?"
            )
            submit_button = st.form_submit_button("Ask")
            
            if submit_button and user_query:
                # Check if this is a duplicate query
                is_duplicate = (st.session_state.last_query is not None and 
                                user_query.strip().lower() == st.session_state.last_query.strip().lower())
                
                if not is_duplicate:
                    # Save this as the last query
                    st.session_state.last_query = user_query
                    
                    # Add user query to history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_query,
                        "time": datetime.now().strftime("%H:%M")
                    })
                    
                    # Process query
                    with st.spinner("Finding answer..."):
                        # Determine retrieval approach based on query type
                        similarity_top_k = 20 if is_tech_spec_query(user_query) else 10
                        
                        # Set up retriever
                        retriever = VectorIndexRetriever(
                            index=index,
                            similarity_top_k=similarity_top_k
                        )
                        
                        # Set up reranker
                        postprocessor = SentenceTransformerRerank(top_n=8 if is_tech_spec_query(user_query) else 5)
                        
                        # Set up prompt
                        prompt_template = PromptTemplate(PROMPT_TEMPLATE)
                        
                        # Create query engine
                        query_engine = RetrieverQueryEngine.from_args(
                            retriever=retriever,
                            llm=llm,
                            memory=memory,
                            node_postprocessors=[postprocessor],
                            text_qa_template=prompt_template
                        )
                        
                        # Execute query
                        response = query_engine.query(user_query)
                        answer = response.response
                        
                        # Handle technical spec queries with non-spec responses
                        if is_tech_spec_query(user_query) and (
                            "technical support" in answer.lower() or
                            "customer service" in answer.lower() or
                            "return merchandise" in answer.lower() or
                            not any(kw in answer.lower() for kw in ["interface", "dimension", "physical", "power", "voltage", "weight"])
                        ):
                            # Try again with more explicit query
                            explicit_query = f"Provide ONLY hardware specifications for {user_query.split('specifications')[1].strip() if 'specifications' in user_query else user_query}"
                            
                            # More aggressive retrieval
                            retriever = VectorIndexRetriever(index=index, similarity_top_k=30)
                            
                            # Update query engine
                            query_engine = RetrieverQueryEngine.from_args(
                                retriever=retriever,
                                llm=llm,
                                memory=memory,
                                node_postprocessors=[SentenceTransformerRerank(top_n=10)],
                                text_qa_template=prompt_template
                            )
                            
                            # Try again
                            response = query_engine.query(explicit_query)
                            new_answer = response.response
                            
                            # If better answer found, use it
                            if any(kw in new_answer.lower() for kw in ["interface", "dimension", "physical", "power", "voltage", "weight"]):
                                answer = new_answer
                            else:
                                answer = f"Based on the available documentation, I couldn't find detailed technical specifications for this product. The documentation appears to focus more on operational procedures, troubleshooting, and support information rather than hardware specifications."
                        
                        # Add response to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "time": datetime.now().strftime("%H:%M"),
                            "sources": [{"text": node.text[:300], "score": node.score} for node in response.source_nodes[:3]]
                        })
    
    with col2:
        # App info
        st.subheader("About")
        st.markdown("""
        This assistant helps you find technical information about products from the documentation.
        
        Try asking about:
        - Technical specifications
        - Product features
        - Operating parameters
        - Configuration options
        """)
        
        # View sources if available
        if st.session_state.chat_history and "sources" in st.session_state.chat_history[-1]:
            st.subheader("Source Information")
            sources = st.session_state.chat_history[-1]["sources"]
            
            for i, source in enumerate(sources):
                with st.expander(f"Source #{i+1}"):
                    st.markdown(f"**Relevance:** {source['score']:.2f}")
                    st.markdown(f"**Content:** {source['text']}...")
        
        # Clear chat button
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.last_query = None

if __name__ == "__main__":
    main()