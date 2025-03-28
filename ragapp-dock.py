#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 17:25:10 2025

@author: rameshbk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 08:02:11 2025

@author: rameshbk
"""

import os
import base64
import tempfile
import uuid
import gc
import streamlit as st
from typing import List, Dict, Any
import time
import pandas as pd

# Configure the Streamlit page first - this must be the first Streamlit command
st.set_page_config(
    page_title="Emission Regulation Assistant",
    layout="wide"
)

# Environment setup
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
CACHE_DIR = os.environ.get("CACHE_DIR", "./.cache")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:8b")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# LlamaIndex imports
from llama_index.core import (
    Settings, 
    PromptTemplate, 
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "index" not in st.session_state:
    st.session_state.index = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "tracking_sources" not in st.session_state:
    st.session_state.tracking_sources = False
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "vehicle_type" not in st.session_state:
    st.session_state.vehicle_type = ""
if "fuel_type" not in st.session_state:
    st.session_state.fuel_type = ""
if "search_term" not in st.session_state:
    st.session_state.search_term = ""
if "expanded_section" not in st.session_state:
    st.session_state.expanded_section = None

session_id = st.session_state.id
#MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:8b")


# Initialize models once at startup
@st.cache_resource
def get_models():
    """Initialize and cache the LLM and embedding models."""
    llm = Ollama(
        model=MODEL_NAME, 
        request_timeout=120.0,
        temperature=0.1,
        num_ctx=4096,
        base_url=OLLAMA_URL
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5", 
        trust_remote_code=True,
        embed_batch_size=32,
        max_length=512
    )
    return llm, embed_model

# Initialize models
llm, embed_model = get_models()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 300
Settings.chunk_overlap = 100

# Document processing functions (keep the existing functions)
def process_multi_documents(dir_path: str, index_id: str) -> VectorStoreIndex:
    """Process multiple documents and create a combined index."""
    # Existing implementation...
    storage_path = os.path.join(CACHE_DIR, f"{session_id}-{index_id}")
    
    if os.path.exists(storage_path):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            return load_index_from_storage(storage_context)
        except Exception as e:
            st.sidebar.warning(f"Could not load cached index: {e}")
    
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=200,
        paragraph_separator="\n\n",
        separator=" ",
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    loader = SimpleDirectoryReader(
        input_dir=dir_path,
        required_exts=[".pdf"],
        recursive=True,
        file_extractor={"pdf": "default"}
    )
    
    st.sidebar.info("Loading all documents...")
    docs = loader.load_data()
    st.sidebar.success(f"Successfully loaded {len(docs)} document chunks")
    
    Settings.chunk_size = 300
    Settings.chunk_overlap = 100
    
    st.sidebar.info("Creating combined index...")
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[parser],
        show_progress=True
    )
    
    st.sidebar.info("Saving index to disk...")
    index.storage_context.persist(persist_dir=storage_path)
    
    return index

def process_document(file_path: str, file_name: str) -> VectorStoreIndex:
    """Process a single document and create an index."""
    # Existing implementation...
    storage_path = os.path.join(CACHE_DIR, f"{session_id}-{file_name.replace('.pdf', '')}")
    
    if os.path.exists(storage_path):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            return load_index_from_storage(storage_context)
        except Exception as e:
            st.sidebar.warning(f"Could not load cached index: {e}")
    
    parser = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=100,
        paragraph_separator="\n\n",
        separator=" ",
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    loader = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor={"pdf": "default"}
    )
    
    docs = loader.load_data()
    
    Settings.chunk_size = 300
    Settings.chunk_overlap = 100
    
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[parser],
        show_progress=True
    )
    
    index.storage_context.persist(persist_dir=storage_path)
    
    return index

# Automotive-specific prompt template (keep the existing template)
AUTO_STANDARD_PROMPT = (
    "Context information from multiple automotive regulatory documents is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "You are a specialized assistant for Bharat Stage VI emission norms and related automotive testing standards.\n"
    "Focus on these specific areas:\n"
    "1. Testing methods and equipment for L2 category vehicles\n"
    "2. Type approval procedures specified in CMV Rules 115, 116, and 126\n"
    "3. Conformity of Production (COP) testing requirements\n"
    "4. Technical specifications and compliance standards, including AIS standards\n\n"
    "Given the context information, answer the query thoroughly and accurately.\n"
    "Include specific regulatory references where applicable.\n"
    "If you find even partial information about the query topic, share what you know while acknowledging limitations.\n"
    "If absolutely no information about the topic exists in the document context, state: 'I don't have information about this specific topic in the uploaded documents.'\n\n"
    "Remember the following when answering:\n"
    "- AIS standards (like AIS-137) are Automotive Industry Standards that specify testing procedures\n"
    "- Even if you only have fragments of information about a standard, provide those details rather than saying you don't have enough information\n"
    "- If a specific part or section number is mentioned in the query but not in your context, provide information about the general standard\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Define emission standards glossary
EMISSION_GLOSSARY = [
    {"term": "AIS 137", "definition": "Automotive Industry Standard that specifies test methods, equipment requirements, and procedures for emission testing of various vehicle categories under BS VI."},
    {"term": "AIS 175", "definition": "Automotive Industry Standard that provides technical specifications for L-category vehicles regarding emissions and other requirements."},
    {"term": "BS VI", "definition": "Bharat Stage VI emission standards, equivalent to Euro 6, that set limits for pollutants like CO, HC, NOx, and PM for different vehicle categories."},
    {"term": "COP", "definition": "Conformity of Production testing ensures that mass-produced vehicles comply with the originally type-approved specifications for emissions."},
    {"term": "GSR", "definition": "Gazette of India Special Release, notification issued by the government with legal provisions for implementing emission norms."},
    {"term": "L2 Category", "definition": "Three-wheeled vehicle with engine capacity not exceeding 50 cm³ and maximum design speed not exceeding 50 km/h."},
    {"term": "OBD", "definition": "On-Board Diagnostics system that monitors emissions performance and alerts when systems are not functioning properly."},
    {"term": "PUC", "definition": "Pollution Under Control certificate required for all vehicles, indicating they meet emission standards during operation."},
    {"term": "Type Approval", "definition": "Process of certifying that a vehicle model meets all applicable standards before mass production."},
    {"term": "IDC", "definition": "Indian Driving Cycle - standardized test cycle used for emission testing of L category vehicles in India."},
    {"term": "MIDC", "definition": "Modified Indian Driving Cycle - used for testing M and N category vehicles."},
    {"term": "WMTC", "definition": "World Motorcycle Test Cycle - harmonized international test cycle for two and three-wheelers."}
]

# Define compliance requirements database
COMPLIANCE_REQUIREMENTS = {
    'L2-Petrol': {
        "title": "L2 Category - Petrol Vehicle Requirements",
        "standards": "AIS 137 (Part 3)",
        "emissionLimits": "CO: 0.5 g/km, HC+NOx: 0.35 g/km",
        "testProcedures": "IDC (Indian Driving Cycle) test as per AIS 137",
        "equipment": "Chassis dynamometer, Gas analyzers for CO, HC, NOx",
        "additionalRequirements": "Evaporative emission control system test",
        "copFrequency": "Every 6 months or after 5000 vehicles, whichever is earlier"
    },
    'L2-CNG': {
        "title": "L2 Category - CNG Vehicle Requirements",
        "standards": "AIS 137 (Part 3) with CNG adaptations",
        "emissionLimits": "CO: 0.5 g/km, HC+NOx: 0.3 g/km",
        "testProcedures": "IDC (Indian Driving Cycle) test as per AIS 137",
        "equipment": "Chassis dynamometer, Gas analyzers for CO, HC, NOx, Methane analyzer",
        "additionalRequirements": "Leak testing for CNG systems",
        "copFrequency": "Every 6 months or after 5000 vehicles, whichever is earlier"
    },
    'L5-Petrol': {
        "title": "L5 Category - Petrol Vehicle Requirements",
        "standards": "AIS 137 (Part 3)",
        "emissionLimits": "CO: 0.5 g/km, HC+NOx: 0.35 g/km, PM: 0.0045 g/km (for DI engines)",
        "testProcedures": "WMTC (World Motorcycle Test Cycle) as per AIS 137",
        "equipment": "Chassis dynamometer, Gas analyzers for CO, HC, NOx, PM measurement system",
        "additionalRequirements": "OBD Stage II, Evaporative emission control system test",
        "copFrequency": "Every 6 months or after 5000 vehicles, whichever is earlier"
    },
    'L5-Diesel': {
        "title": "L5 Category - Diesel Vehicle Requirements",
        "standards": "AIS 137 (Part 3)",
        "emissionLimits": "CO: 0.5 g/km, HC+NOx: 0.3 g/km, PM: 0.025 g/km",
        "testProcedures": "WMTC (World Motorcycle Test Cycle) as per AIS 137",
        "equipment": "Chassis dynamometer, Gas analyzers for CO, HC, NOx, PM measurement system, Opacity meter",
        "additionalRequirements": "OBD Stage II, SCR system or DPF for PM control",
        "copFrequency": "Every 6 months or after 5000 vehicles, whichever is earlier"
    },
    'M1-Petrol': {
        "title": "M1 Category - Petrol Vehicle Requirements",
        "standards": "AIS 137 (Part 2)",
        "emissionLimits": "CO: 1.0 g/km, THC: 0.1 g/km, NMHC: 0.068 g/km, NOx: 0.06 g/km, PM: 0.0045 g/km (for DI engines)",
        "testProcedures": "MIDC (Modified Indian Driving Cycle) as per AIS 137",
        "equipment": "Chassis dynamometer, CVS system, Gas analyzers for CO, THC, NMHC, NOx, PM measurement system",
        "additionalRequirements": "OBD Stage II, Evaporative emission control system test, Durability testing",
        "copFrequency": "Every 6 months or after 10,000 vehicles, whichever is earlier"
    },
    'M1-Diesel': {
        "title": "M1 Category - Diesel Vehicle Requirements",
        "standards": "AIS 137 (Part 2)",
        "emissionLimits": "CO: 0.5 g/km, THC+NOx: 0.17 g/km, NOx: 0.08 g/km, PM: 0.0045 g/km, PN: 6.0×10¹¹ #/km",
        "testProcedures": "MIDC (Modified Indian Driving Cycle) as per AIS 137",
        "equipment": "Chassis dynamometer, CVS system, Gas analyzers for CO, THC, NOx, PM measurement system, PN counter",
        "additionalRequirements": "OBD Stage II, SCR system and DPF for PM control, RDE testing",
        "copFrequency": "Every 6 months or after 10,000 vehicles, whichever is earlier"
    }
}

# Testing procedures content
TESTING_PROCEDURES = {
    "typeApproval": {
        "title": "Type Approval Process",
        "content": """The emission type approval process requires vehicle manufacturers to follow these steps:
1. Submit application to testing agency with vehicle specifications
2. Provide prototype vehicle for testing
3. Complete emission tests according to applicable driving cycles (IDC/MIDC/WMTC)
4. Conduct evaporative emission tests (for gasoline vehicles)
5. Complete durability testing (full durability or deterioration factors)
6. Submit test reports and technical documentation
7. Receive type approval certificate if compliant

Reference: AIS 137, CMV Rule 115, 116"""
    },
    "drivingCycles": {
        "title": "Emission Test Driving Cycles",
        "content": """BS VI emission tests use standardized driving cycles to simulate real-world operation:

**IDC (Indian Driving Cycle)**
Used for L category vehicles. Duration: 1180 seconds with average speed of 19.9 km/h. Includes acceleration, deceleration, constant speed, and idle conditions.

**MIDC (Modified Indian Driving Cycle)**
Used for M and N category vehicles. Based on NEDC with modifications for Indian driving conditions.

**WMTC (World Motorcycle Test Cycle)**
Alternative cycle for L5 category vehicles. International harmonized test cycle that better represents real-world riding.

Reference: AIS 137 Part 3 (Annex 1)"""
    },
    "copTesting": {
        "title": "Conformity of Production (COP) Testing",
        "content": """COP testing ensures production vehicles match type-approved specifications:

• Manufacturers must have quality control processes to ensure ongoing compliance
• Testing agencies conduct periodic verification testing
• Random vehicle selection from production line
• Testing frequency varies by vehicle category (typically every 6 months)
• Statistical methods used for pass/fail decisions
• Non-compliance can result in suspension or withdrawal of type approval

Reference: AIS 137 (Annexes on COP procedures)"""
    },
    "obdRequirements": {
        "title": "On-Board Diagnostics (OBD) Requirements",
        "content": """BS VI mandates OBD systems for monitoring emissions control systems:

**OBD Stages:**
• OBD-I: Basic monitoring of circuit continuity and rationality for key sensors
• OBD-II: Advanced monitoring including catalyst efficiency, misfire detection, oxygen sensor degradation

**Key Requirements:**
• Malfunction Indicator Lamp (MIL) activation when faults detected
• Storage of Diagnostic Trouble Codes (DTCs)
• Standardized communication protocol
• Demonstration testing for type approval

Reference: AIS 137 Parts 2 & 3 (OBD sections)"""
    }
}

# Sidebar: Upload Document and Settings
with st.sidebar:
    st.header("Emission Regulation Assistant")
    
    # Create tabs for different sidebar sections
    sidebar_tabs = st.tabs(["📚 Documents", "⚙️ Settings"])
    
    with sidebar_tabs[0]:
        st.markdown("""
        Upload automotive standards documents related to:
        * Bharat Stage VI emission norms
        * AIS 137, AIS 175 specifications
        * Type approval procedures
        * COP testing requirements
        """)
        
        # Model selection
        available_models = ["llama3.2:8b", "llama3.2:70b", "mistral-7b"]
        selected_model = st.selectbox("Select Model", available_models, index=0)
        
        # Check if model changed
        if "last_model" not in st.session_state or st.session_state.last_model != selected_model:
            st.session_state.last_model = selected_model
            if Settings.llm.model != selected_model:
                Settings.llm = Ollama(model=selected_model, request_timeout=120.0, temperature=0.1)
        
        # Multiple file uploader for standards docs
        uploaded_files = st.file_uploader("Upload automotive standards", type="pdf", 
                                        help="Upload Bharat Stage VI related documents", 
                                        accept_multiple_files=True)
        
        # Display uploaded documents
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} documents:")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        # Process button to handle multiple files
        if uploaded_files and st.button("Process Documents 🔄"):
            with st.spinner("Processing automotive standards documents..."):
                try:
                    st.session_state.processing = True
                    
                    # Create temporary directory for all documents
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file_paths = []
                        
                        # Save all files to temporary location
                        for file in uploaded_files:
                            file_path = os.path.join(temp_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getvalue())
                            temp_file_paths.append(file_path)
                        
                        # Process the documents - show progress bar
                        progress_bar = st.progress(0.0)
                        for i in range(10):
                            # Simulate progress for better UX
                            time.sleep(0.1)
                            progress_bar.progress((i + 1) * 0.1)
                        
                        # Create a combined index ID based on all filenames
                        # Hash filenames to avoid path length issues
                        import hashlib
                        file_hash = hashlib.md5("-".join([f.name for f in uploaded_files]).encode()).hexdigest()
                        index_id = f"multi-{len(uploaded_files)}-{file_hash[:10]}"
                        
                        # Process all documents together
                        index = process_multi_documents(temp_dir, index_id)
                        st.session_state.index = index
                        
                        # Store document names for reference
                        st.session_state.doc_names = [f.name for f in uploaded_files]
                        
                        # Create automotive standard-specific prompt template
                        auto_prompt_tmpl = PromptTemplate(AUTO_STANDARD_PROMPT)
                        
                        # Create a query engine with optimized settings for finding specific standards
                        query_engine = index.as_query_engine(
                            similarity_top_k=10,
                            node_postprocessors=[],
                            verbose=True
                        )
                        
                        # Update with automotive-specific prompt
                        query_engine.update_prompts({"response_synthesizer:text_qa_template": auto_prompt_tmpl})
                        
                        # Cache the query engine
                        st.session_state.query_engine = query_engine
                        
                        # Generate a unique key for the file cache
                        cache_key = f"{session_id}-multi-{file_hash[:10]}"
                        st.session_state.file_cache[cache_key] = query_engine
                        
                        # Clean up
                        st.session_state.processing = False
                        
                        # Force a cache clear to free memory
                        gc.collect()
                        
                        st.success(f"✅ {len(uploaded_files)} documents processed successfully!")
                    
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    st.session_state.processing = False
                    st.stop()
    
    with sidebar_tabs[1]:
        st.subheader("Advanced Settings")
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False, 
                                                help="Show match scores and retrieval details")
        
        # Configure chunk size
        chunk_size = st.slider("Chunk Size", 300, 1000, 512, 
                              help="Larger values capture more context but may reduce precision")
        
        # Configure retrieval parameters
        retrieval_k = st.slider("Retrieval Context Count", 5, 15, 10, 
                              help="Number of text chunks to retrieve from documents")
        
        # Update global settings based on sliders
        if "last_chunk_size" not in st.session_state or st.session_state.last_chunk_size != chunk_size:
            st.session_state.last_chunk_size = chunk_size
            Settings.chunk_size = chunk_size
            Settings.chunk_overlap = max(100, int(chunk_size * 0.4))  # 40% overlap
        
        # Add clear button to sidebar
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            gc.collect()
            st.rerun()

# Main Interface with tabs
main_tabs = st.tabs(["💬 Chat Assistant", "🔍 Compliance Lookup", "📘 Standards Glossary", "📋 Testing Procedures"])

# Tab 1: Chat Interface
with main_tabs[0]:
    st.header("Emission Test Assistant")
    
    st.markdown("""
    This assistant is specialized in Bharat Stage VI emission norms and related testing standards.
    Upload standards documents for accurate information, or use the pre-loaded knowledge base.
    """)
    
    # Display PDF previews in a collapsible section
    if "index" in st.session_state and st.session_state.index is not None and uploaded_files:
        with st.expander("📄 Standards Documents Preview", expanded=False):
            # Create tabs for each document
            if len(uploaded_files) > 1:
                tabs = st.tabs([f"Doc {i+1}: {file.name}" for i, file in enumerate(uploaded_files)])
                
                # Display each document in its tab
                for i, (tab, file) in enumerate(zip(tabs, uploaded_files)):
                    with tab:
                        base64_pdf = base64.b64encode(file.getvalue()).decode("utf-8")
                        pdf_display = f"""
                        <iframe 
                            src="data:application/pdf;base64,{base64_pdf}" 
                            width="100%" 
                            height="500" 
                            type="application/pdf">
                        </iframe>
                        """
                        st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                # Just one document
                base64_pdf = base64.b64encode(uploaded_files[0].getvalue()).decode("utf-8")
                pdf_display = f"""
                <iframe 
                    src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" 
                    height="500" 
                    type="application/pdf">
                </iframe>
                """
                st.markdown(pdf_display, unsafe_allow_html=True)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input with automotive-specific placeholder
    if prompt := st.chat_input("Ask about Bharat Stage VI standards, test methods, or compliance requirements..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if we have a document indexed
        if st.session_state.query_engine is None:
            with st.chat_message("assistant"):
                st.warning("No automotive standards document indexed. Using pre-loaded knowledge base only. For more detailed responses, please upload relevant documents.")
                
                # Respond with basic knowledge
                # Find if the query relates to any predefined glossary items or requirements
                related_terms = [item for item in EMISSION_GLOSSARY if item["term"].lower() in prompt.lower()]
                
                # Create a basic response based on glossary and requirements
                if related_terms:
                    response_text = f"Here's information about {related_terms[0]['term']}:\n\n{related_terms[0]['definition']}"
                else:
                    response_text = "Please upload relevant automotive standards documents for detailed information about your query. You can also use the Compliance Lookup, Standards Glossary, or Testing Procedures tabs for pre-loaded information."
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            # Check if we're processing a document
            if st.session_state.processing:
                with st.chat_message("assistant"):
                    st.warning("Still processing your standards document. Please wait a moment.")
            else:
                # Get the query engine
                query_engine = st.session_state.query_engine
                
                # Display source document tracking (new feature)
                st.session_state.tracking_sources = True
                
                # Rate limit queries for better performance
                current_time = time.time()
                if current_time - st.session_state.last_query_time < 1.0:
                    time.sleep(1.0)
                
                # Update last query time
                st.session_state.last_query_time = time.time()
                
                # Get the response
                try:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        
                        # Add a loading indicator
                        with st.spinner("Searching automotive standards..."):
                            # Modify response generation with improved retrieval parameters
                            response = query_engine.query(prompt)
                            response_text = str(response)
                            
                            # Add source tracking for multi-document setup with improved display
                            if hasattr(response, 'source_nodes') and response.source_nodes:
                                source_docs = []
                                for node in response.source_nodes:
                                    if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                                        source_docs.append(node.metadata['file_name'])
                                        
                                    # Debug information to understand context retrieval
                                    if st.session_state.get('debug_mode', False) and hasattr(node, 'score'):
                                        st.sidebar.write(f"Match score: {node.score:.4f} - {node.metadata.get('file_name', 'unknown')}")
                                
                                if source_docs:
                                    unique_sources = list(set(source_docs))
                                    source_info = "\n\n**Sources:** " + ", ".join(unique_sources)
                                    response_text += source_info
                            
                            response_placeholder.markdown(response_text)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Force a cache clear to free memory
                    gc.collect()
                    
                except Exception as e:
                    with st.chat_message("assistant"):
                        st.error(f"⚠️ Error: {e}")

# Tab 2: Compliance Lookup Tool
with main_tabs[1]:
    st.header("Vehicle Compliance Requirements Lookup")
    st.markdown("Enter your vehicle details to determine specific emission compliance requirements.")
    
    # Form for compliance lookup
    col1, col2 = st.columns(2)
    with col1:
        vehicle_type = st.selectbox(
            "Vehicle Category",
            options=["", "L2", "L5", "M1"],
            format_func=lambda x: {
                "": "Select Vehicle Category",
                "L2": "L2 - Three Wheeler (≤ 50cc)",
                "L5": "L5 - Three Wheeler (> 50cc)",
                "M1": "M1 - Passenger Car (≤ 9 seats)"
            }.get(x, x)
        )
    
    with col2:
        fuel_type = st.selectbox(
            "Fuel Type",
            options=["", "Petrol", "Diesel", "CNG"],
            format_func=lambda x: x if x else "Select Fuel Type"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        vehicle_class = st.selectbox(
            "Vehicle Class (optional)",
            options=["", "Passenger", "Commercial"],
            format_func=lambda x: x if x else "Select Vehicle Class (Optional)"
        )
    
    with col4:
        manufacturing_date = st.date_input(
            "Manufacturing Date (optional)",
            value=None,
            help="Select the manufacturing date of the vehicle"
        )
    
    # Search button
    if st.button("Find Requirements", type="primary", disabled=not (vehicle_type and fuel_type)):
        if vehicle_type and fuel_type:
            # Create lookup key
            lookup_key = f"{vehicle_type}-{fuel_type}"
            
            # Display requirements
            if lookup_key in COMPLIANCE_REQUIREMENTS:
                requirements = COMPLIANCE_REQUIREMENTS[lookup_key]
                
                # Create an expandable section for each requirement component
                st.subheader(requirements["title"])
                
                with st.expander("Applicable Standards", expanded=True):
                    st.write(requirements["standards"])
                
                with st.expander("Emission Limits"):
                    st.write(requirements["emissionLimits"])
                
                with st.expander("Test Procedures"):
                    st.write(requirements["testProcedures"])
                
                with st.expander("Required Test Equipment"):
                    st.write(requirements["equipment"])
                
                with st.expander("Additional Requirements"):
                    st.write(requirements["additionalRequirements"])
                
                with st.expander("COP Testing Frequency"):
                    st.write(requirements["copFrequency"])
                
                # Show suggestion to view related procedures
                st.info("ℹ️ View the Testing Procedures tab for detailed explanations of the test methods referenced above.")
            else:
                st.warning("No specific requirements found for the selected combination. Please try a different selection.")

# Tab 3: Standards Glossary
with main_tabs[2]:
    st.header("Emission Standards Glossary")
    st.markdown("Reference guide to key terms in AIS 137, AIS 175, and GSR regulations.")
    
    # Search box
    search_term = st.text_input("Search terms or definitions...", value="")
    
    # Filter glossary items based on search
    filtered_glossary = [
        item for item in EMISSION_GLOSSARY 
        if search_term.lower() in item["term"].lower() or search_term.lower() in item["definition"].lower()
    ]
    
    # Display glossary items
    if filtered_glossary:
        for item in filtered_glossary:
            with st.expander(item["term"], expanded=True if search_term else False):
                st.write(item["definition"])
    else:
        if search_term:
            st.info("No matching terms found.")
        else:
            # Display all items when no search term
            for item in EMISSION_GLOSSARY:
                with st.expander(item["term"]):
                    st.write(item["definition"])

# Tab 4: Testing Procedures
with main_tabs[3]:
    st.header("Testing Procedures Guide")
    st.markdown("Simplified explanations of key testing procedures from AIS 137 and related standards.")
    
    # Create columns for procedure selection
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Type Approval Process", use_container_width=True):
            st.session_state.expanded_section = "typeApproval"
    
        if st.button("Driving Cycles", use_container_width=True):
            st.session_state.expanded_section = "drivingCycles"
    
    with col2:
        if st.button("COP Testing", use_container_width=True):
            st.session_state.expanded_section = "copTesting"
        
        if st.button("OBD Requirements", use_container_width=True):
            st.session_state.expanded_section = "obdRequirements"
    
    # Display selected procedure content
    if st.session_state.expanded_section:
        procedure = TESTING_PROCEDURES[st.session_state.expanded_section]
        st.subheader(procedure["title"])
        st.markdown(procedure["content"])
        
        # Add option to download as PDF (placeholder functionality)
        st.download_button(
            "Download as PDF",
            data=procedure["content"],
            file_name=f"{procedure['title'].replace(' ', '_')}.pdf",
            mime="application/pdf",
            help="Download this procedure explanation as a PDF document"
        )

# Add footer with guidance
st.markdown("""
---
**Usage Tips:**
- Upload official Bharat Stage VI emission norms documents for detailed information
- Use the Chat Assistant for document-specific queries
- Check the Compliance Lookup tool for quick reference to requirements
- Browse the Standards Glossary for definitions of technical terms
""")