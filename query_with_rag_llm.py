

import streamlit as st
# Configure the Streamlit page
st.set_page_config(
    page_title="Emission Regulation Assistant",
    layout="wide"
)
import os
import uuid
import gc
import time
import re
import logging
from datetime import datetime
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call this before initializing embed_model
set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LlamaIndex imports
from llama_index.core import (
    Settings, 
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import data
from data_source import EMISSION_GLOSSARY, COMPLIANCE_REQUIREMENTS, TESTING_PROCEDURES

# Environment settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CACHE_DIR = os.environ.get("CACHE_DIR", "./.cache")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:1b")  # Use model available in your system

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Standard prompt template
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

# Try to import domain knowledge
try:
    from domain_knowledge import (
        EMISSION_CONTROL_SYSTEMS,
        FUEL_TYPES,
        VEHICLE_CATEGORIES,
        match_inference_rule,
        get_system_compatibility,
        get_emission_limits,
        is_system_required
    )
    from enhanced_prompt import (
        ENHANCED_AUTO_STANDARD_PROMPT,
        enhance_query_with_domain_knowledge,
        enhance_response_with_domain_knowledge
    )
    DOMAIN_KNOWLEDGE_AVAILABLE = True
    logger.info("Domain knowledge modules loaded successfully")
except ImportError as e:
    DOMAIN_KNOWLEDGE_AVAILABLE = False
    logger.warning(f"Domain knowledge modules not available: {e}")
    
    # Create empty placeholder functions and data to prevent errors
    EMISSION_CONTROL_SYSTEMS = {}
    FUEL_TYPES = {}
    VEHICLE_CATEGORIES = {}
    ENHANCED_AUTO_STANDARD_PROMPT = AUTO_STANDARD_PROMPT
    
    def match_inference_rule(query): 
        return None
        
    def get_system_compatibility(system_name, fuel_type): 
        return None
        
    def get_emission_limits(vehicle_category, fuel_type): 
        return None
        
    def is_system_required(system_code, vehicle_category, standard="bs_vi"): 
        return None
        
    def enhance_query_with_domain_knowledge(query): 
        return query, False, None
        
    def enhance_response_with_domain_knowledge(query, response): 
        return response

# Initialize models (cached)
@st.cache_resource
def get_models(model_name=DEFAULT_MODEL_NAME):
    """Initialize and cache the LLM and embedding models."""
    try:
        # Set seed for embedding model consistency
        import torch, numpy as np, random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        llm = Ollama(
            model=model_name, 
            request_timeout=120.0,
            temperature=0.0,  # 🔒 lock randomness
            num_ctx=4096,
            base_url=OLLAMA_URL
        )
        
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", 
            trust_remote_code=True,
            embed_batch_size=32,
            max_length=512
        )
        
        # Update global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        return llm, embed_model
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        st.error(f"Error initializing models: {e}")
        return None, None

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0
    if "last_model" not in st.session_state:
        st.session_state.last_model = DEFAULT_MODEL_NAME
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = 10
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False
    if "expanded_section" not in st.session_state:
        st.session_state.expanded_section = None
    if "vehicle_type" not in st.session_state:
        st.session_state.vehicle_type = ""
    if "fuel_type" not in st.session_state:
        st.session_state.fuel_type = ""
    if "auto_loaded" not in st.session_state:
        st.session_state.auto_loaded = False
    if "domain_knowledge_enabled" not in st.session_state:
        st.session_state.domain_knowledge_enabled = DOMAIN_KNOWLEDGE_AVAILABLE

def find_most_recent_session():
    """Find the most recently created index session"""
    try:
        cache_items = os.listdir(CACHE_DIR) if os.path.exists(CACHE_DIR) else []
        index_dirs = [d for d in cache_items if os.path.isdir(os.path.join(CACHE_DIR, d)) and "-" in d]
        
        if not index_dirs:
            return None
        
        # Sort by creation time, newest first
        newest_dir = max(index_dirs, key=lambda d: os.path.getctime(os.path.join(CACHE_DIR, d)))
        
        # Extract session ID (everything before the first dash)
        if "-" in newest_dir:
            return newest_dir.split("-")[0]
        return None
    except Exception as e:
        logger.error(f"Error finding most recent session: {e}")
        return None

def load_indexed_documents(session_id):
    """Load pre-indexed documents from storage based on session ID."""
    if not session_id:
        return None
        
    try:
        # Look for any directories starting with the session_id
        session_dirs = [d for d in os.listdir(CACHE_DIR) if d.startswith(f"{session_id}-")]
        
        if not session_dirs:
            st.warning(f"No indexed documents found for session ID: {session_id}")
            return None
        
        # Sort by creation time, newest first
        session_dirs.sort(key=lambda d: os.path.getctime(os.path.join(CACHE_DIR, d)), reverse=True)
        
        # Use the most recent directory
        index_path = os.path.join(CACHE_DIR, session_dirs[0])
        
        st.info(f"Loading index from {index_path}")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        st.error(f"Error loading index: {e}")
        logger.error(f"Error loading index: {e}")
        return None

def create_query_engine(index, similarity_top_k=10, debug_mode=False):
    """Create a query engine with automotive-specific prompt template."""
    try:
        # Choose prompt template based on domain knowledge availability
        if st.session_state.domain_knowledge_enabled:
            auto_prompt_tmpl = PromptTemplate(ENHANCED_AUTO_STANDARD_PROMPT)
            logger.info("Using enhanced prompt template with domain knowledge")
        else:
            auto_prompt_tmpl = PromptTemplate(AUTO_STANDARD_PROMPT)
            logger.info("Using standard prompt template")
        
        # Create query engine with optimized settings for finding specific standards
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=[],
            verbose=debug_mode
        )
        
        # Update with automotive-specific prompt
        query_engine.update_prompts({"response_synthesizer:text_qa_template": auto_prompt_tmpl})
        
        return query_engine
    except Exception as e:
        logger.error(f"Error creating query engine: {e}")
        return None

def generate_response(query_engine, prompt, debug_mode=False):
    """Generate a response from the query engine with domain knowledge enhancement."""
    try:
        # Apply domain knowledge if available
        if st.session_state.domain_knowledge_enabled:
            # Pre-process query with domain knowledge
            try:
                enhanced_query, direct_match, inference_answer = enhance_query_with_domain_knowledge(prompt)
                
                # Log the enhancement
                if enhanced_query != prompt:
                    logger.info(f"Original query: {prompt}")
                    logger.info(f"Enhanced query: {enhanced_query}")
                
                # If we have a direct inference match, return it immediately
                if direct_match:
                    logger.info(f"Direct inference match found")
                    return inference_answer, []
                
                # Use enhanced query for execution
                used_prompt = enhanced_query
            except Exception as e:
                logger.error(f"Error enhancing query: {e}")
                used_prompt = prompt
        else:
            # Use original prompt if domain knowledge not available
            used_prompt = prompt
        
        # Execute query
        response = query_engine.query(used_prompt)
        response_text = str(response)
        source_docs = []
        
        # Apply domain knowledge post-processing if available
        if st.session_state.domain_knowledge_enabled:
            # Post-process response with domain knowledge
            try:
                enhanced_response = enhance_response_with_domain_knowledge(prompt, response_text)
                
                # Log if enhancement was applied
                if enhanced_response != response_text:
                    logger.info("Response was enhanced with domain knowledge")
                    
                # Use the enhanced response
                final_response = enhanced_response
            except Exception as e:
                logger.error(f"Error enhancing response: {e}")
                final_response = response_text
        else:
            # Use the original response if domain knowledge not available
            final_response = response_text
        
        # Add source information if available
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                    source_docs.append(node.metadata['file_name'])
                    
                    # Debug information
                    if debug_mode and hasattr(node, 'score'):
                        score_info = f"Match score: {node.score:.4f} - {node.metadata.get('file_name', 'unknown')}"
                        st.sidebar.write(score_info)
                        logger.debug(score_info)
            
            if source_docs:
                unique_sources = list(set(source_docs))
                source_info = "\n\n**Sources:** " + ", ".join(unique_sources)
                final_response += source_info
        
        return final_response, source_docs
    except Exception as e:
        error_msg = f"Error generating response: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg, []

def query_with_fallback(query_engine, prompt, emission_glossary, debug_mode=False):
    """Query with fallback to domain knowledge if no index available or no relevant info found."""
    # If domain knowledge is available, check for direct match first
    if st.session_state.domain_knowledge_enabled:
        try:
            # Check for direct domain knowledge matches regardless of index
            _, direct_match, inference_answer = enhance_query_with_domain_knowledge(prompt)
            if direct_match:
                logger.info("Using direct domain knowledge answer")
                return inference_answer, []
        except Exception as e:
            logger.error(f"Error checking domain knowledge: {e}")
    
    # If no query engine is available, use fallbacks
    if query_engine is None:
        logger.info("No query engine available, using fallbacks")
        
        # Domain knowledge fallback if available
        if st.session_state.domain_knowledge_enabled:
            try:
                # Check for domain-specific keywords and patterns in the query
                system_pattern = r'(?i)\b(TWC|DOC|DPF|SCR|EGR|EVP|OBD|PCV)\b'
                fuel_pattern = r'(?i)\b(diesel|petrol|gasoline|cng|lpg)\b'
                vehicle_pattern = r'(?i)\b(L[1-5]|M[1-3]|N[1-3])\b'
                
                system_match = re.search(system_pattern, prompt)
                fuel_match = re.search(fuel_pattern, prompt)
                vehicle_match = re.search(vehicle_pattern, prompt)
                
                # If query involves specific emission system and fuel type compatibility
                if system_match and fuel_match:
                    system_code = system_match.group(1).upper()
                    fuel_type = fuel_match.group(1).capitalize()
                    if fuel_type.lower() == "gasoline":
                        fuel_type = "Petrol"
                    
                    # Set a default vehicle type if not matched
                    vehicle_type = vehicle_match.group(1).upper() if vehicle_match else "M1"
                    
                    compatibility = get_system_compatibility(system_code, fuel_type)
                    if compatibility:
                        if compatibility["compatible"]:
                            response_text = (f"{compatibility['system']} ({system_code}) **IS applicable** to {fuel_type} vehicles. "
                                            f"{compatibility['description']}. {compatibility['notes']}")
                        else:
                            # Try domain knowledge if available
                            if st.session_state.domain_knowledge_enabled and DOMAIN_KNOWLEDGE_AVAILABLE:
                                try:
                                    # Try to get emission limits from domain knowledge
                                    limits = get_emission_limits(vehicle_type, fuel_type)
                                    if limits:
                                        # Create a formatted response with emission limits
                                        response_text = f"BS VI emission limits for {vehicle_type} {fuel_type} vehicles:\n\n"
                                        limits_text = ""
                                        for pollutant, limit in limits.items():
                                            if pollutant != "test_procedure":
                                                limits_text += f"- **{pollutant}**: {limit}\n"
                                        
                                        response_text += limits_text
                                        
                                        if "test_procedure" in limits:
                                            response_text += f"\nTest procedure: {limits['test_procedure']}"
                                    else:
                                        response_text = "No specific emission limits found for the given vehicle and fuel type."
                                    
                                    # Add note about source
                                    response_text += "\n\n*Note: This information is from embedded domain knowledge.*"
                                
                                except Exception as e:
                                    logger.error(f"Error retrieving domain knowledge emission limits: {e}")
                                    response_text = "Unable to retrieve emission limits due to an error."
                            else:
                                response_text = "No domain knowledge available for emission limits."
                            
                            # Compatibility response
                            response_text += f"\n\n{compatibility['system']} ({system_code}) **IS NOT applicable** to {fuel_type} vehicles. " \
                                            f"{compatibility['description']}. {compatibility['notes']}"
                            
                            return response_text, []
                
                # If query involves specific emission system requirements for vehicle category
                if system_match and vehicle_match:
                    system_code = system_match.group(1).upper()
                    vehicle_category = vehicle_match.group(1).upper()
                    
                    requirement = is_system_required(system_code, vehicle_category)
                    if requirement:
                        status_text = "required" if requirement["required"] else "not required"
                        response_text = (f"{requirement['system']} ({system_code}) is **{status_text}** for {vehicle_category} "
                                        f"category vehicles under BS VI norms.")
                        return response_text, []
                
                # If query involves emission limits for specific vehicle and fuel type
                if vehicle_match and fuel_match:
                    vehicle_category = vehicle_match.group(1).upper()
                    fuel_type = fuel_match.group(1).capitalize()
                    if fuel_type.lower() == "gasoline":
                        fuel_type = "Petrol"
                        
                    limits = get_emission_limits(vehicle_category, fuel_type)
                    if limits:
                        response_text = f"BS VI emission limits for {vehicle_category} {fuel_type} vehicles:\n\n"
                        for pollutant, limit in limits.items():
                            if pollutant != "test_procedure":
                                response_text += f"- {pollutant}: {limit}\n"
                        if "test_procedure" in limits:
                            response_text += f"\nTest procedure: {limits['test_procedure']}"
                        return response_text, []
            
            except Exception as e:
                logger.error(f"Error using domain knowledge fallback: {e}")
        
        # Glossary fallback
        related_terms = [item for item in emission_glossary if item["term"].lower() in prompt.lower()]
        if related_terms:
            response_text = f"Here's information about {related_terms[0]['term']}:\n\n{related_terms[0]['definition']}"
            return response_text, []
        
        # Final fallback message
        if st.session_state.domain_knowledge_enabled:
            response_text = ("Based on my domain knowledge of automotive emission systems, I cannot provide a specific answer to this query. "
                           "For more detailed information, please upload relevant BS VI emission standards documents "
                           "or rephrase your question to focus on specific emission systems, vehicle categories, or fuel types.")
        else:
            response_text = "No indexed documents are loaded. Please run the indexing app first to process your documents."
        
        return response_text, []
    
    # Use the query engine for response generation
    return generate_response(query_engine, prompt, debug_mode)

def auto_load_index():
    """Automatically find and load the most recent index"""
    if st.session_state.auto_loaded:
        return

    # Find the most recent session
    session_id = find_most_recent_session()
    if not session_id:
        st.warning("No indexed documents found. Please run the indexing app first.")
        return
    
    st.session_state.session_id = session_id
    
    # Initialize models
    llm, embed_model = get_models(st.session_state.last_model)
    if llm is None or embed_model is None:
        st.error("Failed to initialize models. Please check your Ollama connection.")
        return
    
    # Load the index
    with st.spinner(f"Automatically loading the most recent index (Session ID: {session_id})..."):
        index = load_indexed_documents(session_id)
        
        if index is not None:
            st.session_state.index = index
            
            # Create query engine
            query_engine = create_query_engine(
                index, 
                similarity_top_k=st.session_state.retrieval_k, 
                debug_mode=st.session_state.debug_mode
            )
            
            if query_engine is None:
                st.error("Failed to create query engine.")
                return
                
            st.session_state.query_engine = query_engine
            st.session_state.index_loaded = True
            st.session_state.auto_loaded = True
            
            st.success(f"Successfully loaded index from session: {session_id}")
        else:
            st.session_state.index = None
            st.session_state.query_engine = None
            st.session_state.index_loaded = False

def main():
    # Initialize session state
    initialize_session_state()
    
    # App title and header
    st.title("🔍 Emission Regulation Assistant")
    st.write("Query indexed automotive standards documents for emission regulations and testing requirements.")
    
    # Auto-load the most recent index
    auto_load_index()
    
    # Sidebar for settings and controls
    with st.sidebar:
        st.header("Settings")
        
        # Domain knowledge toggle
        if DOMAIN_KNOWLEDGE_AVAILABLE:
            st.session_state.domain_knowledge_enabled = st.checkbox(
                "Enable Domain Intelligence", 
                value=True,
                help="Use specialized automotive emission knowledge for more accurate answers"
            )
            if st.session_state.domain_knowledge_enabled:
                st.success("✅ Domain Intelligence: Active")
            else:
                st.info("❌ Domain Intelligence: Disabled")
        else:
            st.warning("⚠️ Domain Intelligence modules not found")
        
        # Session ID information (read-only)
        if st.session_state.session_id:
            st.success(f"Using Session ID: {st.session_state.session_id}")
        else:
            st.warning("No session currently loaded")
        
        # Model selection
        try:
            import requests
            response = requests.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                if not available_models:
                    available_models = ["llama3.2:1b", "llama3:8b", "llama3.2:latest"]
            else:
                available_models = ["llama3.2:1b", "llama3:8b", "llama3.2:latest"]
        except Exception:
            available_models = ["llama3.2:1b", "llama3:8b", "llama3.2:latest"]
        
        selected_model = st.selectbox("Select Model", available_models, index=0)
        
        # Update model if changed
        if selected_model != st.session_state.last_model:
            st.session_state.last_model = selected_model
            
            if st.button("Apply Model Change"):
                llm, embed_model = get_models(selected_model)
                
                if st.session_state.index is not None and llm is not None and embed_model is not None:
                    st.session_state.query_engine = create_query_engine(
                        st.session_state.index,
                        similarity_top_k=st.session_state.retrieval_k,
                        debug_mode=st.session_state.debug_mode
                    )
                    st.success(f"Updated to model: {selected_model}")
        
        # Manual session selection (optional, for advanced users)
        with st.expander("Advanced Session Management", expanded=False):
            session_input = st.text_input(
                "Session ID (optional)", 
                value=st.session_state.session_id,
                help="Enter a specific session ID if you don't want to use the automatic one"
            )
            
            if st.button("Load Specific Session"):
                if session_input and session_input != st.session_state.session_id:
                    st.session_state.session_id = session_input
                    
                    # Load the index
                    with st.spinner(f"Loading session: {session_input}..."):
                        index = load_indexed_documents(session_input)
                        
                        if index is not None:
                            st.session_state.index = index
                            
                            # Create query engine
                            query_engine = create_query_engine(
                                index, 
                                similarity_top_k=st.session_state.retrieval_k, 
                                debug_mode=st.session_state.debug_mode
                            )
                            
                            if query_engine is not None:
                                st.session_state.query_engine = query_engine
                                st.session_state.index_loaded = True
                                st.session_state.auto_loaded = True
                                
                                st.success(f"Successfully loaded session: {session_input}")
                            else:
                                st.error("Failed to create query engine.")
                        else:
                            st.error(f"Failed to load session: {session_input}")
        
        # Advanced settings
        with st.expander("Query Settings", expanded=False):
            # Debug mode toggle
            st.session_state.debug_mode = st.checkbox("Debug Mode", value=False, 
                                                    help="Show match scores and retrieval details")
            
            # Configure retrieval parameters
            retrieval_k = st.slider("Retrieval Context Count", 5, 15, 10, 
                                  help="Number of text chunks to retrieve from documents")
            
            # Update retrieval parameter if changed
            if st.session_state.retrieval_k != retrieval_k:
                st.session_state.retrieval_k = retrieval_k
                
                # Update query engine if we have an index
                if st.session_state.index is not None:
                    with st.spinner("Updating retrieval settings..."):
                        st.session_state.query_engine = create_query_engine(
                            st.session_state.index,
                            similarity_top_k=retrieval_k,
                            debug_mode=st.session_state.debug_mode
                        )
                        st.success("Settings updated")
        
        # Clear button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            gc.collect()
            st.rerun()
        
        # Link to indexing app  
        st.markdown("---")
        st.info("Need to index new documents? Run the indexing app.")

    # Main tabs
    tabs = st.tabs(["💬 Chat Assistant", "🔍 Compliance Lookup", "📘 Standards Glossary", "📋 Testing Procedures"])
    
    # Tab 1: Chat Interface
    with tabs[0]:
        # Show status of indexed documents
        if st.session_state.index_loaded:
            st.success(f"📚 Using indexed documents from session: {st.session_state.session_id}")
            
            # Show domain intelligence status
            if st.session_state.domain_knowledge_enabled and DOMAIN_KNOWLEDGE_AVAILABLE:
                st.info("🧠 Domain Intelligence: Active - Specialized automotive knowledge is being used to enhance responses")
        else:
            st.warning("⚠️ No indexed documents loaded. Please run the indexing app first to process documents.")
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("Ask about Bharat Stage VI standards, test methods, or compliance requirements..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
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
                        # Generate response
                        response_text, source_docs = query_with_fallback(
                            st.session_state.query_engine,
                            prompt,
                            EMISSION_GLOSSARY,
                            st.session_state.debug_mode
                        )
                        
                        response_placeholder.markdown(response_text)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                with st.chat_message("assistant"):
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    logger.error(error_msg, exc_info=True)
                
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
    
    # Tab 2: Compliance Lookup
    with tabs[1]:
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
                    
                    # If domain knowledge is available, add emission control systems info
                    if st.session_state.domain_knowledge_enabled and DOMAIN_KNOWLEDGE_AVAILABLE:
                        try:
                            # Try to get required emission control systems
                            with st.expander("Required Emission Control Systems"):
                                st.write("Based on domain knowledge, this vehicle category requires these emission control systems:")
                                
                                systems_info = []
                                for system_code, system_info in EMISSION_CONTROL_SYSTEMS.items():
                                    applicable_fuels = system_info.get("applicable_fuels", [])
                                    if fuel_type in applicable_fuels:
                                        requirement = is_system_required(system_code, vehicle_type)
                                        if requirement and requirement.get("required", False):
                                            systems_info.append({
                                                "system": system_info["full_name"],
                                                "code": system_code,
                                                "description": system_info["description"]
                                            })
                                
                                if systems_info:
                                    for system in systems_info:
                                        st.write(f"- **{system['system']}** ({system['code']}): {system['description']}")
                                else:
                                    st.write("No specific emission control systems identified for this vehicle category and fuel type combination.")
                        except Exception as e:
                            logger.error(f"Error showing emission control systems: {e}")
                    
                    # Show suggestion to view related procedures
                    st.info("ℹ️ View the Testing Procedures tab for detailed explanations of the test methods referenced above.")
                        # Tab 3: Standards Glossary
    with tabs[2]:
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
    with tabs[3]:
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
        if st.session_state.expanded_section and st.session_state.expanded_section in TESTING_PROCEDURES:
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

    # Footer
    st.markdown("---")
    if st.session_state.domain_knowledge_enabled and DOMAIN_KNOWLEDGE_AVAILABLE:
        st.markdown("""
        **Usage Tips:**
        - The app automatically loads the most recent indexed documents
        - Domain Intelligence provides accurate answers about emission systems and vehicle compatibility
        - Use the Chat Assistant for technical queries about emission standards
        - Check the Compliance Lookup tool for specific vehicle requirements
        - Browse the Standards Glossary for definitions of technical terms
        
        **Domain Intelligence Highlights:**
        - Automatically identifies compatibility between emission systems and fuel types
        - Provides correct answers about EVP, DPF, SCR and other emission systems
        - Enhanced understanding of vehicle categories and emission limits
        """)
    else:
        st.markdown("""
        **Usage Tips:**
        - The app automatically loads the most recent indexed documents
        - Use the Chat Assistant for document-specific queries
        - Check the Compliance Lookup tool for quick reference to requirements
        - Browse the Standards Glossary for definitions of technical terms
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
                    
                    
                    

# --- Domain knowledge handler integration ---
from domain_knowledge_updated import EMISSION_LIMITS, RDE_REQUIREMENTS, OBD_REQUIREMENTS, TEST_CYCLES, IMPLEMENTATION_DATES

def rule_based_handler(query):
    query_lower = query.lower()

    # Check for emission limits
    if "limit" in query_lower:
        for cat in EMISSION_LIMITS["BS-VI"]:
            if cat.lower() in query_lower:
                for pol in EMISSION_LIMITS["BS-VI"][cat]:
                    if pol.lower() in query_lower:
                        val = EMISSION_LIMITS["BS-VI"][cat][pol]
                        return f"The BS-VI {pol} limit for category {cat} vehicles is {val}."
    
    # RDE check
    if "rde" in query_lower:
        for cat in RDE_REQUIREMENTS["BS-VI"]:
            if cat.lower() in query_lower:
                data = RDE_REQUIREMENTS["BS-VI"][cat]
                factors = data.get("conformity_factors", {})
                return (
                    f"RDE is applicable to category {cat} vehicles starting from {data['start_date']}. "
                    f"Conformity factors: " +
                    ", ".join([f"{k} = {v}" for k, v in factors.items()])
                )

    # OBD check
    if "obd" in query_lower:
        for stage in OBD_REQUIREMENTS["BS-VI"]:
            if stage.lower() in query_lower:
                data = OBD_REQUIREMENTS["BS-VI"][stage]
                return f"{stage} became applicable on {data['start_date']} and monitors: {', '.join(data['monitored_components'])}."

    # Test cycle check
    if "test cycle" in query_lower:
        for cat in TEST_CYCLES["BS-VI"]:
            if cat.lower() in query_lower:
                return f"The BS-VI test cycle for category {cat} is {TEST_CYCLES['BS-VI'][cat]}."

    # Implementation date check
    if "bs-vi" in query_lower and "start" in query_lower:
        return f"BS-VI norms were implemented for all vehicles on {IMPLEMENTATION_DATES['BS-VI']['all_vehicles']}."

    # Fallback
    return None


# Enhanced intelligent answer handler
def enhanced_rule_based_handler(query):
    query_lower = query.lower()

    if "diesel" in query_lower and "mass emission" in query_lower:
        return (
            "Under BS-VI norms, mass emission limits for diesel vehicles depend on the test type:\n\n"
            "- For **engine-level testing** (using WHSC cycle), key limits are:\n"
            "  - CO: 1500 mg/kWh\n"
            "  - NOx: 400 mg/kWh\n"
            "  - PM: 10 mg/kWh\n"
            "  - PN: 8.0x10¹¹ #/kWh\n\n"
            "- For **vehicle-level testing** (e.g., Light Commercial Vehicles):\n"
            "  - PM: 0.005 g/km\n"
            "  - NOx: 0.08 g/km\n"
            "  - CO: 0.5 g/km\n\n"
            "These limits are defined in AIS-137 Part 4 (Clause 5.3) and notified via GSR 889(E)."
        )

    return rule_based_handler(query)


def intelligent_mass_emission_summary(query):
    query_lower = query.lower()

    diesel_keywords = ["diesel"]
    petrol_keywords = ["petrol", "gasoline"]
    engine_level_keywords = ["engine", "mg/kwh", "whsc", "whtc", "mass emission"]
    vehicle_level_keywords = ["vehicle", "g/km", "pm limit", "n1", "m1", "bs-vi", "light commercial"]

    is_diesel = any(k in query_lower for k in diesel_keywords)
    is_petrol = any(k in query_lower for k in petrol_keywords)
    is_engine_level = any(k in query_lower for k in engine_level_keywords)
    is_vehicle_level = any(k in query_lower for k in vehicle_level_keywords)

    if is_diesel and is_engine_level:
        return (
            "Under BS-VI norms, mass emission limits for diesel engines tested under the WHSC cycle are:\n"
            "- CO: 1500 mg/kWh\n"
            "- NOx: 400 mg/kWh\n"
            "- PM: 10 mg/kWh\n"
            "- PN: 8.0×10¹¹ #/kWh\n\n"
            "These values apply to heavy-duty diesel engine testing and are specified in AIS-137 Part 4, Table 3."
        )

    if is_diesel and is_vehicle_level:
        return (
            "For BS-VI vehicle-level testing of diesel Light Commercial Vehicles (N1 category), the mass emission limits are:\n"
            "- PM: 0.005 g/km\n"
            "- NOx: 0.08 g/km\n"
            "- CO: 0.5 g/km\n\n"
            "These limits apply under the World Harmonized Light-duty Test Procedure (WLTP) and are defined in AIS-137."
        )

    if is_petrol and is_vehicle_level:
        return (
            "Under BS-VI, petrol vehicle mass emission limits (e.g., M1 category) are:\n"
            "- PM: 0.0045 g/km\n"
            "- NOx: 0.06 g/km\n"
            "- CO: 1.0 g/km\n\n"
            "These are tested under WLTP and vary slightly by vehicle sub-category."
        )

    return (
        "BS-VI emission limits vary by vehicle category, fuel type, and test cycle.\n\n"
        "- For engine-level testing of diesel vehicles (WHSC cycle):\n"
        "  - CO: 1500 mg/kWh\n"
        "  - NOx: 400 mg/kWh\n"
        "  - PM: 10 mg/kWh\n"
        "  - PN: 8.0×10¹¹ #/kWh\n\n"
        "- For vehicle-level testing (e.g., N1 LCVs):\n"
        "  - PM: 0.005 g/km\n"
        "  - NOx: 0.08 g/km\n"
        "  - CO: 0.5 g/km\n\n"
        "Refer to AIS-137 Part 4 and GSR 889(E) for detailed tables and compliance thresholds."
    )


def classify_and_answer(query):
    query_lower = query.lower()

    # Handle mass emission queries first
    if "mass emission" in query_lower or "mg/kwh" in query_lower:
        return intelligent_mass_emission_summary(query)

    # Handle PM/NOx/CO limits for known categories
    if "limit" in query_lower and any(cat in query_lower for cat in ["n1", "m1", "n2", "n3"]):
        return rule_based_handler(query)

    # Handle RDE requirement queries
    if "rde" in query_lower:
        return rule_based_handler(query)

    # Handle OBD compliance questions
    if "obd" in query_lower:
        return rule_based_handler(query)

    # Handle test cycle or cycle type
    if "test cycle" in query_lower:
        return rule_based_handler(query)

    # Handle BS-VI start date or implementation timeline
    if "bs-vi" in query_lower and ("start" in query_lower or "applicable" in query_lower or "implementation" in query_lower):
        return rule_based_handler(query)

    # Handle fallback queries
    return None  # signal that fallback to RAG or LLM is needed



from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import openai
import json

# Initialize Qdrant and embedding model
qdrant = QdrantClient(host="localhost", port=6333)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "emission_docs"

# Load few-shot examples
def load_few_shot_examples(jsonl_path="emission_prompted_qa_full.jsonl"):
    examples = []
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                examples.append(json.loads(line))
    except:
        pass
    return examples

few_shot_examples = load_few_shot_examples()

# Retrieve top-k similar document chunks from Qdrant
def retrieve_context(query, top_k=3):
    query_vector = embedding_model.encode(query).tolist()
    hits = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=top_k,
    with_payload=True,
    with_vectors=False,
    exact=True  # if your setup supports it (Qdrant ≥ v1.3)
)

    return "\n".join(hit.payload.get("text", "") for hit in hits)

# Build LLM prompt using few-shot examples + retrieved context
def build_prompt_with_examples(user_query, examples, context):
    prompt = "You are an expert on Bharat Stage VI emission regulations. Use the examples and context to answer user queries."
    for ex in examples[:3]:  # limit to 3 for token efficiency
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    if context:
        prompt += f"Context:"

    prompt += f"Q: {user_query}\nA:"
    return prompt

# Call LLM using OpenAI (replace with local model if needed)
def call_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # replace with llama3 or others if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response['choices'][0]['message']['content'].strip()

# Final query handler
def handle_query(user_query):
    answer = classify_and_answer(user_query)
    if answer:
        return answer

    # fallback: use RAG + LLM prompting
    context = retrieve_context(user_query)
    prompt = build_prompt_with_examples(user_query, few_shot_examples, context)
    return call_llm(prompt)
