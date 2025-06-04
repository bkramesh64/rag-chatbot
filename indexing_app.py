
import streamlit as st

# Configure the Streamlit page
st.set_page_config(
    page_title="Document Indexing for Emission Regulation Assistant",
    layout="wide"
)

import os
import tempfile
import uuid
import time
import gc
import glob
import base64
from datetime import datetime

# LlamaIndex imports
from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Environment settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CACHE_DIR = os.environ.get("CACHE_DIR", "./.cache")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:8b")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)


# Initialize models (cached)
@st.cache_resource
def get_models(model_name=DEFAULT_MODEL_NAME):
    """Initialize and cache the LLM and embedding models."""
    llm = Ollama(
        model=model_name, 
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
    
    # Update global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

def initialize_session_state():
    """Initialize session state variables"""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "last_model" not in st.session_state:
        st.session_state.last_model = DEFAULT_MODEL_NAME
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 300
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 100
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "index_complete" not in st.session_state:
        st.session_state.index_complete = False
    if "index_directory" not in st.session_state:
        st.session_state.index_directory = ""

def configure_settings(chunk_size=300, chunk_overlap=100):
    """Configure indexing settings"""
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

def process_multi_documents(dir_path, index_id, session_id, force_reindex=False):
    """Process multiple documents and create a combined index"""
    storage_path = os.path.join(CACHE_DIR, f"{session_id}-{index_id}")
    
    # Try to load from existing cache unless force_reindex is True
    if os.path.exists(storage_path) and not force_reindex:
        try:
            st.info(f"Loading cached index from {storage_path}")
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            return load_index_from_storage(storage_context)
        except Exception as e:
            st.warning(f"Could not load cached index: {e}")
            st.info("Will create new index instead.")
    
    # Create new index
    st.info(f"Creating new index for documents in {dir_path}")
    parser = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
        paragraph_separator="\n\n",
        separator=" ",
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    # Load documents
    st.info("Loading documents...")
    loader = SimpleDirectoryReader(
        input_dir=dir_path,
        required_exts=[".pdf"],
        recursive=True,
        file_extractor={"pdf": "default"}
    )
    
    docs = loader.load_data()
    st.success(f"Successfully loaded {len(docs)} document chunks")
    
    # Create index
    st.info("Creating index with vector embeddings...")
    progress_bar = st.progress(0.0)
    
    # Progress simulation for better UX
    for i in range(10):
        time.sleep(0.2)
        progress_bar.progress((i + 1) * 0.1)
        
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[parser],
        show_progress=True
    )
    
    # Save index to disk
    st.info(f"Saving index to {storage_path}")
    index.storage_context.persist(persist_dir=storage_path)
    progress_bar.progress(1.0)
    
    return index

def create_combined_index_id(files_path):
    """Create a unique index ID based on directory path"""
    import hashlib
    dir_hash = hashlib.md5(files_path.encode()).hexdigest()
    return f"multi-{dir_hash[:10]}"

def get_file_list(directory):
    """Get list of PDF files in the directory"""
    file_list = glob.glob(os.path.join(directory, "*.pdf"))
    return [os.path.basename(f) for f in file_list]

def save_uploaded_files(uploaded_files, temp_dir):
    """Save uploaded files to temporary directory and return paths"""
    temp_file_paths = []
    
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        temp_file_paths.append(file_path)
    
    return temp_file_paths

def main():
    # Initialize session state
    initialize_session_state()
    
    # App title and description
    st.title("📚 Document Indexing for Emission Regulation Assistant")
    st.write("Upload and index automotive standards documents for the RAG-based chat assistant.")
    
    # Display existing indexed sessions
    st.sidebar.header("Existing Indexed Sessions")
    existing_indices = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
    
    if existing_indices:
        st.sidebar.write("Previously indexed sessions:")
        for idx in existing_indices:
            parts = idx.split('-')
            if len(parts) >= 2:
                session_id = parts[0]
                # Get creation time
                creation_time = datetime.fromtimestamp(os.path.getctime(os.path.join(CACHE_DIR, idx)))
                # Display in the sidebar
                st.sidebar.write(f"• Session ID: `{session_id}` (Created: {creation_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        st.sidebar.write("No existing indexed sessions found.")
    
    # Create two columns for the main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Documents")
        
        # Session ID input
        session_id = st.text_input(
            "Session ID (you'll need this for querying later)",
            value=st.session_state.session_id
        )
        
        # Model selection
        available_models = ["llama3.2:8b", "llama3.2:70b", "mistral-7b"]
        selected_model = st.selectbox("Select Model", available_models, index=0)
        
        # Update session ID and model if changed
        if session_id != st.session_state.session_id:
            st.session_state.session_id = session_id
        
        if selected_model != st.session_state.last_model:
            st.session_state.last_model = selected_model
            get_models(selected_model)
        
        # Chunking settings
        col_chunk1, col_chunk2 = st.columns(2)
        with col_chunk1:
            chunk_size = st.slider("Chunk Size", 100, 1000, st.session_state.chunk_size, 
                                  help="Size of document chunks for embedding")
        with col_chunk2:
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, st.session_state.chunk_overlap,
                                    help="Overlap between chunks")
        
        # Update settings if changed
        if chunk_size != st.session_state.chunk_size or chunk_overlap != st.session_state.chunk_overlap:
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            configure_settings(chunk_size, chunk_overlap)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload automotive standards documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload Bharat Stage VI related documents"
        )
        
        # Process button
        process_button = st.button(
            "Process and Index Documents", 
            type="primary",
            disabled=not uploaded_files or st.session_state.processing
        )
    
    with col2:
        st.header("Indexing Status")
        
        # Show current settings
        st.subheader("Current Settings")
        st.write(f"• Session ID: `{st.session_state.session_id}`")
        st.write(f"• Model: {st.session_state.last_model}")
        st.write(f"• Chunk Size: {st.session_state.chunk_size}")
        st.write(f"• Chunk Overlap: {st.session_state.chunk_overlap}")
        
        # Show indexed files
        if st.session_state.indexed_files:
            st.subheader("Indexed Files")
            for file in st.session_state.indexed_files:
                st.write(f"✅ {file}")
        
        # Show completion status
        if st.session_state.index_complete:
            st.success("Indexing Complete!")
            st.info(f"Use Session ID `{st.session_state.session_id}` in the Query App")
            
            # Add button to launch query app
            if st.button("Launch Query App"):
                st.info("Please run the Querying App separately with this command:")
                st.code("streamlit run querying_app.py")
    
    # Process documents when button is clicked
    if process_button and uploaded_files:
        st.session_state.processing = True
        st.session_state.index_complete = False
        st.session_state.indexed_files = []
        
        try:
            # Placeholder for output
            status_placeholder = st.empty()
            status_placeholder.info("Starting document processing...")
            
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                temp_file_paths = save_uploaded_files(uploaded_files, temp_dir)
                status_placeholder.info(f"Saved {len(temp_file_paths)} files to temporary directory")
                
                # Store filenames
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                
                # Create index ID
                index_id = create_combined_index_id(temp_dir)
                
                # Get models
                llm, embed_model = get_models(st.session_state.last_model)
                
                # Configure settings
                configure_settings(st.session_state.chunk_size, st.session_state.chunk_overlap)
                
                # Process documents
                index = process_multi_documents(
                    temp_dir, 
                    index_id, 
                    st.session_state.session_id, 
                    force_reindex=True
                )
                
                # Update status
                status_placeholder.success("Indexing completed successfully!")
                
                # Store index directory for reference
                st.session_state.index_directory = os.path.join(CACHE_DIR, f"{st.session_state.session_id}-{index_id}")
                
                # Set completion flag
                st.session_state.index_complete = True
                
                # Force garbage collection
                gc.collect()
        except Exception as e:
            st.error(f"Error during indexing: {e}")
        finally:
            st.session_state.processing = False
    
    # Show document preview if files are uploaded
    if uploaded_files:
        with st.expander("Document Preview", expanded=False):
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

if __name__ == "__main__":
    main()