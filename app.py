import streamlit as st
import tempfile
import os
from pathlib import Path
import PyPDF2
from io import BytesIO

# Simple text search using TF-IDF (no complex dependencies)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    st.error("Please install required packages: scikit-learn")

# Optional: OpenAI for better responses (falls back to simple retrieval if not available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Simple WMA RAG App",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Simple WMA RAG App")
st.markdown("Upload PDFs, ask questions, get answers with optional summaries!")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorizer' not in st.session_state:
    if SEARCH_AVAILABLE:
        st.session_state.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    else:
        st.session_state.vectorizer = None
if 'doc_vectors' not in st.session_state:
    st.session_state.doc_vectors = None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def create_document_vectors(chunks):
    """Create TF-IDF vectors for text chunks"""
    if not st.session_state.vectorizer or not chunks:
        return None
    
    try:
        vectors = st.session_state.vectorizer.fit_transform(chunks)
        return vectors
    except Exception as e:
        st.error(f"Error creating document vectors: {str(e)}")
        return None

def find_relevant_chunks(query, top_k=3):
    """Find most relevant chunks for the query using TF-IDF"""
    if st.session_state.doc_vectors is None or not st.session_state.vectorizer:
        return []
    
    try:
        # Transform query using the fitted vectorizer
        query_vector = st.session_state.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, st.session_state.doc_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                relevant_chunks.append({
                    'text': st.session_state.documents[idx],
                    'similarity': similarities[idx]
                })
        return relevant_chunks
    except Exception as e:
        st.error(f"Error finding relevant chunks: {str(e)}")
        return []

def generate_answer(query, context_chunks):
    """Generate answer using OpenAI or simple concatenation"""
    if not context_chunks:
        return "No relevant information found in the uploaded documents."
    
    # Combine relevant chunks
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
    # Check if OpenAI is available and configured
    openai_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') else os.getenv("OPENAI_API_KEY")
    
    if OPENAI_AVAILABLE and openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If the answer isn't in the context, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"OpenAI error: {str(e)}. Falling back to simple retrieval.")
    
    # Fallback: Simple context return
    return f"Based on the uploaded documents:\n\n{context[:1000]}{'...' if len(context) > 1000 else ''}"

def summarize_text(text):
    """Summarize text using OpenAI or simple truncation"""
    openai_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') else os.getenv("OPENAI_API_KEY")
    
    if OPENAI_AVAILABLE and openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the following text in 2-3 sentences."},
                    {"role": "user", "content": text[:3000]}  # Limit input length
                ],
                max_tokens=150,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Summarization error: {str(e)}. Providing simple summary.")
    
    # Fallback: Simple truncation
    sentences = text.split('. ')
    return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # OpenAI API Key input
    api_key = st.text_input("OpenAI API Key (optional)", type="password", help="For better responses and summaries")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("✅ PDF Upload")
    st.markdown("✅ Question Answering")
    st.markdown("✅ Text Summarization")
    st.markdown("✅ Works without OpenAI")

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            if not SEARCH_AVAILABLE:
                st.error("Text search not available. Please check requirements.")
                st.stop()
            
            with st.spinner("Processing documents..."):
                # Clear previous data
                st.session_state.documents = []
                st.session_state.doc_vectors = None
                
                # Process each file
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: {uploaded_file.name}")
                    
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    if text.strip():
                        # Chunk text
                        chunks = chunk_text(text)
                        st.session_state.documents.extend(chunks)
                        st.success(f"✅ {uploaded_file.name} processed ({len(chunks)} chunks)")
                    else:
                        st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                
                # Create document vectors for all chunks
                if st.session_state.documents:
                    st.session_state.doc_vectors = create_document_vectors(st.session_state.documents)
                    st.success(f"🎉 Processing complete! Total chunks: {len(st.session_state.documents)}")

with col2:
    st.header("❓ Ask Questions")
    
    if st.session_state.documents:
        st.success(f"📚 {len(st.session_state.documents)} text chunks ready for questions")
        
        # Question input
        question = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        col_ask, col_summarize = st.columns(2)
        
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary")
        
        with col_summarize:
            summarize_button = st.button("📝 Summarize All")
        
        if ask_button and question:
            with st.spinner("Finding answer..."):
                # Find relevant chunks
                relevant_chunks = find_relevant_chunks(question)
                
                if relevant_chunks:
                    # Generate answer
                    answer = generate_answer(question, relevant_chunks)
                    
                    st.subheader("💡 Answer")
                    st.write(answer)
                    
                    # Show sources
                    with st.expander("📄 Sources", expanded=False):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"**Source {i+1}** (Similarity: {chunk['similarity']:.3f})")
                            st.write(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                            st.write("---")
                    
                    # Optional summarization
                    if st.button("📝 Summarize this answer"):
                        with st.spinner("Summarizing..."):
                            summary = summarize_text(answer)
                            st.subheader("📋 Summary")
                            st.info(summary)
                else:
                    st.warning("No relevant information found for your question.")
        
        if summarize_button:
            with st.spinner("Creating summary of all documents..."):
                # Combine all documents
                all_text = " ".join(st.session_state.documents[:10])  # Limit to first 10 chunks
                summary = summarize_text(all_text)
                
                st.subheader("📋 Document Summary")
                st.info(summary)
    
    else:
        st.info("👆 Upload and process PDF files first to start asking questions!")

# Footer
st.markdown("---")
st.markdown("**Simple WMA RAG App** - Upload PDFs, ask questions, get answers! 🚀")
