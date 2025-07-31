# app.py (Final Version: Professional UI + Your Preferred Result Layout)

import os
import re
import streamlit as st
from typing import List, Dict
import numpy as np

# --- NLTK for Sentence Splitting (with robust downloader) ---
try:
    import nltk
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            with st.spinner(f"First-time setup: Downloading NLTK '{resource}' model..."):
                nltk.download(resource, quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# --- Core AI & Data Processing Libraries ---
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# --- File Reading Libraries ---
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# =====================================================================================
# 1. HELPER & DATA PROCESSING FUNCTIONS (Defined First)
# =====================================================================================

def inject_custom_css():
    """Injects custom CSS for a modern, professional, and theme-aware UI."""
    st.markdown("""
        <style>
            /* This CSS is designed for Streamlit's default "dark" theme */
            /* Main App Styling */
            .stApp { background-color: #0E1117; }
            .st-emotion-cache-1y4p8pa { padding: 2rem 3rem; }
            
            /* Sidebar Styling */
            [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #262730; }
            
            /* --- Card Styling --- */
            .result-card {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-left: 5px solid #2F81F7;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }
            .result-card-top {
                background: linear-gradient(to right, #2F81F7, #1F6FEB);
                border-left: 5px solid #58A6FF;
            }
            
            /* --- Text inside cards (YOUR PREFERRED LOGIC'S STYLE) --- */
            .card-header-info {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #30363D;
            }
            .relevance-score { font-size: 1.1rem; font-weight: 600; color: #58A6FF; }
            .metadata { font-size: 0.9rem; color: #8B949E; text-align: right; }
            
            .main-sentence {
                font-size: 1.1rem;
                color: #C9D1D9;
                line-height: 1.6;
                padding: 10px;
                background-color: #0D1117;
                border-radius: 6px;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            .context {
                font-size: 0.9rem;
                color: #8B949E;
                font-style: italic;
                line-height: 1.5;
            }
        </style>
    """, unsafe_allow_html=True)

def clean_and_isolate_narrative(text: str) -> str:
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    try:
        start_index = text.index(start_marker) + len(start_marker)
    except ValueError: start_index = 0
    try:
        end_index = text.index(end_marker, start_index)
    except ValueError: end_index = len(text)
    return text[start_index:end_index]

def create_sentence_data(sentences: List[str], filename: str, location_prefix: str = "", is_pdf: bool = False) -> List[Dict]:
    processed_content = []
    total_sentences = len(sentences)
    for i, sent in enumerate(sentences):
        cleaned_sent = re.sub(r'\s+', ' ', sent).strip()
        if len(cleaned_sent) > 25:
            location = location_prefix if is_pdf else f"Approx. {((i + 1) / total_sentences) * 100:.1f}%"
            processed_content.append({
                "text": cleaned_sent, "source": filename, "location": location,
                "context_before": re.sub(r'\s+', ' ', sentences[i-1]).strip() if i > 0 else "",
                "context_after": re.sub(r'\s+', ' ', sentences[i+1]).strip() if i < total_sentences - 1 else ""
            })
    return processed_content

@st.cache_data(show_spinner="Processing all uploaded documents...")
def process_all_files(uploaded_files: List) -> List[Dict]:
    if not uploaded_files: return []
    all_sentences_data = []
    for file in uploaded_files:
        filename = file.name
        file_extension = os.path.splitext(filename)[1].lower()
        try:
            if file_extension == ".pdf":
                if not PDF_AVAILABLE: st.error("PyPDF2 is not installed."); continue
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    sentences = nltk.sent_tokenize(page_text)
                    all_sentences_data.extend(create_sentence_data(sentences, filename, f"Page {page_num + 1}", is_pdf=True))
            else:
                if file_extension == ".docx":
                    if not DOCX_AVAILABLE: st.error("python-docx is not installed."); continue
                    doc = DocxDocument(file)
                    full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                else:
                    full_text = file.getvalue().decode("utf-8", errors='ignore')
                narrative_text = clean_and_isolate_narrative(full_text)
                sentences = nltk.sent_tokenize(narrative_text)
                all_sentences_data.extend(create_sentence_data(sentences, filename))
        except Exception as e:
            st.error(f"Failed to process '{filename}': {e}")
    return all_sentences_data

@st.cache_resource
def load_semantic_model(model_name='all-MiniLM-L6-v2'):
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        st.error("Sentence Transformers library not found. Please run: pip install sentence-transformers")
        return None
    return SentenceTransformer(model_name)

# =====================================================================================
# 2. SIMILARITY MODEL CLASS (Defined before use)
# =====================================================================================

class SemanticSimilarity:
    def __init__(self):
        self.model = load_semantic_model()
        self.embeddings = None
    def train(self, content_list: List[Dict]):
        if self.model: self.embeddings = self.model.encode([item['text'] for item in content_list], convert_to_tensor=True, show_progress_bar=True)
    def find_similarities(self, query: str) -> np.ndarray:
        if self.embeddings is None or self.model is None: return np.array([])
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        return util.cos_sim(query_embedding, self.embeddings)[0].cpu().numpy()

# =====================================================================================
# 3. MAIN APPLICATION FUNCTION
# =====================================================================================

def main():
    st.set_page_config(page_title="Cognitive Search Engine", layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()

    if "results" not in st.session_state: st.session_state.results = None

    with st.sidebar:
        st.title("Cognitive Search")
        st.markdown("Your intelligent research assistant.")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "1. Upload Your Documents", 
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True
        )
        
        query = st.text_input("2. Enter Your Query", placeholder="e.g., the morality of escaping society")
        search_button = st.button("3. Search Documents", type="primary", use_container_width=True)

    st.header("Search Dashboard")

    all_content = process_all_files(uploaded_files)

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Documents Uploaded", len(uploaded_files))
    col2.metric("📊 Sentences Processed", f"{len(all_content):,}")
    status = "Ready to Search" if uploaded_files else "Awaiting Documents"
    if 'search_button' in locals() and search_button and all_content and not query:
        st.warning("Please enter a query to start searching.")
    col3.metric("💡 Status", status)
    st.markdown("---")

    if search_button and query:
        if not all_content:
            st.error("Please upload at least one document to search.")
        else:
            with st.spinner(f"Running semantic search for '{query}'..."):
                model = SemanticSimilarity()
                model.train(all_content)
                scores = model.find_similarities(query)
                st.session_state.results = {"query": query, "scores": scores, "content": all_content}

    if st.session_state.results:
        query_text = st.session_state.results["query"]
        scores = st.session_state.results["scores"]
        content = st.session_state.results["content"]
        
        st.subheader(f"Search Results for: \"{query_text}\"")

        if len(scores) > 0:
            sorted_results = sorted(zip(scores, content), key=lambda x: x[0], reverse=True)
            most_relevant_file = sorted_results[0][1]['source']
            top_score = sorted_results[0][0] * 100
            
            st.markdown(f"""
                <div class="result-card result-card-top">
                    <p class="card-header-info" style="color: white; font-size: 1.2rem; border-bottom: none;">🏆 Most Relevant Document Found</p>
                    <p style="font-size: 1.5rem; font-weight: bold; color: white; margin:0;">{most_relevant_file}</p>
                    <p style="color: #E0E0E0; margin-top: 5px;">This file contains the top-scoring sentence with a relevance of {top_score:.2f}%.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Top Relevant Sentences Across All Files:")
            for score, item in sorted_results[:25]:
                relevance_percent = score * 100
                context_before_html = f'<div class="context">...{item["context_before"]}</div>' if item["context_before"] else ""
                main_sentence_html = f'<div class="main-sentence">{item["text"]}</div>'
                context_after_html = f'<div class="context">{item["context_after"]}...</div>' if item["context_after"] else ""

                st.markdown(f"""
                    <div class="result-card">
                        <div class="card-header-info">
                            <span class="relevance-score">Relevance: {relevance_percent:.2f}%</span>
                            <span class="metadata">
                                <span>📄 **{item['source']}**</span> | 
                                <span>📍 {item['location']}</span>
                            </span>
                        </div>
                        {context_before_html if context_before_html else ""}
                        {main_sentence_html}
                        {context_after_html if context_after_html else ""}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No relevant results found for your query.")
    elif not uploaded_files:
        st.info("Upload documents and enter a query in the sidebar to get started.")

if __name__ == "__main__":
    if not all([NLTK_AVAILABLE, SENTENCE_TRANSFORMER_AVAILABLE, PDF_AVAILABLE, DOCX_AVAILABLE]):
        st.error("One or more required libraries are missing. Please check the installation instructions.")
    else:
        main()