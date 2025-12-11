"""
CSR RAG Chatbot - Claude-Style Web Interface
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from groq import Groq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CSR FMCG Chatbot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Claude-like appearance
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #1e1e1e;
        overflow-x: hidden;
    }
    
    /* Fix mobile keyboard overlap */
    @media (max-width: 768px) {
        .stChatFloatingInputContainer {
            position: fixed !important;
            bottom: 0 !important;
            background-color: #1e1e1e !important;
            padding-bottom: env(safe-area-inset-bottom) !important;
            z-index: 999 !important;
        }
        
        /* Add padding to bottom of chat to prevent overlap with input */
        .main .block-container {
            padding-bottom: 100px !important;
        }
        
        /* Ensure chat scrolls properly */
        .stChatMessageContainer {
            margin-bottom: 20px !important;
        }
    }
    
    /* Fix desktop scroll bounce/drag issue */
    .main .block-container {
        overflow-y: auto !important;
        overscroll-behavior: contain !important;
        -webkit-overflow-scrolling: touch !important;
    }
    
    /* Prevent rubber band scrolling */
    body {
        overscroll-behavior-y: none !important;
        overflow: hidden !important;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        max-width: 100%;
        word-wrap: break-word;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        background-color: #2d2d2d;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252525;
    }
    
    /* Input box - Fixed positioning */
    .stChatFloatingInputContainer {
        background-color: #1e1e1e !important;
        border-top: 1px solid #404040 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextInput input {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #404040;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }
    
    /* Text */
    p, li, span {
        color: #e0e0e0;
    }
    
    /* Source citations */
    .source-box {
        background-color: #3d3d3d;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #4CAF50 !important;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .stChatMessage {
            padding: 12px;
            font-size: 0.95em;
        }
        
        h1 {
            font-size: 1.5em !important;
        }
        
        /* Prevent text overflow on mobile */
        .main {
            padding: 1rem 0.5rem !important;
        }
        
        /* Better button sizing on mobile */
        .stButton button {
            width: 100% !important;
            margin: 5px 0 !important;
        }
    }
    
    /* Hide Streamlit branding on mobile for cleaner look */
    @media (max-width: 768px) {
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    }
</style>

<script>
// Auto-scroll to bottom on new messages (desktop)
window.addEventListener('load', function() {
    const mainContainer = document.querySelector('.main');
    if (mainContainer) {
        mainContainer.scrollTop = mainContainer.scrollHeight;
    }
});

// Hide keyboard on scroll (mobile)
if (window.innerWidth <= 768) {
    let lastScrollTop = 0;
    window.addEventListener('scroll', function() {
        const st = window.pageYOffset || document.documentElement.scrollTop;
        if (st > lastScrollTop) {
            // Scrolling down - blur input to hide keyboard
            const input = document.querySelector('input[type="text"]');
            if (input) input.blur();
        }
        lastScrollTop = st <= 0 ? 0 : st;
    }, false);
}
</script>
""", unsafe_allow_html=True)

# Configuration
VECTOR_DB_PATH = "faiss_index"
# MUST match the model used to create FAISS index!
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K_RESULTS = 4

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "groq_client" not in st.session_state:
    st.session_state.groq_client = None

if "quick_query" not in st.session_state:
    st.session_state.quick_query = None

# Sidebar
with st.sidebar:
    st.title("🏭 CSR FMCG Chatbot")
    st.markdown("---")
    
    st.subheader("📊 About")
    st.markdown("""
    Chatbot ini dapat menjawab pertanyaan tentang program CSR dari:
    
    **Companies:**
    - 🏢 Danone
    - 🏢 Indofood  
    - 🏢 Mayora
    - 🏢 Ultra Jaya
    - 🏢 Unilever
    
    **Years:** 2019-2024
    """)
    
    st.markdown("---")
    
    st.subheader("💡 Example Questions")
    st.markdown("""
    - "What is Unilever's water conservation program?"
    - "Apa program CSR Indofood untuk pendidikan?"
    - "Compare energy efficiency between companies"
    - "Program lingkungan Danone tahun 2023?"
    """)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of sources", 2, 8, TOP_K_RESULTS)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.3, 0.1)
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.vectorstore:
        st.subheader("📊 Statistics")
        st.metric("Chat Messages", len(st.session_state.messages))
        st.metric("Total Documents", "25 reports")
        st.metric("Years Covered", "2019-2024")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Export chat history
    if len(st.session_state.messages) > 0:
        chat_history = "\n\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages
        ])
        st.download_button(
            label="💾 Download Chat",
            data=chat_history,
            file_name="csr_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    st.caption("Built with Streamlit + Groq + FAISS")

# Initialize components
@st.cache_resource
def load_vectorstore():
    """Load FAISS vector database with HF token"""
    import os
    
    # Get HuggingFace token from secrets or env
    hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
    
    # Set token for sentence-transformers
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu', 'token': hf_token} if hf_token else {'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore

@st.cache_resource
def load_groq_client():
    """Initialize Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found in .env file!")
        st.stop()
    return Groq(api_key=api_key)

# Load resources with loading message
if st.session_state.vectorstore is None:
    with st.spinner("🔄 Loading vector database..."):
        st.session_state.vectorstore = load_vectorstore()

if st.session_state.groq_client is None:
    with st.spinner("🔄 Connecting to AI model..."):
        st.session_state.groq_client = load_groq_client()

# Prompt template
PROMPT_TEMPLATE = """You are a helpful assistant specialized in Corporate Social Responsibility (CSR) information for Indonesian FMCG companies.

Use the following context from CSR reports to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the provided context
- Mention specific company and year when relevant
- Be concise but informative
- You can answer in English or Indonesian
- If you don't have enough information, say so honestly

Answer:"""

def query_chatbot(question, top_k=4, temp=0.3):
    """Query the RAG chatbot"""
    
    # Retrieve relevant documents
    relevant_docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
    
    # Prepare context
    context = "\n\n---\n\n".join([
        f"Source: {doc.metadata['company']} {doc.metadata['year']}\n{doc.page_content}"
        for doc in relevant_docs
    ])
    
    # Create prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Query Groq
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful CSR information assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODEL,
            temperature=temp,
            max_tokens=1000
        )
        
        answer = chat_completion.choices[0].message.content
        
        # Format sources
        sources = []
        seen = set()
        for doc in relevant_docs:
            key = f"{doc.metadata['company']}_{doc.metadata['year']}"
            if key not in seen:
                sources.append({
                    "company": doc.metadata['company'],
                    "year": doc.metadata['year']
                })
                seen.add(key)
        
        return answer, sources
    
    except Exception as e:
        return f"Error: {str(e)}", []

# Main chat interface
st.title("💬 CSR FMCG Assistant")
st.markdown("Ask me anything about Corporate Social Responsibility programs of Indonesian FMCG companies!")

# Welcome message when chat is empty
if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Welcome!** I can help you learn about CSR programs from:
    
    🏢 **Danone** • **Indofood** • **Mayora** • **Ultra Jaya** • **Unilever**
    
    📅 **Data Coverage:** 2019-2024
    
    💡 **Try asking:**
    - "What is Unilever's water conservation program?"
    - "Apa program CSR Indofood untuk pendidikan?"
    - "Compare sustainability initiatives across companies"
    """)
    
    # Quick action buttons
    st.markdown("**🚀 Quick Start:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌊 Water Programs", use_container_width=True):
            st.session_state.quick_query = "What water conservation programs do these companies have?"
    
    with col2:
        if st.button("⚡ Energy Efficiency", use_container_width=True):
            st.session_state.quick_query = "Compare energy efficiency initiatives"
    
    with col3:
        if st.button("🎓 Education CSR", use_container_width=True):
            st.session_state.quick_query = "What education programs are part of CSR?"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📎 View Sources", expanded=False):
                    for i, s in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}.** {s['company']} - {s['year']}")
                        if "raw_docs" in message and i <= len(message.get("raw_docs", [])):
                            with st.container():
                                st.caption("Preview:")
                                preview = message["raw_docs"][i-1].page_content[:200] + "..."
                                st.text(preview)

# Add scroll to bottom button (floats on top)
if len(st.session_state.messages) > 3:
    st.markdown("""
    <style>
        .scroll-button {
            position: fixed;
            bottom: 120px;
            right: 30px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }
        
        .scroll-button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        }
        
        @media (max-width: 768px) {
            .scroll-button {
                bottom: 100px;
                right: 20px;
                width: 45px;
                height: 45px;
                font-size: 20px;
            }
        }
    </style>
    <button class="scroll-button" onclick="window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});">
        ↓
    </button>
    """, unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Ask about CSR programs... (English or Indonesian)")

# Handle quick query buttons
if st.session_state.quick_query and not prompt:
    prompt = st.session_state.quick_query
    st.session_state.quick_query = None

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = query_chatbot(prompt, top_k, temperature)
            st.markdown(answer)
            
            # Display sources
            if sources:
                st.markdown("---")
                st.markdown("**📎 Sources:**")
                sources_text = " • ".join([
                    f"{s['company']} {s['year']}" 
                    for s in sources
                ])
                st.markdown(f'<div class="source-box">{sources_text}</div>', 
                          unsafe_allow_html=True)
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    
    # Auto-scroll to bottom after new message
    st.markdown("""
    <script>
        setTimeout(function() {
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        }, 100);
    </script>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>CSR FMCG Chatbot | By : Marcelino </p>
</div>
""", unsafe_allow_html=True)