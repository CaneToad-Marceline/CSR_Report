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
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        background-color: #2d2d2d;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252525;
    }
    
    /* Input box */
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
    }
    
    .stButton button:hover {
        background-color: #45a049;
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
</style>
""", unsafe_allow_html=True)

# Configuration
VECTOR_DB_PATH = "faiss_index"
# Use smaller, faster model for deployment
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>CSR FMCG Chatbot | Powered by Groq + FAISS | Data: 2019-2024</p>
</div>
""", unsafe_allow_html=True)