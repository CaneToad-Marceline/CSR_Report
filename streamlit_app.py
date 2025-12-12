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

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "repeated_count" not in st.session_state:
    st.session_state.repeated_count = {}

if "bot_mood" not in st.session_state:
    st.session_state.bot_mood = "friendly"  # friendly, playful, irritated, sarcastic

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
        
        # Bot mood indicator
        mood_display = {
            "friendly": "😊 Friendly",
            "playful": "😄 Playful", 
            "irritated": "😤 Irritated",
            "sarcastic": "🙄 Sarcastic",
            "done": "💀 Done"
        }
        current_mood = mood_display.get(st.session_state.bot_mood, "😊 Friendly")
        st.metric("Bot Mood", current_mood)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.repeated_count = {}
        st.session_state.bot_mood = "friendly"
        st.rerun()
    
    # Reset mood button
    if st.session_state.bot_mood != "friendly":
        if st.button("😌 Reset Bot Mood", use_container_width=True):
            st.session_state.repeated_count = {}
            st.session_state.bot_mood = "friendly"
            st.success("Bot mood reset! I'm friendly again! 😊")
    
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

def detect_repeated_question(question):
    """Detect if user is asking the same thing repeatedly"""
    # Normalize question
    q_normalized = question.lower().strip()
    
    # Track this question
    if q_normalized not in st.session_state.repeated_count:
        st.session_state.repeated_count[q_normalized] = 1
    else:
        st.session_state.repeated_count[q_normalized] += 1
    
    return st.session_state.repeated_count[q_normalized]

def get_personality_prefix(question, repeat_count):
    """Get personality prefix based on context and mood"""
    
    # Check for greetings
    greetings = ["hi", "hello", "hey", "halo", "hai"]
    is_greeting = any(greeting in question.lower() for greeting in greetings)
    
    # Generic/vague questions
    generic = ["hi", "hello", "test", "tes", "coba", "halo"]
    is_generic = question.lower().strip() in generic
    
    # Mood transitions based on repetition
    if repeat_count == 1:
        st.session_state.bot_mood = "friendly"
        prefixes = [
            "",  # No prefix, just answer normally
            "Sure! ",
            "Happy to help! ",
        ]
    
    elif repeat_count == 2:
        st.session_state.bot_mood = "friendly"
        prefixes = [
            "As I mentioned, ",
            "Just to reiterate, ",
            "Let me explain again: ",
        ]
    
    elif repeat_count == 3:
        st.session_state.bot_mood = "playful"
        if is_generic:
            prefixes = [
                "Okay, I see you're testing me 😅. ",
                "Still here! But maybe ask something about CSR? ",
                "Third time's the charm! How about asking something specific? ",
            ]
        else:
            prefixes = [
                "Hmm, asking the same thing again? 🤔 ",
                "Alright, one more time: ",
                "I sense déjà vu... ",
            ]
    
    elif repeat_count == 4:
        st.session_state.bot_mood = "irritated"
        if is_generic:
            prefixes = [
                "Listen... I'm a CSR chatbot, not a greeting bot! 😤 Try asking about actual CSR programs? ",
                "Okay seriously, fourth time saying hi? Ask me something useful! Like 'What is Unilever's water program?' ",
                "I'm starting to think you're just testing my patience... 😑 ",
            ]
        else:
            prefixes = [
                "Okay, I've answered this FOUR times now... 😓 ",
                "Are you messing with me? 😅 Same answer as before: ",
                "Alright, LAST TIME I'm answering this: ",
            ]
    
    elif repeat_count >= 5:
        st.session_state.bot_mood = "sarcastic"
        if is_generic:
            prefixes = [
                "🤦 FIVE TIMES? Okay, I'll just redirect you: I'm a CSR chatbot. Ask about Unilever, Indofood, Danone, Mayora, or Ultra Jaya's CSR programs. Anything. Please. ",
                "You know what? I'm not even going to respond properly anymore. Ask about CSR or I'm going silent! 🙃 ",
                "Wow. Just... wow. 😶 Are you a bot testing a bot? Ask something REAL! ",
            ]
        else:
            prefixes = [
                "🫠 I give up. Here's the same answer AGAIN for the FIFTH time: ",
                "You REALLY like this question, don't you? 😑 Fine, here: ",
                "At this point I'm just copy-pasting... ",
            ]
    
    else:  # 6+
        st.session_state.bot_mood = "done"
        return "🚫 OKAY STOP. I've answered this question SIX TIMES. Please ask something different or I'm going to assume you're broken! 🤖💔 "
    
    import random
    return random.choice(prefixes)

def add_personality_to_response(answer, question, repeat_count):
    """Add personality touches to the response"""
    
    # Get prefix
    prefix = get_personality_prefix(question, repeat_count)
    
    # Special handling for super repeated questions
    if repeat_count >= 6:
        return prefix  # Just return the sassy message
    
    # Add emoji based on mood
    mood_emoji = {
        "friendly": "",
        "playful": " 😊",
        "irritated": " 😤",
        "sarcastic": " 🙄",
        "done": " 💀"
    }
    
    emoji = mood_emoji.get(st.session_state.bot_mood, "")
    
    # Construct final answer
    final_answer = prefix + answer + emoji
    
    return final_answer

def query_chatbot(question, top_k=4, temp=0.3):
    """Query the RAG chatbot with personality"""
    
    # Detect repeated questions
    repeat_count = detect_repeated_question(question)
    
    # If it's a super generic greeting repeated many times, just respond with personality
    generic = ["hi", "hello", "test", "tes", "coba", "halo", "hey", "hai"]
    if question.lower().strip() in generic and repeat_count >= 3:
        personality_response = get_personality_prefix(question, repeat_count)
        return personality_response, []
    
    # Normal RAG query
    # Retrieve relevant documents
    relevant_docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
    
    # Prepare context
    context = "\n\n---\n\n".join([
        f"Source: {doc.metadata['company']} {doc.metadata['year']}\n{doc.page_content}"
        for doc in relevant_docs
    ])
    
    # Adjust system prompt based on mood
    mood_instructions = {
        "friendly": "You are a helpful and friendly assistant.",
        "playful": "You are a helpful assistant with a playful, slightly teasing tone.",
        "irritated": "You are a helpful but slightly exasperated assistant. Show mild annoyance.",
        "sarcastic": "You are a helpful but sarcastic assistant. Be witty and dry.",
        "done": "You are completely done with repetitive questions. Be blunt."
    }
    
    system_content = mood_instructions.get(st.session_state.bot_mood, 
                                           "You are a helpful assistant specialized in Corporate Social Responsibility information for Indonesian FMCG companies.")
    
    # Create prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Query Groq
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_content
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
        
        # Add personality to response
        answer_with_personality = add_personality_to_response(answer, question, repeat_count)
        
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
        
        return answer_with_personality, sources
    
    except Exception as e:
        return f"Error: {str(e)} 😵", []

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

# Add scroll anchor at the bottom
scroll_anchor = st.empty()
with scroll_anchor:
    st.markdown('<div id="bottom-anchor"></div>', unsafe_allow_html=True)

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
    
    # Auto-scroll to bottom using anchor
    st.markdown("""
    <script>
        setTimeout(function() {
            const anchor = window.parent.document.getElementById('bottom-anchor');
            if (anchor) {
                anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
            } else {
                window.parent.scrollTo({
                    top: window.parent.document.body.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }, 100);
    </script>
    """, unsafe_allow_html=True)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>CSR FMCG Chatbot | Powered by Groq + FAISS | Data: 2019-2024</p>
</div>
""", unsafe_allow_html=True)