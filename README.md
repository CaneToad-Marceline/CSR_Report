# 🏭 CSR FMCG Chatbot

A RAG-powered chatbot that answers questions about Corporate Social Responsibility (CSR) programs of Indonesian FMCG companies.

## 📊 Companies Covered
- Danone
- Indofood
- Mayora
- Ultra Jaya
- Unilever

## 📅 Data Coverage
Years: 2019-2024 (25 CSR reports)

## 🚀 Features
- Natural language Q&A about CSR programs
- Bilingual support (English & Indonesian)
- Source citations with company and year
- Claude-style chat interface
- Powered by Groq AI + FAISS vector database

## 💻 Tech Stack
- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Embeddings**: Sentence Transformers (Multilingual)
- **Vector DB**: FAISS
- **Framework**: LangChain

## 🎯 Example Questions
- "What is Unilever's water conservation program in 2023?"
- "Apa program CSR Indofood untuk pendidikan?"
- "Compare energy efficiency initiatives across companies"
- "Which companies have environmental programs?"

## 🔧 Local Setup

1. Clone repository
```bash
git clone <your-repo-url>
cd csr-fmcg-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

4. Run the app
```bash
streamlit run streamlit_app.py
```

## 🌐 Live Demo
[Add your Streamlit Cloud URL here after deployment]

## 📝 License
MIT License

## 👨‍💻 Author
[Your Name]
[Your University/Institution]

---
Built with ❤️ using Streamlit + Groq + FAISS