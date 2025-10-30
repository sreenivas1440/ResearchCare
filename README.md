# 🧠 ResearchCare: A Multi-Agent Health & Academic Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-blueviolet)](https://github.com/langchain-ai/langgraph)

---

## 🎯 Overview

**ResearchCare** is an intelligent multi-agent platform that combines healthcare intelligence and academic research capabilities. It features a sophisticated orchestrator that routes queries to specialized agents—one for medical information and one for research topics—while maintaining conversational context and document awareness.

Built for healthcare professionals, researchers, and students who need reliable, cited answers backed by document analysis and academic rigor.

---

## ✨ Key Features

- 🤖 **Multi-Agent Supervisor Architecture** — Intelligent routing between Health Agent and Research Agent using LangGraph orchestration
- 📄 **Document Intelligence** — Upload and analyze:
  - PDFs (medical documents, research papers)
  - Medical Images (prescriptions, lab reports, scans)
  - Word Documents (DOCX)
  - PowerPoint presentations (PPTX)
- 🏥 **Health Agent Capabilities**:
  - 💊 Medication and drug information
  - 🩺 Symptom analysis with context questions
  - 📋 Prescription and lab report interpretation
  - 🚨 Emergency symptom detection & escalation
  - ✓ Source-cited health information
- 🔬 **Research Agent Capabilities**:
  - 📚 Academic paper search and analysis
  - 🧪 Technology and science topics
  - 📖 Paper summarization and Q&A
  - 🔍 Citation tracking
  - 📊 Academic guidelines
- 🗂️ **Semantic Search** — Qdrant vector database for intelligent document retrieval
- 💬 **Production UI** — Beautiful Streamlit interface with animated chat, file uploads, and real-time status
- 🛡️ **Safety-First** — Emergency detection, no diagnosis, fact-verified citations
- ⚡ **Fast & Scalable** — Optimized for production deployment

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.28+ | Interactive UI with chat & file upload |
| **Agent Framework** | LangChain + LangGraph | Agent orchestration & state management |
| **LLM** | Gemini API / Perplexity API | Intelligence & reasoning |
| **Vector DB** | Qdrant | Semantic search & document retrieval |
| **Document Processing** | PyPDF2, Pillow, Docling | Extract text from PDFs & images |
| **APIs** | Perplexity Search, PubMed | Research & medical knowledge |
| **Embeddings** | OpenAI / Google Embeddings | Vector representations |

---

## 🚀 4. Quick Start

### 🛠️ Prerequisites
- 🐍 **Python 3.10+**
- 🔑 **API Keys** (pick one):
  - [Groq](https://console.groq.com/keys) for Llama 3.1 (fastest!).
  - [OpenAI](https://platform.openai.com/api-keys) for GPT series.
  - [Google AI Studio](https://aistudio.google.com/app/apikey) for Gemma/Gemini.

### ⚙️ System Requirements
- 💻 **Recommended RAM:** 8GB+  
- 🧠 **GPU (Optional but Recommended):**  
  - A **CUDA-enabled GPU** is strongly advised for **loading complex or multi-format documents** via **Docling**.  
  - On **CPU-only systems**, large or image-heavy PDFs may fail to load and raise a `pin_memory` error.  
  - This limitation can be partially mitigated using **Unstructured** document loading, though it performs **less accurately than Docling** on structured research papers.  

### 📦 Setup & Run
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/sreenivas1440/ResearchCare
   cd ResearchCare
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets** (create `.env`):
   ```env
   perplexity_api_key=your_perplexity_key_here
   GOOGLE_API_KEY=your_gemini_key_here
   ```

4. **Launch the App**:
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) – query away! 🎉



## 📁 Project Structure

```
ResearchCare/
├── .vscode/  
├── Agents/
│ ├── Main_Agent.py             # Supervisor/Orchestrator
│ ├── Health_Agent              # Health specialist agent
        ├── Health_agent.py
│       └── pubmed.py
│ └── Research_Agent            # Research specialist agent
        ├── Research_agent.py
        └── Memory.py
        └── vectordb.db/collection
├──app.py                       # Streamlit app
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── README.md                   # Documentation
└── LICENSE                     # LicenseMIT License
```


---

## 🙏 Acknowledgments

- **Parsing**: [Docling](https://github.com/DS4SD/docling)
- **Vector DB**: [Qdrant](https://qdrant.tech/)
- **Framework**: [LangChain](https://langchain.com/) & [LangGraph](https://github.com/langchain-ai/langgraph)
- **APIs**: [Google Gemini](https://ai.google.dev/) & [Perplexity](https://www.perplexity.ai/)
- **UI**: [Streamlit](https://streamlit.io/)

---

## 📄 License

MIT License - See LICENSE file

---

## 👨‍💻 Author

**Sreenivas**
- GitHub: [@sreenivas1440](https://github.com/sreenivas1440)

---

Built with ❤️ for AI-powered knowledge discovery. Star ⭐ if you find it useful!



