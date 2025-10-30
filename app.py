import streamlit as st
import tempfile
import os
from datetime import datetime
from Agents.Main_Agent import ask as ask_agent

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Health & Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ADAPTED RAG CSS STYLES ====================

st.markdown("""
<style>
    /* ========== GREYISH BACKGROUND ========== */
    .stApp {
        background: linear-gradient(135deg, #e8eaf0 0%, #d5d9e3 100%) !important;
    }

    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #c5cae9 0%, #9fa8da 100%) !important;
    }

    [data-testid="stBottom"],
    [data-testid="stChatInputContainer"],
    footer {
        background: linear-gradient(135deg, #e8eaf0 0%, #d5d9e3 100%) !important;
    }
    
    .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }

    /* ========== USER BUBBLE - BLUE ========== */
    .user-bubble {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #000000 !important;
        padding: 16px 20px !important;
        border-radius: 20px 20px 4px 20px !important;
        margin: 12px 0 12px auto !important;
        max-width: 70% !important;
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4) !important;
        animation: slideInRight 0.4s ease !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        font-weight: 500 !important;
    }
    
    .user-bubble * {
        color: #000000 !important;
    }
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="color: #0d47a1; 
                font-size: 48px; 
                font-weight: 800; 
                margin-bottom: 10px; 
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                background: linear-gradient(135deg, #1e40af 0%, #1976d2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;">
            🤖 AI Health & Research Assistant
        </h1>
        <p style="color: #1565c0; 
                font-size: 18px; 
                font-weight: 500;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
            Welcome! Ask me anything about health or research. You can also upload medical documents for analysis.
        </p>
    </div>
    
    /* ========== ASSISTANT BUBBLE - WHITE ========== */
    .assistant-bubble {
        background: #ffffff !important;
        color: #000000 !important;
        padding: 18px 22px !important;
        border-radius: 20px 20px 20px 4px !important;
        margin: 12px auto 12px 0 !important;
        max-width: 75% !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
        animation: slideInLeft 0.4s ease !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }
    
    .assistant-bubble * {
        color: #000000 !important;
    }
    
    .msg-meta {
        font-size: 11px !important;
        opacity: 0.7 !important;
        margin-bottom: 6px !important;
        font-weight: 600 !important;
        color: #000000 !important;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* ========== CHAT INPUT ========== */
    .stChatInput {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        border-radius: 25px !important;
        border: 2px solid #64b5f6 !important;
        padding: 8px !important;
    }

    .stChatInput input {
        background: white !important;
        color: #1565c0 !important;
        border-radius: 20px !important;
        padding: 12px 20px !important;
        font-size: 15px !important;
    }

    .stChatInput input::placeholder {
        color: #64b5f6 !important;
    }

    .stChatInput button {
        background: linear-gradient(135deg, #42a5f5 0%, #2196f3 100%) !important;
        color: white !important;
        border-radius: 50% !important;
    }

    /* ========== BADGES ========== */
    .agent-badge {
        display: inline-block !important;
        padding: 0.4rem 0.8rem !important;
        border-radius: 15px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase !important;
    }
    
    .file-badge {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        padding: 0.4rem 0.8rem !important;
        border-radius: 15px !important;
        font-size: 0.8rem !important;
        display: inline-block !important;
        margin-top: 0.5rem !important;
        font-weight: 600 !important;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f5f5 0%, #e0e0e0 100%) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1976d2 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {
        color: #424242 !important;
    }

    .stats-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stats-box p {
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%) !important;
        color: #1976d2 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%) !important;
        color: white !important;
    }

    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #1f77b4 !important;
        background-color: #f0f8ff !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "processing" not in st.session_state:
    st.session_state.processing = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🤖 AI Assistant Control")
    
    # Statistics box
    st.markdown(f"""
    <div class="stats-box">
        <p>💬 Total Messages: {len(st.session_state.messages)}</p>
        <p>📎 Files Uploaded: {len(st.session_state.uploaded_files)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_files = {}
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_files = {}
            st.rerun()
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This Assistant", expanded=True):
        st.markdown("""
        **🏥 Health Agent**
        - Medical questions & symptoms
        - Prescription analysis
        - Lab report interpretation
        - Health advice & wellness
        
        **🔬 Research Agent**
        - Academic papers & research
        - Technology topics
        - General knowledge queries
        
        **📎 Supported Files**
        - PDF documents
        - Images (JPG, PNG)
        - Word documents (DOCX)
        """)
    
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For Medical Documents:**
        - Upload clear, readable images
        - Supported: Prescriptions, lab reports
        - Ask specific questions
        
        **For Research Questions:**
        - Be specific about your topic
        - Ask for summaries or analysis
        - Provide relevant context
        """)
    
    st.markdown("---")
    st.caption("🚀 Powered by LangChain + LangGraph + Gemini")

# ==================== CHAT DISPLAY ====================
st.markdown('<p style="color: #1976d2; font-size: 48px; font-weight: 800; text-align: center; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">🧠ResearchCare: A Multi Agent Health & Academic Assitant</p>', unsafe_allow_html=True)

st.markdown("---")

# Display messages using PURE HTML (no st.chat_message)
for message in st.session_state.messages:
    if message["role"] == "user":
        # User bubble
        file_html = ""
        if "file_info" in message and message["file_info"]:
            fi = message["file_info"]
            file_html = f'<div style="background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white; padding: 6px 12px; border-radius: 15px; font-size: 13px; display: inline-block; margin-top: 8px; font-weight: 600;">📎 {fi["name"]} • {fi["size_kb"]:.1f} KB</div>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: #000000;
                    padding: 16px 20px;
                    border-radius: 20px 20px 4px 20px;
                    margin: 12px 0 12px auto;
                    max-width: 70%;
                    box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
                    font-size: 15px;
                    line-height: 1.6;">
            <div style="font-size: 11px; opacity: 0.7; margin-bottom: 8px; font-weight: 600;">YOU 👤</div>
            <div style="color: #000000;">{message['content']}</div>
            {file_html}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Assistant bubble
        badge_html = ""
        if message.get("agent_name") == "health_agent":
            badge_html = '<div style="background: linear-gradient(135deg, #dbeafe, #bfdbfe); color: #0c4a6e; padding: 6px 12px; border-radius: 15px; font-size: 12px; display: inline-block; margin-bottom: 10px; font-weight: 700; border: 1px solid #7dd3fc; text-transform: uppercase;">🏥 HEALTH AGENT</div>'
        elif message.get("agent_name") == "research_agent":
            badge_html = '<div style="background: linear-gradient(135deg, #e9d5ff, #d8b4fe); color: #581c87; padding: 6px 12px; border-radius: 15px; font-size: 12px; display: inline-block; margin-bottom: 10px; font-weight: 700; border: 1px solid #c084fc; text-transform: uppercase;">🔬 RESEARCH AGENT</div>'
        
        st.markdown(f"""
        <div style="background: #ffffff;
                    color: #000000;
                    padding: 18px 22px;
                    border-radius: 20px 20px 20px 4px;
                    margin: 12px auto 12px 0;
                    max-width: 75%;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
                    border: 2px solid #e0e0e0;
                    font-size: 15px;
                    line-height: 1.7;">
            <div style="font-size: 11px; opacity: 0.7; margin-bottom: 8px; font-weight: 600;">🤖 AI ASSISTANT</div>
            {badge_html}
            <div style="color: #000000; margin-top: 10px;">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)



# ==================== SEPARATOR ====================
st.markdown("---")

# ==================== INPUT SECTION ====================
col1, col2 = st.columns([0.85, 0.15], gap="small")

with col1:
    user_input = st.chat_input("💬 Ask me about health or research...", key="user_input")

with col2:
    uploaded_file = st.file_uploader(
        "📎 Upload",
        type=["pdf", "jpg", "jpeg", "png", "docx"],
        label_visibility="collapsed",
        key="file_upload",
        help="Upload medical document or image"
    )


# ==================== PROCESS USER INPUT ====================
if user_input and not st.session_state.processing:
    st.session_state.processing = True
    
    file_path = None
    file_info = None
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name
        file_info = {"name": uploaded_file.name, "size_kb": uploaded_file.size / 1024}
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "file_info": file_info,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Generate response
    with st.spinner("🔍 Processing..."):
        try:
            response = ask_agent(user_input, file_path=file_path)
            if file_path or "medical" in response.lower() or "health" in response.lower():
                agent_name = "health_agent"
            else:
                agent_name = "research_agent"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "agent_name": agent_name,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
            st.session_state.processing = False
    
    st.rerun()
