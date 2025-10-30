import os
from typing import Literal,Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_docling import DoclingLoader
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain.tools import tool
from .Health_Agent.Health_agent import Health_agent 
from .Research_Agent.Research_agent import research_Agent
from dotenv import load_dotenv
load_dotenv()



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm_Generation =ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)

class AgentState(MessagesState):
    query: Optional[str] = None
    document: Optional[str] = None
    next_agent: Optional[str] = None
    
def health_agent_node(state: AgentState) -> dict:
    query = state["query"]
    document = state.get("document")
    context = f"USER QUERY: {query}\n\nDOCUMENT:\n{document}" if document else query
    result = Health_agent(context)
    
    return {
        "messages":state["messages"]+ [{
            "role": "assistant",
            "content": result["output"],
            "name": "health_agent"
        }]
    }

    
def research_agent_node(state: AgentState) -> dict:
    query = state["query"]
    result = research_Agent(query)
    
    return {
        "messages": state["messages"]+[{
            "role": "assistant",
            "content": result["output"],
            "name": "research_agent"
        }]
    }
    

def process_medical_document(file_path: str) -> dict:
    """Extract structured data from uploaded medical documents (PDF, DOCX, images)"""
    try:
        chunker = HybridChunker()
        loader = DoclingLoader(file_path=file_path,chunker=chunker)
        docs = loader.load()
        
        full_content = "\n\n".join([doc.page_content for doc in docs])
        return full_content
    except Exception as e:
        return None

def Main_Agent(state: AgentState) -> dict: 
    user_query = state["messages"][-1].content
    if "File path:" in user_query:
        query = user_query.split("File path:")[0].strip()
        file_path = user_query.split("File path:")[1].strip()
    else:
        query = user_query
        file_path = None   
    
    document = process_medical_document(file_path) if file_path else None

    prompt = f"""You are the Main Agent for a health and research system.

        USER QUERY: {query}
        
        AVAILABLE SPECIALIST AGENTS:
        1. HEALTH AGENT - Handles:
        - Medical questions (symptoms, diseases, treatments, medications)
        - Health concerns (pain, illness, infections, chronic conditions)
        - Wellness topics (nutrition, exercise, fitness, mental health)
        - Healthcare advice (when to see doctor, emergency situations)
        - Medical research and clinical guidelines

        2. RESEARCH AGENT - Handles:
        - Academic papers and scientific research (non-medical)
        - General information and knowledge questions
        - Technology and computer science topics
        - Historical information
        - Current events and news

        YOUR TASK:
        Analyze the query context (not just keywords) and decide which specialist should handle it.

        ROUTING RULES:
        - Medical/health context → health_agent
        - "Research on headaches" → health_agent (medical research)
        - "Research on AI" → research_agent (non-medical research)
        - Patient describing symptoms → health_agent
        - Academic/general science → research_agent
        - If ambiguous → health_agent (safety first)

        Respond with ONLY ONE WORD: "health_agent" OR "research_agent"
        """
    
    decision = llm_Generation.invoke(prompt).content.strip().lower()
    print(f"Main Agent Decision: {decision}")
    return {
        "query": query,
        "document": document,
        "next_agent":decision
    }
    


def orchestrator():
    graph = StateGraph(AgentState)
    graph.add_node("Main_Agent", Main_Agent)
    graph.add_node("health_agent", health_agent_node)
    graph.add_node("research_agent", research_agent_node)
    
    graph.add_edge(START, "Main_Agent")
    graph.add_conditional_edges(
        "Main_Agent",
        lambda state: state["next_agent"],
        {
            "health_agent": "health_agent",
            "research_agent": "research_agent"
        }
    )
    graph.add_edge("health_agent", END)
    graph.add_edge("research_agent", END)
    
    print(" Orchestrator Ready!\n")
    
    return graph.compile()

main_graph = orchestrator()



def ask(user_input: str, file_path: str = None) -> str:
    if file_path:
        User_query = f"{user_input}\n\nFile path: {file_path}"
    else:
        User_query = user_input
    
    result = main_graph.invoke({
        "messages": [{"role": "user", "content": User_query}]
    })
    
    response = result["messages"][-1].content
    return response





  


