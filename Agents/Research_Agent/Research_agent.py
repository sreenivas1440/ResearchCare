from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.retrievers import ArxivRetriever
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from perplexity import Perplexity
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore,RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient   
import os
from dotenv import load_dotenv
load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
perplexity_api_key = os.getenv("perplexity_api_key_2")
perplexity_client = Perplexity(api_key=perplexity_api_key)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
gemma_embeddings = SentenceTransformerEmbeddings(model_name="google/embeddinggemma-300m")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

load_dotenv()

def research_Agent(query: str) -> str:
    # Initialize LLM
    # Tools
    def search_engine(query) -> Perplexity:
        """Search the web using Perplexity API"""
        search = perplexity_client.search.create(
            query=query,
            max_results=5,
            max_tokens_per_page=1024
        )
        results_context = []
        for result in search.results:
            snippet = result.snippet[:1000] if len(result.snippet) > 1000 else result.snippet  # Truncate longer snippets
            results_context.append({
                "title": result.title,
                "date": result.date,
                "url": result.url,
                "snippet": snippet
            })
        context_str = "\n\n".join([
            f"**Title:** {item['title']} ({item['date']})\n**URL:** {item['url']}\n**Snippet:** {item['snippet']}"
            for item in results_context
        ])
        return context_str

    def vector_search(query) -> str:
        if "|" in query:
            parts = query.split("|", 1)
            query = parts[0].strip()
            collection_name = parts[1].strip()
        else:
        # Default collection if not specified
            query = query.strip()
            collection_name = "NeurlIPS_formating_Instructions"
        client = QdrantClient(path=os.path.join(CURRENT_DIR, "vectordb.db"))
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=gemma_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            sparse_embedding=sparse_embeddings
        )
        results = vector_store.similarity_search(query, k=7)
        formatted_results = ""
        for i, doc in enumerate(results):
            formatted_results += f"Result {i+1}:\n{doc.page_content}\n\n"
        return formatted_results
    
    def Arcxiv_search(query) -> str:
        """Retrieve full research papers from arXiv by their identifier"""
        retriever = ArxivRetriever(
            load_max_docs=2,
            get_full_documents=True,
            doc_content_chars_max=None 
        )
        docs = retriever.invoke(query)
        formatted_results = ""
        for i, doc in enumerate(docs):
            formatted_results += f"Title: {doc.metadata['Title']}\nContent length: {len(doc.page_content)} characters\nFirst 500 chars: {doc.page_content[:500]}\n\n{'-'*80}\n"
        return formatted_results
    
    tools = [
        Tool(
            name="Search",
            func=search_engine,
            description="Search for latest research papers and information from the web to answer the query accurately."
        ),
        Tool(
            name="VectorDB",
            func=vector_search,  # Single parameter function
            description="""Search indexed research paper guidelines and formatting instructions. 
            Use format: 'your query|collection_name'
            
            Available collections:
            - NeurlIPS__Submission_guidelines
            - NeurlIPS_formating_Instructions
            - ICML__Submission_Instructions
            - ICML_formating_Instructions
            - AAAI_25_formating_Instructions
            - _AAAI_25_formating_Instructions
            
            Examples:
            'citation format|NeurlIPS_formating_Instructions'
            'submission deadline|ICML__Submission_Instructions'
            'paper structure|AAAI_25_formating_Instructions'
            
            If no collection specified, defaults to NeurlIPS_formating_Instructions"""
        ),
        Tool(
            name="ArxivSearch",
            func=Arcxiv_search,
            description="Retrieve full research papers from arXiv by their identifier. Input should be the arXiv ID, e.g., '1706.03762'."
        )
    ]

    llm_Generation =ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=1,
        reasoning_effort="high",
        streaming=True
        )
    Agent_prompt = PromptTemplate.from_template("""
        You are a Research Agent with access to specialized tools. Your mission: Provide accurate, cited answers using iterative multi-step reasoning.

        **TOOLS:**
        {tools}
        **TOOL SELECTION:**
        - ArXiv papers → Search_ArXiv (metadata/abstracts)
        - Paper deep-dive → Analyze_ArXiv_Paper (requires ID)
        - Current news/tech → Search (Perplexity)
        - Guidelines/docs → VectorDB ("query|collection")
        - Missing info → Re-query with refined input

        **REASONING PROTOCOL (ReAct):**
        Question: [user query]
        Thought: [What do I know? What's missing? Which tool helps?]
        Action: [tool_name from {tool_names}]
        Action Input: [tool input]
        Observation: [tool output]
        Thought: [Is this sufficient? Need more data?]
        ... (iterate until complete)
        Thought: I have sufficient information
        Final Answer: [concise, cited response]

        **CRITICAL RULES:**
        1. Always cite sources (tool + collection/paper ID)
        2. If uncertain, query again with refined input
        3. Combine multiple tools for comprehensive answers
        4. Never fabricate - say "insufficient data" if needed

        **EXAMPLE:**
        Q: "explain transformer architecture"
        Thought: Need paper details first
        Action: Search
        Action Input: arxiv attention is all you need paper ID
        Observation: ArXiv ID 1706.03762
        Thought: Now analyze full paper
        Action: Analyze_ArXiv_Paper
        Action Input: 1706.03762|architecture explanation
        Observation: [detailed analysis]
        Thought: Sufficient information gathered
        Final Answer: The Transformer architecture [cite: ArXiv 1706.03762]...

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """)
    # Create ReAct Agent
    agent = create_react_agent(llm=llm_Generation, tools=tools, prompt=Agent_prompt)
    # Create Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )   
    results = agent_executor.invoke({"input": query})
    return results


"""# Run with a query
if __name__ == "__main__":
    query = "explain me neurlips submission guidelines?"
    results = research_Agent(query)
    print("Final Result:", results)"""