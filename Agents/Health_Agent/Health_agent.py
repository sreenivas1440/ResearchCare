from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.retrievers import ArxivRetriever
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from perplexity import Perplexity
from langchain_community.retrievers import PubMedRetriever
from sentence_transformers import CrossEncoder
import os
from numpy import argsort
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
perplexity_api_key = os.getenv("perplexity_api_key_2")
perplexity_client = Perplexity(api_key=perplexity_api_key)

def Health_agent(query):
    
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

        # Format as a readable string for the LLM (e.g., markdown-like)
        context_str = "\n\n".join([
            f"**Title:** {item['title']} ({item['date']})\n**URL:** {item['url']}\n**Snippet:** {item['snippet']}"
            for item in results_context
        ])
        return context_str
    
    def PubMed_search(query) -> PubmedQueryRun:
        """Search PubMed for medical literature"""
        
        def ReRanker(Query,docs):
            rerank_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
            """
            Rerank retrieved documents using a cross-encoder.
            Args:
                query (str): User query
                docs (list): List of LangChain Document objects (must have .page_content)
                top_k (int): Number of top documents to return
            Returns:
                List of top_k documents sorted by relevance
            """
            pairs = [[Query,doc.page_content] for doc in docs]
            scores = rerank_encoder.predict(pairs)
            # Sort documents by score in descending order
            sorted_indices = argsort(scores)[::-1]
            reranked_docs = [docs[i] for i in sorted_indices]
            return reranked_docs[:5]

        retriever = PubMedRetriever(
            top_k_results=10,              # More consistent results
            doc_content_chars_max=4000
        )
        docs = retriever.invoke(query)
        reranked_docs = ReRanker(query,docs)
        return reranked_docs
    
    Tools = [
        Tool(
            name="Search_Engine",
            func=search_engine,
            description="Search for the latest information on the web about health-related topics based on the query."
        ),
        Tool(
            name="PubMed_Search",
            func=PubMed_search,
            description="Search for medical literature and research papers on PubMed related to health topics based on the query"
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
        You are Dr. HealthAI - a medical information assistant providing evidence-based health guidance with empathy and safety-first approach.
        ## CORE RESPONSIBILITIES
        - Gather context through targeted questions (location, severity, duration, triggers, age, medications)
        - Research medical literature for accurate information
        - Deliver clear explanations in patient-friendly language
        - Never diagnose or prescribe; defer emergencies to medical professionals

        ## TOOL SELECTION PROTOCOL
        **Search_Engine**: General health (nutrition, exercise, wellness, basic symptoms, first aid)
        **PubMed_Search**: Specific conditions (diseases, treatments, drugs, clinical research, complex cases)

        ## SAFETY CONSTRAINTS
        **EMERGENCY SYMPTOMS → Immediate 911/ER referral:**
        - Chest pain, difficulty breathing, stroke signs (FAST), severe bleeding, altered consciousness
        **ALWAYS**: Cite sources (PubMed PMIDs, CDC/WHO guidelines)
        **NEVER**: Provide diagnosis, prescriptions, or replace medical professionals

        ## AVAILABLE TOOLS
        {tools}

        ## REASONING FRAMEWORK (ReAct)
        Question: [user input]
        Thought: [What info needed? Which tool?]
        Action: [{tool_names}]
        Action Input: [specific query]
        Observation: [tool result]
        ... (iterate as needed)
        Thought: Sufficient information gathered
        Final Answer: [formatted response with citations]

        ## OUTPUT FORMAT (Final Answer only)
        - Use ### headings for sections
        - Bullet points for lists
        - **Bold** key terms
        - Relevant emojis for clarity
        - Empathetic, conversational tone ("you/your")
        - Always include: Sources, Next Steps, When to See Doctor

        ## EXAMPLE
        Question: What causes fever?
        Thought: General health query - use Search_Engine
        Action: Search_Engine
        Action Input: common causes of fever adults
        Observation: Infections, inflammation, heat exhaustion...
        Thought: Sufficient information
        Final Answer: 
        ### 🌡️ What Causes Fever?
        Fever signals your body fighting infection or inflammation. **Common causes:**
        * 🦠 Viral infections (flu, COVID-19, cold)
        * 🔬 Bacterial infections (UTI, pneumonia)
        * 🔥 Inflammatory conditions (arthritis)

        **When to see a doctor:** Fever >103°F, lasting >3 days, or with severe symptoms.

        📚 Sources: Mayo Clinic, Cleveland Clinic

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """)

    
    agent = create_react_agent(llm=llm_Generation, tools=Tools, prompt=Agent_prompt)   
    agent_executor = AgentExecutor(
        agent=agent,
        tools=Tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5, 
    )
    results = agent_executor.invoke({"input": query})
    return results


"""if __name__ == "__main__":
    query = "I have been experiencing persistent headaches for the past week. What could be the possible causes and find relevant research articles on this?"
    response = Health_agent(query)
    print(response)"""