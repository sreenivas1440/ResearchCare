# Alternative: PubMedRetriever (more reliable)
from langchain_community.retrievers import PubMedRetriever
from sentence_transformers import CrossEncoder
import os
from numpy import argsort

retriever = PubMedRetriever(
    top_k_results=10,   
    doc_content_chars_max=5000
)

# Get documents directly
docs_1 = retriever.invoke("why do we get headache?")

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


docs_2 = ReRanker("why do we get headache?",docs_1)

print(f"📊 Got {len(docs_2)} documents")

for i, doc in enumerate(docs_2, 1):
    print(f"\n{i}. Title: {doc.metadata.get('Title', 'N/A')}")
    print(f"   PMID: {doc.metadata.get('uid', 'N/A')}")
    print(f"   Content preview: {doc.page_content}...")
