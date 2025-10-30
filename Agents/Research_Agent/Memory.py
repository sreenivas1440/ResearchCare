from langchain_docling import DoclingLoader
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore,RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient, models      
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams


def build_vectordb(file_path,collection_name) -> QdrantVectorStore:
    chunker = HybridChunker()
    loader = DoclingLoader(file_path,chunker=chunker)
    chunks = loader.load()
    print("✅ Loaded with Docling")
    # Embeddings
    gemma_embeddings = SentenceTransformerEmbeddings(model_name="google/embeddinggemma-300m")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    #Qdrant VectorDB
    client = QdrantClient(path="./vectordb.db")
    collection_name = collection_name
    client.recreate_collection(
        collection_name=collection_name,
       vectors_config={
            "dense": VectorParams( 
                size=gemma_embeddings.client.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        }
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=gemma_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
        sparse_embedding=sparse_embeddings
    )
    vector_store.add_documents(documents=chunks)
    print("✅ VectorDB built and documents added")
    return vector_store
    

# Build and forget - data is stored in Qdrant
"""build_vectordb("https://neurips.cc/public/guides/PaperChecklist", "NeurlIPS__Submission_guidelines")
build_vectordb("neurips_2024.pdf", "NeurlIPS_formating_Instructions")
print("✅ NeurlIPS guidelines added")


build_vectordb("https://icml.cc/Conferences/2025/AuthorInstructions", "ICML__Submission_Instructions")
build_vectordb("ICML_guidelines.pdf", "ICML_formating_Instructions")
print("✅ ICML guidelines added")

build_vectordb("https://aaai.org/conference/aaai/aaai-25/submission-instructions/", "AAAI-25_Submission_Instructions")
build_vectordb("formatting-instructions-word-2025.pdf", "_AAAI_25_formating_Instructions")
build_vectordb("anonymous-submission-word-2025.pdf", "AAAI-25_Submission_Instructions")"""



    