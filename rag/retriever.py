"""RAG retriever module for querying documents from ChromaDB."""

import os
from typing import Dict, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config


class RAGRetriever:
    """Class for retrieving relevant documents from ChromaDB using similarity search."""
    
    def __init__(self) -> None:
        """Initialize the RAG retriever with embeddings and vector store."""
        try:
            # Initialize embeddings model (same as ingestor)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Vector store path
            self.vectorstore_path = "rag/vectorstore"
            
            # Initialize vector store
            self.vectorstore = None
            self._load_vectorstore()
            
        except Exception as e:
            print(f"Failed to initialize RAG retriever: {e}")
            self.embeddings = None
            self.vectorstore = None
            self.vectorstore_path = None
    
    def _load_vectorstore(self) -> None:
        """Load the persisted ChromaDB vector store."""
        try:
            if not os.path.exists(self.vectorstore_path):
                print(f"Vector store directory not found: {self.vectorstore_path}")
                return
            
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings
            )
            print("Vector store loaded successfully")
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vectorstore = None
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: Search query string
            top_k: Number of top results to return (default: 5)
            
        Returns:
            List of dictionaries containing content, source, and similarity_score.
        """
        if not self.vectorstore:
            print("Warning: Vector store not available. Returning empty results.")
            return []
        
        if not query or not query.strip():
            print("Warning: Empty query provided. Returning empty results.")
            return []
        
        try:
            # Perform similarity search
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # Convert to desired format
            results = []
            for doc, score in docs_with_scores:
                result = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "similarity_score": float(1 / (1 + score))  # Convert distance to similarity
                }
                results.append(result)
            
            print(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if the vector store is available for querying.
        
        Returns:
            True if vector store is loaded and ready, False otherwise.
        """
        return self.vectorstore is not None
    
    def get_collection_info(self) -> Dict[str, int]:
        """
        Get information about the vector store collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        if not self.vectorstore:
            return {"status": "unavailable", "document_count": 0}
        
        try:
            # Get collection from ChromaDB
            collection = self.vectorstore._collection
            
            if collection:
                count = collection.count()
                return {
                    "status": "available",
                    "document_count": count,
                    "collection_name": collection.name
                }
            else:
                return {"status": "empty", "document_count": 0}
                
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"status": "error", "document_count": 0}
    
    def refresh_vectorstore(self) -> bool:
        """
        Refresh the vector store by reloading from disk.
        
        Returns:
            True if refresh was successful, False otherwise.
        """
        try:
            self._load_vectorstore()
            return self.is_available()
        except Exception as e:
            print(f"Error refreshing vector store: {e}")
            return False