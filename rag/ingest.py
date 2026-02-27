"""RAG ingestion module for processing and storing documents in ChromaDB."""

import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF for PDF processing
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config


class RAGIngestor:
    """Class for ingesting documents into ChromaDB for RAG operations."""
    
    def __init__(self) -> None:
        """Initialize the RAG ingestor with embeddings and text splitter."""
        try:
            # Initialize embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("Embeddings model loaded successfully")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Vector store path
            self.vectorstore_path = "rag/vectorstore"
            
        except Exception as e:
            print(f"Failed to initialize RAG ingestor: {e}")
            self.embeddings = None
            self.text_splitter = None
            self.vectorstore_path = None
    
    def ingest(self, folder_path: str) -> bool:
        """
        Ingest all supported documents from a folder into ChromaDB.
        
        Args:
            folder_path: Path to folder containing .txt and .pdf files
            
        Returns:
            True if ingestion was successful, False otherwise.
        """
        if not self.embeddings or not self.text_splitter or not self.vectorstore_path:
            print("RAG ingestor not properly initialized")
            return False
        
        try:
            # Validate folder path
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                print(f"Invalid folder path: {folder_path}")
                return False
            
            # Find all supported files
            documents = []
            file_count = 0
            
            # Process .txt files
            for txt_file in folder.glob("*.txt"):
                try:
                    content = self._read_txt_file(txt_file)
                    if content:
                        documents.extend(self._process_document(content, str(txt_file)))
                        file_count += 1
                        print(f"Processed TXT file: {txt_file.name}")
                except Exception as e:
                    print(f"Error processing TXT file {txt_file}: {e}")
            
            # Process .pdf files
            for pdf_file in folder.glob("*.pdf"):
                try:
                    content = self._read_pdf_file(pdf_file)
                    if content:
                        documents.extend(self._process_document(content, str(pdf_file)))
                        file_count += 1
                        print(f"Processed PDF file: {pdf_file.name}")
                except Exception as e:
                    print(f"Error processing PDF file {pdf_file}: {e}")
            
            if not documents:
                print("No documents found or processed")
                return False
            
            # Create vector store directory if it doesn't exist
            os.makedirs(self.vectorstore_path, exist_ok=True)
            
            # Store documents in ChromaDB
            print(f"Storing {len(documents)} document chunks in ChromaDB...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.vectorstore_path
            )
            
            # Persist the vector store
            vectorstore.persist()
            
            print(f"Successfully ingested {file_count} files ({len(documents)} chunks) into ChromaDB")
            return True
            
        except Exception as e:
            print(f"Error during ingestion: {e}")
            return False
    
    def _read_txt_file(self, file_path: Path) -> str:
        """
        Read content from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading {file_path} with alternative encoding: {e}")
                return ""
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def _read_pdf_file(self, file_path: Path) -> str:
        """
        Read content from a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDF content as string.
        """
        try:
            content = []
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    content.append(text)
            
            return "\n".join(content)
            
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def _process_document(self, content: str, source: str) -> List:
        """
        Process document content into chunks with metadata.
        
        Args:
            content: Document content
            source: Source file path
            
        Returns:
            List of document chunks with metadata.
        """
        try:
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
                
                # Create LangChain document-like object
                from langchain.schema import Document
                doc = Document(page_content=chunk, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing document from {source}: {e}")
            return []