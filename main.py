"""Financial RAG Pipeline - Main Module


* Description:
  Entry point for the RAG system that ties all components together.
  Implements an interactive query loop where users can ask financial
  questions and compare different retrieval and generation approaches.
  Demonstrates four approaches: BM25, Vector retrieval, Hybrid retrieval
  with reranking, and a finetuned model with hybrid retrieval.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Used in text processing and classification of document relevance
  II. Semantics | Probabilistic Models: 
     - Applied in BM25 retrieval which uses probabilistic relevance model
     - Vector embeddings to capture semantic relationships
  III. Language Modeling | Transformers: 
     - Leveraged in both base and finetuned language models for answer generation
  IV. Applications | Custom Statistical or Symbolic: 
     - Application of RAG system to financial domain
     - Integration of multiple approaches into a cohesive QA system

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import torch
from typing import Dict, Any, List
from langchain.schema import BaseRetriever

from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv()

from ingestion import ingest_documents

# Load environment variables
DEVICE = os.getenv('DEVICE', 'cpu')
TOP_K = int(os.getenv('TOP_K', '5'))
FUSION_K = int(os.getenv('FUSION_K', '10')) 
FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', './Finetuned_model')

def run_rag_system():
    """
    Run the complete RAG system with both improvements:
    1. Hybrid retrieval with reranking
    2. Finetuned model with hybrid retrieval
    """
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Using CPU.")
    print(f"Using device: {DEVICE}")

    


if __name__ == "__main__":
    run_rag_system()