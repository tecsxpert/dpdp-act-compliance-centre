# =============================================================================
# services/chroma_client.py
# Merged version: Contains AI Dev 2's robust class structure + AI Dev 1's RAG chunking
# =============================================================================

import logging
import os

logger = logging.getLogger(__name__)

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Slices text into chunks of `chunk_size` characters with an `overlap`.
    Required by Day 5 Capstone guide.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move the start pointer forward, minus the overlap
        start += chunk_size - overlap
        
    return chunks

class ChromaClient:
    def __init__(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            # Dev 2's robust initialization
            self.client = chromadb.PersistentClient(path="./chroma_data")
            self.collection = self.client.get_or_create_collection("dpdp_compliance_docs")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._available = True
            logger.info(f"ChromaDB ready — {self.collection.count()} docs loaded")

        except ImportError:
            logger.warning("chromadb / sentence-transformers not installed. ChromaDB disabled.")
            self._available = False
            self.collection = _FakeCollection()

        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}. Using stub.")
            self._available = False
            self.collection = _FakeCollection()

    def ingest_document(self, doc_id: str, text: str, metadata=None):
        """
        AI Dev 1's Chunking Ingestion Pipeline.
        """
        if not self._available:
            return
        if metadata is None:
            metadata = {"source": "manual_ingestion"}

        try:
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            embeddings = self.model.encode(chunks).tolist()
            
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [metadata for _ in range(len(chunks))]
            
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully ingested document '{doc_id}' into {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"ChromaDB ingest error: {e}")

    def add_document(self, doc_id: str, text: str):
        """AI Dev 2's original single-doc add method (kept for compatibility)."""
        if not self._available:
            return
        try:
            embedding = self.model.encode(text).tolist()
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id]
            )
            logger.info(f"ChromaDB: added doc '{doc_id}'")
        except Exception as e:
            logger.error(f"ChromaDB add error: {e}")

    def query(self, question: str, top_k: int = 3) -> list:
        """AI Dev 2's query method for the /categorise endpoint."""
        if not self._available:
            return []
        try:
            embedding = self.model.encode(question).tolist()
            results   = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, max(1, self.collection.count()))
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return []


class _FakeCollection:
    """Stub used when ChromaDB is not installed — prevents import errors."""
    def count(self):
        return 0

    def add(self, **kwargs):
        pass

    def query(self, **kwargs):
        return {"documents": [[]]}

# --- AI Dev 1's Quick Test Block ---
if __name__ == '__main__':
    test_document = (
        "The Digital Personal Data Protection Act (DPDP Act) of India mandates that Data Fiduciaries "
        "must obtain verifiable parental consent before processing any personal data of a child (a person under 18). "
        "Furthermore, fiduciaries are prohibited from undertaking tracking or behavioral monitoring of children "
        "or targeted advertising directed at children. In the event of a personal data breach, the Data Fiduciary "
        "must intimate the Data Protection Board of India and each affected Data Principal. Penalties for non-compliance "
        "can reach up to 250 crore rupees per instance."
    )
    
    print("Testing RAG Ingestion Pipeline...")
    client = ChromaClient()
    client.ingest_document(doc_id="dpdp_rule_001", text=test_document, metadata={"category": "children_and_breaches"})
    
    print(f"Total chunks now stored in ChromaDB: {client.collection.count()}")