# =============================================================================
# services/chroma_client.py
# AI Developer 2 — used from Day 5 onwards
#
# WHAT THIS FILE DOES:
#   ChromaDB vector database wrapper.
#   Stores text documents as vectors (embeddings).
#   Lets you search by meaning, not just keywords.
# =============================================================================

import logging
import os

logger = logging.getLogger(__name__)


class ChromaClient:
    def __init__(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.client     = chromadb.PersistentClient(path="./chroma_data")
            self.collection = self.client.get_or_create_collection("dpdp_knowledge")
            self.model      = SentenceTransformer("all-MiniLM-L6-v2")
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

    def add_document(self, doc_id: str, text: str):
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
