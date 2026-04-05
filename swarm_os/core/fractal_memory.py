import numpy as np
import json
import os
import base64
import networkx as nx
from typing import Dict, List, Optional, Any, Set
from sentence_transformers import SentenceTransformer
import asyncio
from dataclasses import dataclass, field

class EmbeddingSingleton:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingSingleton, cls).__new__(cls)
            print("Loading sentence-transformers model...")
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text)

@dataclass
class FractalNode:
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    is_compressed: bool = False

class FractalGraphMemory:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, FractalNode] = {}
        self.embedding_model = EmbeddingSingleton()
        self.load()

    def add_node(self, id: str, content: str, metadata: Dict[str, Any] = None, parent_id: Optional[str] = None):
        if metadata is None:
            metadata = {}

        embedding = self.embedding_model.encode(content)
        node = FractalNode(
            id=id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            parent_id=parent_id
        )
        self.nodes[id] = node
        self.graph.add_node(id, embedding=embedding, content=content)

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.add(id)
            self.graph.add_edge(parent_id, id)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[FractalNode]:
        if not self.nodes:
            return []

        query_embedding = self.embedding_model.encode(query)

        similarities = []
        for node_id, node in self.nodes.items():
            sim = self.cosine_similarity(query_embedding, node.embedding)
            similarities.append((sim, node))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in similarities[:top_k]]

    def expand_node(self, node_id: str) -> List[FractalNode]:
        """Expand a compressed parent node to its children."""
        node = self.nodes.get(node_id)
        if not node:
            return []

        children = [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]
        return children

    async def compress_nodes(self, parent_id: str, groq_client, summarizer_prompt: str) -> str:
        """
        Takes children of a parent node, uses LLM to summarize them,
        updates the parent node with the summary, and marks it as compressed.
        """
        parent_node = self.nodes.get(parent_id)
        if not parent_node or not parent_node.children_ids:
            return ""

        children_content = "\n".join([self.nodes[child_id].content for child_id in parent_node.children_ids])

        prompt = f"{summarizer_prompt}\n\nContent to compress:\n{children_content}"

        # We will use groq_client here to summarize
        summary = await groq_client.generate(prompt)

        parent_node.content = summary
        parent_node.embedding = self.embedding_model.encode(summary)
        parent_node.is_compressed = True

        self.graph.nodes[parent_id]['content'] = summary
        self.graph.nodes[parent_id]['embedding'] = parent_node.embedding

        return summary

    def save(self, filepath: str = "memory.json"):
        data = {}
        for node_id, node in self.nodes.items():
            # Convert embedding to base64 string
            emb_bytes = node.embedding.tobytes()
            emb_b64 = base64.b64encode(emb_bytes).decode('utf-8')
            data[node_id] = {
                "id": node.id,
                "content": node.content,
                "embedding_b64": emb_b64,
                "embedding_shape": node.embedding.shape,
                "embedding_dtype": str(node.embedding.dtype),
                "metadata": node.metadata,
                "parent_id": node.parent_id,
                "children_ids": list(node.children_ids),
                "is_compressed": node.is_compressed
            }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Memory saved to {filepath}")

    def load(self, filepath: str = "memory.json"):
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        for node_id, node_data in data.items():
            emb_bytes = base64.b64decode(node_data["embedding_b64"])
            embedding = np.frombuffer(emb_bytes, dtype=np.dtype(node_data["embedding_dtype"])).copy()
            embedding = embedding.reshape(node_data["embedding_shape"])

            node = FractalNode(
                id=node_data["id"],
                content=node_data["content"],
                embedding=embedding,
                metadata=node_data["metadata"],
                parent_id=node_data["parent_id"],
                children_ids=set(node_data["children_ids"]),
                is_compressed=node_data["is_compressed"]
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, embedding=embedding, content=node.content)

        # Reconstruct edges
        for node_id, node in self.nodes.items():
            for child_id in node.children_ids:
                if child_id in self.nodes:
                    self.graph.add_edge(node_id, child_id)
        print(f"Memory loaded from {filepath}")
