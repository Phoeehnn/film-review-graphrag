import asyncio
import os

import numpy as np

from .graph import GraphStore, filter_where
from .routing import ResolvedFilters
from .settings import Settings

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


class Encoder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.lock = asyncio.Lock()

    async def load(self):
        async with self.lock:
            if self.model is None:
                self.model = await asyncio.to_thread(self._load)

    def _load(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.settings.embedding_model, revision=self.settings.embedding_revision, device='cpu')
        if model.get_embedding_dimension() != 384:
            raise ValueError('查询编码器必须为原来的 384 维 SBERT 模型')
        model.encode('服务启动检查', normalize_embeddings=True, show_progress_bar=False)
        return model

    async def encode(self, text: str) -> list[float]:
        await self.load()
        async with self.lock:
            value = await asyncio.to_thread(self.model.encode, text, normalize_embeddings=True, show_progress_bar=False)
        if value.shape != (384,) or not np.isfinite(value).all():
            raise ValueError('无效查询向量')
        return value.astype(float).tolist()


class Retriever:
    def __init__(self, settings: Settings, graph: GraphStore, encoder: Encoder):
        self.settings, self.graph, self.encoder = settings, graph, encoder
        self.chroma = None

    async def vector(self, question: str, scope: ResolvedFilters, limit=5) -> list[dict]:
        vector = await self.encoder.encode(question)
        if self.settings.vector_backend == 'chroma':
            return await self._chroma_search(vector, scope, limit)
        if scope.has_scope():
            where, params = filter_where(scope)
            # Rank the entire filtered cohort, never ANN top-k followed by filtering.
            return await self.graph.run(f'''MATCH (c:Comment)
                WHERE {where} AND c.embedding IS NOT NULL
                WITH c, vector.similarity.cosine(c.embedding, $vector) AS neo4j_score
                RETURN c.comment_id AS comment_id, (2*neo4j_score-1) AS similarity
                ORDER BY similarity DESC, comment_id LIMIT $limit''', {**params, 'vector': vector, 'limit': limit})
        return await self.graph.run('''CALL db.index.vector.queryNodes($index, $limit, $vector)
            YIELD node, score RETURN node.comment_id AS comment_id, (2*score-1) AS similarity
            ORDER BY similarity DESC, comment_id''', {'index': self.settings.vector_index, 'limit': limit, 'vector': vector})

    async def _chroma_search(self, vector, scope, limit):
        if scope.emotion or scope.emotions or scope.consistency or scope.min_star is not None or scope.max_star is not None:
            # The contingency data package predates these facets. Fail explicitly
            # until it is re-imported and accepted; never drop a filter silently.
            raise ValueError('Chroma备选数据包尚未验证情绪/一致性/星级筛选，请使用已验收的Neo4j后端')
        def search():
            if self.chroma is None:
                import chromadb
                self.chroma = chromadb.PersistentClient(path=str(self.settings.chroma_dir)).get_collection('comments', embedding_function=None)
            conditions = []
            if scope.movies:
                conditions.append({'movie': {'$in': scope.movies}})
            if scope.topics:
                conditions.append({'topic': {'$in': scope.topics}})
            if scope.sentiment:
                conditions.append({'sentiment': scope.sentiment})
            if scope.date_from:
                conditions.append({'date_ordinal': {'$gte': __import__('datetime').date.fromisoformat(scope.date_from).toordinal()}})
            if scope.date_to:
                conditions.append({'date_ordinal': {'$lte': __import__('datetime').date.fromisoformat(scope.date_to).toordinal()}})
            where = {'$and': conditions} if len(conditions) > 1 else conditions[0] if conditions else None
            result = self.chroma.query(query_embeddings=[vector], n_results=limit, where=where, include=['distances'])
            return [{'comment_id': identifier, 'similarity': 1-distance} for identifier, distance in zip(result['ids'][0], result['distances'][0])]
        return await asyncio.to_thread(search)
