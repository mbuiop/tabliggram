"""
موتور گراف دانش پیشرفته برای ذخیره و بازیابی اطلاعات
با قابلیت یادگیری از مقالات و اسناد
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import json
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
import sqlite3
from datetime import datetime, timedelta
import spacy
from transformers import AutoTokenizer, AutoModel
import hnswlib
import mmap
import os
from pathlib import Path
import threading
import queue
import heapq
from enum import Enum

class NodeType(Enum):
    """نوع گره‌ها در گراف دانش"""
    DOCUMENT = "document"
    CONCEPT = "concept"
    ENTITY = "entity"
    TOPIC = "topic"
    KEYWORD = "keyword"
    RELATIONSHIP = "relationship"
    QUERY = "query"
    RESPONSE = "response"

class EdgeType(Enum):
    """نوع یال‌ها در گراف دانش"""
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    INSTANCE_OF = "instance_of"

@dataclass
class KnowledgeNode:
    """گره دانش"""
    id: str
    type: NodeType
    content: Any
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0
    vector_id: Optional[int] = None

@dataclass
class KnowledgeEdge:
    """یال دانش"""
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedKnowledgeGraph:
    """گراف دانش پیشرفته با قابلیت جستجوی برداری"""
    
    def __init__(self, dimension: int = 4096):
        self.dimension = dimension
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        
        # ایندکس برداری با FAISS
        self.index = faiss.IndexFlatIP(dimension)
        self.index_to_id: Dict[int, str] = {}
        
        # ایندکس HNSW برای جستجوی سریع‌تر
        self.hnsw_index = hnswlib.Index(space='ip', dim=dimension)
        self.hnsw_index.init_index(max_elements=1000000, ef_construction=200, M=48)
        
        # کش و حافظه
        self.cache = redis.Redis(host='localhost', port=6379, decode_responses=True, db=1)
        self.local_cache = {}
        
        # دیتابیس SQLite
        self.db_path = "knowledge_graph.db"
        self._init_database()
        
        # مدل‌های NLP
        self.nlp = spacy.load("en_core_web_trf")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        
        # صف‌های پردازش
        self.processing_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # آمار و متریک‌ها
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'queries_performed': 0,
            'avg_query_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # قفل‌ها
        self.write_lock = threading.RLock()
        self.index_lock = threading.Lock()
        
        # شروع پردازشگر پس‌زمینه
        self.running = True
        self.background_processor = threading.Thread(target=self._process_queue, daemon=True)
        self.background_processor.start()
    
    def _init_database(self):
        """ایجاد دیتابیس SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول گره‌ها
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT,
                content BLOB,
                embedding BLOB,
                metadata TEXT,
                timestamp TIMESTAMP,
                access_count INTEGER,
                importance REAL,
                vector_id INTEGER
            )
        ''')
        
        # جدول یال‌ها
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                type TEXT,
                weight REAL,
                metadata TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id)
            )
        ''')
        
        # ایندکس‌ها
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_vector ON nodes(vector_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)')
        
        conn.commit()
        conn.close()
    
    async def add_document(self, content: str, metadata: Dict = None) -> str:
        """افزودن سند به گراف دانش"""
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        # ایجاد embedding
        embedding = await self._create_embedding(content)
        
        # ایجاد گره اصلی سند
        doc_node = KnowledgeNode(
            id=doc_id,
            type=NodeType.DOCUMENT,
            content=content[:1000],  # خلاصه
            embedding=embedding,
            metadata=metadata or {},
            importance=1.0
        )
        
        with self.write_lock:
            # افزودن به FAISS
            vector_id = len(self.index_to_id)
            self.index.add(np.array([embedding]))
            self.hnsw_index.add_items(np.array([embedding]), np.array([vector_id]))
            doc_node.vector_id = vector_id
            self.index_to_id[vector_id] = doc_id
            
            # افزودن به گراف
            self.nodes[doc_id] = doc_node
            self.graph.add_node(doc_id, **doc_node.__dict__)
            
            # استخراج مفاهیم و entities
            concepts = await self._extract_concepts(content)
            entities = await self._extract_entities(content)
            
            # افزودن مفاهیم به گراف
            for concept in concepts:
                concept_id = await self._add_concept(concept)
                self.add_edge(doc_id, concept_id, EdgeType.CONTAINS)
            
            # افزودن entities به گراف
            for entity in entities:
                entity_id = await self._add_entity(entity)
                self.add_edge(doc_id, entity_id, EdgeType.CONTAINS)
        
        # ذخیره در دیتابیس
        self._save_to_database(doc_node)
        
        self.stats['total_nodes'] += 1
        return doc_id
    
    async def _create_embedding(self, text: str) -> np.ndarray:
        """ایجاد embedding برای متن"""
        # توکنایز کردن
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # دریافت embedding
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # نرمال‌سازی
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    async def _extract_concepts(self, text: str) -> List[str]:
        """استخراج مفاهیم اصلی از متن"""
        doc = self.nlp(text)
        
        concepts = []
        
        # استخراج noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # محدودیت طول
                concepts.append(chunk.text.lower())
        
        # استخراج named entities
        for ent in doc.ents:
            concepts.append(ent.text.lower())
        
        # حذف تکراری‌ها
        concepts = list(set(concepts))
        
        return concepts[:50]  # محدودیت تعداد
    
    async def _extract_entities(self, text: str) -> List[Dict]:
        """استخراج موجودیت‌های نام‌دار"""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    async def _add_concept(self, concept: str) -> str:
        """افزودن مفهوم به گراف"""
        concept_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()}"
        
        if concept_id not in self.nodes:
            embedding = await self._create_embedding(concept)
            
            concept_node = KnowledgeNode(
                id=concept_id,
                type=NodeType.CONCEPT,
                content=concept,
                embedding=embedding,
                importance=0.8
            )
            
            with self.write_lock:
                vector_id = len(self.index_to_id)
                self.index.add(np.array([embedding]))
                self.hnsw_index.add_items(np.array([embedding]), np.array([vector_id]))
                concept_node.vector_id = vector_id
                self.index_to_id[vector_id] = concept_id
                
                self.nodes[concept_id] = concept_node
                self.graph.add_node(concept_id, **concept_node.__dict__)
            
            self._save_to_database(concept_node)
            self.stats['total_nodes'] += 1
        
        return concept_id
    
    async def _add_entity(self, entity: Dict) -> str:
        """افزودن موجودیت به گراف"""
        entity_text = entity['text']
        entity_id = f"entity_{hashlib.md5(f'{entity_text}_{entity["label"]}'.encode()).hexdigest()}"
        
        if entity_id not in self.nodes:
            embedding = await self._create_embedding(entity_text)
            
            entity_node = KnowledgeNode(
                id=entity_id,
                type=NodeType.ENTITY,
                content=entity,
                embedding=embedding,
                metadata={'label': entity['label']},
                importance=0.9
            )
            
            with self.write_lock:
                vector_id = len(self.index_to_id)
                self.index.add(np.array([embedding]))
                self.hnsw_index.add_items(np.array([embedding]), np.array([vector_id]))
                entity_node.vector_id = vector_id
                self.index_to_id[vector_id] = entity_id
                
                self.nodes[entity_id] = entity_node
                self.graph.add_node(entity_id, **entity_node.__dict__)
            
            self._save_to_database(entity_node)
            self.stats['total_nodes'] += 1
        
        return entity_id
    
    def add_edge(self, source: str, target: str, edge_type: EdgeType, weight: float = 1.0, metadata: Dict = None):
        """افزودن یال بین دو گره"""
        edge = KnowledgeEdge(
            source=source,
            target=target,
            type=edge_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        with self.write_lock:
            self.edges.append(edge)
            self.graph.add_edge(source, target, **edge.__dict__)
            
            # افزایش اهمیت گره‌ها بر اساس تعداد ارتباطات
            if source in self.nodes:
                self.nodes[source].importance += 0.01
            if target in self.nodes:
                self.nodes[target].importance += 0.01
        
        self.stats['total_edges'] += 1
        
        # ذخیره در دیتابیس
        self._save_edge_to_database(edge)
    
    async def search(self, query: str, k: int = 10, node_type: Optional[NodeType] = None) -> List[Dict]:
        """جستجوی پیشرفته در گراف دانش"""
        start_time = datetime.now()
        
        # بررسی کش
        cache_key = f"search:{query}:{k}:{node_type}"
        cached = self.cache.get(cache_key)
        if cached:
            self.stats['cache_hits'] += 1
            return json.loads(cached)
        
        self.stats['cache_misses'] += 1
        
        # ایجاد embedding برای query
        query_embedding = await self._create_embedding(query)
        
        # جستجو در FAISS
        distances, indices = self.index.search(np.array([query_embedding]), k * 2)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.index_to_id:
                node_id = self.index_to_id[idx]
                node = self.nodes.get(node_id)
                
                if node and (node_type is None or node.type == node_type):
                    # دریافت context از گراف
                    context = self._get_node_context(node_id, depth=2)
                    
                    results.append({
                        'node': {
                            'id': node.id,
                            'type': node.type.value,
                            'content': node.content,
                            'importance': node.importance,
                            'metadata': node.metadata
                        },
                        'similarity': float(dist),
                        'context': context,
                        'connections': len(list(self.graph.neighbors(node_id)))
                    })
        
        # مرتب‌سازی بر اساس ترکیبی از شباهت و اهمیت
        results.sort(
            key=lambda x: x['similarity'] * 0.7 + x['node']['importance'] * 0.3,
            reverse=True
        )
        
        results = results[:k]
        
        # ذخیره در کش
        self.cache.setex(cache_key, timedelta(hours=1), json.dumps(results))
        
        # به‌روزرسانی آمار
        self.stats['queries_performed'] += 1
        query_time = (datetime.now() - start_time).total_seconds()
        self.stats['avg_query_time'] = 0.9 * self.stats['avg_query_time'] + 0.1 * query_time
        
        return results
    
    def _get_node_context(self, node_id: str, depth: int = 2) -> Dict:
        """دریافت context یک گره از گراف"""
        if node_id not in self.graph:
            return {}
        
        context = {
            'neighbors': [],
            'paths': [],
            'subgraph': {}
        }
        
        # همسایه‌های مستقیم
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            if edge_data:
                context['neighbors'].append({
                    'id': neighbor,
                    'type': self.nodes[neighbor].type.value if neighbor in self.nodes else 'unknown',
                    'edge_type': list(edge_data.values())[0].get('type', 'unknown') if edge_data else 'unknown'
                })
        
        # مسیرهای کوتاه
        if depth > 1:
            for other_node in list(self.graph.nodes())[:10]:  # محدودیت
                if other_node != node_id:
                    try:
                        path = nx.shortest_path(self.graph, node_id, other_node)
                        if len(path) <= depth + 1:
                            context['paths'].append({
                                'target': other_node,
                                'path': path,
                                'length': len(path) - 1
                            })
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
        
        return context
    
    async def find_related_concepts(self, concept: str, k: int = 5) -> List[Dict]:
        """یافتن مفاهیم مرتبط"""
        # جستجوی مفهوم
        concept_results = await self.search(concept, k=1, node_type=NodeType.CONCEPT)
        
        if not concept_results:
            return []
        
        concept_id = concept_results[0]['node']['id']
        
        # یافتن گره‌های مرتبط در گراف
        related = []
        
        if concept_id in self.graph:
            for neighbor in self.graph.neighbors(concept_id):
                if neighbor in self.nodes:
                    node = self.nodes[neighbor]
                    edge_data = self.graph.get_edge_data(concept_id, neighbor)
                    
                    related.append({
                        'id': neighbor,
                        'content': node.content,
                        'type': node.type.value,
                        'relationship': list(edge_data.values())[0].get('type', 'unknown') if edge_data else 'unknown',
                        'strength': node.importance
                    })
        
        # مرتب‌سازی بر اساس اهمیت
        related.sort(key=lambda x: x['strength'], reverse=True)
        
        return related[:k]
    
    async def get_knowledge_summary(self, topic: str) -> Dict:
        """دریافت خلاصه دانش در مورد یک موضوع"""
        # جستجوی موضوع
        results = await self.search(topic, k=20)
        
        summary = {
            'topic': topic,
            'main_concepts': [],
            'key_entities': [],
            'relationships': [],
            'documents': [],
            'confidence': 0.0
        }
        
        concept_count = 0
        entity_count = 0
        doc_count = 0
        
        for result in results:
            node = result['node']
            
            if node['type'] == NodeType.CONCEPT.value and concept_count < 5:
                summary['main_concepts'].append({
                    'concept': node['content'],
                    'relevance': result['similarity']
                })
                concept_count += 1
            
            elif node['type'] == NodeType.ENTITY.value and entity_count < 10:
                summary['key_entities'].append({
                    'entity': node['content']['text'] if isinstance(node['content'], dict) else node['content'],
                    'label': node['metadata'].get('label', 'unknown'),
                    'relevance': result['similarity']
                })
                entity_count += 1
            
            elif node['type'] == NodeType.DOCUMENT.value and doc_count < 5:
                summary['documents'].append({
                    'id': node['id'],
                    'summary': node['content'],
                    'relevance': result['similarity']
                })
                doc_count += 1
            
            # استخراج روابط
            if 'context' in result and result['context'].get('neighbors'):
                for neighbor in result['context']['neighbors']:
                    summary['relationships'].append({
                        'source': node['id'],
                        'target': neighbor['id'],
                        'type': neighbor['edge_type']
                    })
        
        # محاسبه اطمینان
        if results:
            summary['confidence'] = sum(r['similarity'] for r in results) / len(results)
        
        return summary
    
    def _save_to_database(self, node: KnowledgeNode):
        """ذخیره گره در دیتابیس"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO nodes 
            (id, type, content, embedding, metadata, timestamp, access_count, importance, vector_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.id,
            node.type.value,
            pickle.dumps(node.content),
            node.embedding.tobytes() if node.embedding is not None else None,
            json.dumps(node.metadata),
            node.timestamp,
            node.access_count,
            node.importance,
            node.vector_id
        ))
        
        conn.commit()
        conn.close()
    
    def _save_edge_to_database(self, edge: KnowledgeEdge):
        """ذخیره یال در دیتابیس"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO edges (source, target, type, weight, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            edge.source,
            edge.target,
            edge.type.value,
            edge.weight,
            json.dumps(edge.metadata),
            edge.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def _process_queue(self):
        """پردازشگر پس‌زمینه برای عملیات سنگین"""
        while self.running:
            try:
                item = self.processing_queue.get(timeout=1)
                if item['type'] == 'update_importance':
                    self._update_node_importance(item['node_id'])
                elif item['type'] == 'prune_graph':
                    self._prune_graph()
                elif item['type'] == 'reindex':
                    self._reindex()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"خطا در پردازشگر پس‌زمینه: {e}")
    
    def _update_node_importance(self, node_id: str):
        """به‌روزرسانی اهمیت گره بر اساس استفاده"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # افزایش اهمیت بر اساس تعداد دسترسی
            node.access_count += 1
            node.importance = min(1.0, node.importance + 0.01)
            
            # ذخیره در دیتابیس
            self._save_to_database(node)
    
    def _prune_graph(self, threshold: float = 0.1):
        """هرس گراف برای حذف گره‌های کم‌اهمیت"""
        with self.write_lock:
            nodes_to_remove = []
            
            for node_id, node in self.nodes.items():
                if node.importance < threshold and node.access_count < 5:
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                if node_id in self.graph:
                    self.graph.remove_node(node_id)
                if node_id in self.nodes:
                    del self.nodes[node_id]
                
                # حذف از ایندکس
                if node_id in self.index_to_id.values():
                    # TODO: پیاده‌سازی حذف از FAISS
                    pass
            
            self.stats['total_nodes'] -= len(nodes_to_remove)
            logger.info(f"🧹 {len(nodes_to_remove)} گره کم‌اهمیت هرس شدند")
    
    def _reindex(self):
        """بازایندکس کردن تمام گره‌ها"""
        with self.index_lock:
            # بازنشانی ایندکس‌ها
            self.index = faiss.IndexFlatIP(self.dimension)
            self.hnsw_index = hnswlib.Index(space='ip', dim=self.dimension)
            self.hnsw_index.init_index(max_elements=len(self.nodes) + 1000, ef_construction=200, M=48)
            
            self.index_to_id = {}
            
            # افزودن مجدد همه گره‌ها
            for i, (node_id, node) in enumerate(self.nodes.items()):
                if node.embedding is not None:
                    self.index.add(np.array([node.embedding]))
                    self.hnsw_index.add_items(np.array([node.embedding]), np.array([i]))
                    node.vector_id = i
                    self.index_to_id[i] = node_id
            
            logger.info(f"🔄 بازایندکس کامل شد: {len(self.nodes)} گره")
    
    def get_statistics(self) -> Dict:
        """دریافت آمار کامل"""
        return {
            'total_nodes': self.stats['total_nodes'],
            'total_edges': self.stats['total_edges'],
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph)
            },
            'index_stats': {
                'faiss_size': self.index.ntotal,
                'hnsw_size': self.hnsw_index.get_current_count()
            },
            'performance': {
                'queries': self.stats['queries_performed'],
                'avg_query_time': self.stats['avg_query_time'],
                'cache_hit_rate': self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            },
            'cache_size': len(self.cache.keys())
        }
    
    def export_graph(self, format: str = 'json') -> str:
        """خروجی گرفتن از گراف"""
        if format == 'json':
            data = {
                'nodes': [
                    {
                        'id': n.id,
                        'type': n.type.value,
                        'content': str(n.content)[:100],
                        'importance': n.importance
                    }
                    for n in self.nodes.values()
                ],
                'edges': [
                    {
                        'source': e.source,
                        'target': e.target,
                        'type': e.type.value,
                        'weight': e.weight
                    }
                    for e in self.edges
                ]
            }
            return json.dumps(data, indent=2)
        
        elif format == 'graphml':
            return ''.join(nx.generate_graphml(self.graph))
        
        return ""

class DocumentProcessor:
    """پردازشگر اسناد برای استخراج دانش"""
    
    def __init__(self, knowledge_graph: AdvancedKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.supported_formats = ['txt', 'pdf', 'docx', 'md', 'csv', 'json']
        self.processing_queue = asyncio.Queue()
        
    async def process_document_batch(self, file_paths: List[str]) -> List[str]:
        """پردازش دسته‌ای اسناد"""
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self.process_single_document(file_path))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def process_single_document(self, file_path: str) -> str:
        """پردازش یک سند"""
        logger.info(f"📄 در حال پردازش: {file_path}")
        
        try:
            # خواندن فایل
            content = await self._read_file(file_path)
            
            # پیش‌پردازش
            cleaned_content = self._preprocess_text(content)
            
            # افزودن به گراف دانش
            doc_id = await self.knowledge_graph.add_document(
                cleaned_content,
                metadata={'source': file_path, 'type': Path(file_path).suffix}
            )
            
            logger.info(f"✅ پردازش شد: {file_path} -> {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ خطا در پردازش {file_path}: {e}")
            return ""
    
    async def _read_file(self, file_path: str) -> str:
        """خواندن فایل با فرمت‌های مختلف"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt' or ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == '.pdf':
            # پردازش PDF
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        
        elif ext == '.docx':
            # پردازش DOCX
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        elif ext == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        
        return ""
    
    def _preprocess_text(self, text: str) -> str:
        """پیش‌پردازش متن"""
        # حذف کاراکترهای اضافی
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
        
        # محدودیت طول
        if len(text) > 10000:
            text = text[:10000]
        
        return text.strip()

# نمونه‌سازی و تست
if __name__ == "__main__":
    import asyncio
    
    async def test():
        kg = AdvancedKnowledgeGraph()
        processor = DocumentProcessor(kg)
        
        # افزودن یک سند نمونه
        doc_id = await kg.add_document(
            "هوش مصنوعی شاخه‌ای از علوم کامپیوتر است که به ساخت ماشین‌های هوشمند می‌پردازد.",
            metadata={'source': 'test', 'type': 'txt'}
        )
        
        print(f"سند افزوده شد: {doc_id}")
        
        # جستجو
        results = await kg.search("هوش مصنوعی", k=5)
        print(f"نتایج جستجو: {len(results)}")
        
        # آمار
        stats = kg.get_statistics()
        print(f"آمار: {stats}")
    
    asyncio.run(test())
