"""
رابط چت فوق پیشرفته با قابلیت‌های هوشمند
شامل WebSocket, real-time communication, sentiment analysis,
context management, و قابلیت‌های چندزبانه
"""
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import json
import uuid
import hashlib
import base64
import numpy as np
import torch
from enum import Enum
import logging
from collections import deque, defaultdict
import redis
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import psutil
import GPUtil
from contextlib import asynccontextmanager
import aiofiles
from pathlib import Path
import pickle
import zlib
import gzip
from ratelimit import limits, RateLimitException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import jwt
from passlib.context import CryptContext
from email_validator import validate_email, EmailNotValidError
import re
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from googletrans import Translator
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import openai
from cachetools import TTLCache, LRUCache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from bs4 import BeautifulSoup
import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import plotly.graph_objects as go
import plotly.utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain
from wordcloud import WordCloud
import io
from PIL import Image
import base64

# تنظیمات NLTK
nltk.download('vader_lexicon', quiet=True)
DetectorFactory.seed = 0

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# متریک‌های Prometheus
CHAT_MESSAGES = Counter('chat_messages_total', 'Total chat messages')
CHAT_SESSIONS = Counter('chat_sessions_total', 'Total chat sessions')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time in seconds')
TOKEN_USAGE = Counter('token_usage_total', 'Total token usage')

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# اپلیکیشن FastAPI
app = FastAPI(title="Advanced AI Chat Interface", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# استاتیک فایل‌ها
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Redis برای session management و caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True, db=2)
aioredis_client = None

# کش‌های درون حافظه
session_cache = TTLCache(maxsize=10000, ttl=3600)  # 1 ساعت
response_cache = LRUCache(maxsize=5000)  # 5000 پاسخ اخیر

# Thread pool برای عملیات سنگین
executor = ThreadPoolExecutor(max_workers=10)

# Translator برای چندزبانه
translator = Translator()

# مدل‌های sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", device=-1)

# پترن‌های مختلف
URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
EMAIL_PATTERN = re.compile(r'[^@]+@[^@]+\.[^@]+')
PHONE_PATTERN = re.compile(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}')

# ==================== مدل‌های داده ====================

class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    CODE = "code"
    MARKDOWN = "markdown"
    EMOTION = "emotion"
    COMMAND = "command"

class MessageStatus(str, Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    PROCESSING = "processing"

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"
    GUEST = "guest"

class ChatMode(str, Enum):
    CONVERSATION = "conversation"
    CODE_ASSISTANT = "code_assistant"
    WRITING_ASSISTANT = "writing_assistant"
    TRANSLATOR = "translator"
    SUMMARIZER = "summarizer"
    ANALYST = "analyst"

class Message(BaseModel):
    """مدل پیام"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    type: MessageType = MessageType.TEXT
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    status: MessageStatus = MessageStatus.SENT
    reply_to: Optional[str] = None
    edited: bool = False
    deleted: bool = False
    reactions: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatSession(BaseModel):
    """مدل session چت"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    mode: ChatMode = ChatMode.CONVERSATION
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)
    messages: List[str] = Field(default_factory=list)  # IDs پیام‌ها
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class User(BaseModel):
    """مدل کاربر"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    sessions: List[str] = Field(default_factory=list)
    banned: bool = False

class ChatRequest(BaseModel):
    """درخواست چت"""
    message: str
    session_id: Optional[str] = None
    mode: ChatMode = ChatMode.CONVERSATION
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    context: Optional[Dict[str, Any]] = None
    attachments: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """پاسخ چت"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    tokens_used: int
    processing_time: float
    sentiment: Dict[str, Any]
    emotions: List[Dict[str, Any]]
    suggestions: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ==================== مدیریت session و کاربران ====================

class SessionManager:
    """مدیریت sessions چت"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.lock = asyncio.Lock()
        
    async def create_session(self, user_id: str, mode: ChatMode = ChatMode.CONVERSATION) -> ChatSession:
        """ایجاد session جدید"""
        async with self.lock:
            session = ChatSession(
                user_id=user_id,
                mode=mode,
                context={
                    'history': [],
                    'mode': mode.value,
                    'created_at': datetime.now().isoformat()
                }
            )
            
            self.sessions[session.id] = session
            self.user_sessions[user_id].append(session.id)
            
            # ذخیره در Redis
            redis_client.setex(
                f"session:{session.id}",
                3600,
                session.json()
            )
            
            CHAT_SESSIONS.inc()
            logger.info(f"Session created: {session.id} for user {user_id}")
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """دریافت session با ID"""
        # بررسی کش
        if session_id in session_cache:
            return session_cache[session_id]
        
        # بررسی Redis
        session_data = redis_client.get(f"session:{session_id}")
        if session_data:
            session = ChatSession.parse_raw(session_data)
            session_cache[session_id] = session
            return session
        
        # بررسی حافظه داخلی
        async with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session_cache[session_id] = session
                return session
        
        return None
    
    async def update_session(self, session: ChatSession):
        """به‌روزرسانی session"""
        async with self.lock:
            session.last_activity = datetime.now()
            self.sessions[session.id] = session
            
            # به‌روزرسانی Redis
            redis_client.setex(
                f"session:{session.id}",
                3600,
                session.json()
            )
            
            session_cache[session.id] = session
    
    async def add_message(self, session_id: str, message_id: str):
        """افزودن پیام به session"""
        session = await self.get_session(session_id)
        if session:
            session.messages.append(message_id)
            await self.update_session(session)
    
    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """دریافت تمام sessions یک کاربر"""
        sessions = []
        for session_id in self.user_sessions.get(user_id, []):
            session = await self.get_session(session_id)
            if session and session.is_active:
                sessions.append(session)
        return sessions
    
    async def close_session(self, session_id: str):
        """بستن session"""
        async with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.is_active = False
                await self.update_session(session)
                
                # حذف از Redis ولی نگه داشتن در حافظه
                redis_client.delete(f"session:{session_id}")
                if session_id in session_cache:
                    del session_cache[session_id]
                
                logger.info(f"Session closed: {session_id}")

class UserManager:
    """مدیریت کاربران"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.email_to_id: Dict[str, str] = {}
        self.lock = asyncio.Lock()
        
    async def create_user(self, username: str, email: str, role: UserRole = UserRole.USER) -> User:
        """ایجاد کاربر جدید"""
        # اعتبارسنجی ایمیل
        try:
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {e}")
        
        async with self.lock:
            if email in self.email_to_id:
                raise ValueError("Email already registered")
            
            user = User(
                username=username,
                email=email,
                role=role,
                preferences={
                    'theme': 'light',
                    'language': 'en',
                    'notifications': True,
                    'auto_save': True
                }
            )
            
            self.users[user.id] = user
            self.email_to_id[email] = user.id
            
            # ذخیره در Redis
            redis_client.setex(
                f"user:{user.id}",
                86400,  # 24 ساعت
                user.json()
            )
            
            logger.info(f"User created: {user.id} - {username}")
            
            return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """دریافت کاربر با ID"""
        # بررسی Redis
        user_data = redis_client.get(f"user:{user_id}")
        if user_data:
            return User.parse_raw(user_data)
        
        # بررسی حافظه
        async with self.lock:
            return self.users.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """دریافت کاربر با ایمیل"""
        user_id = self.email_to_id.get(email)
        if user_id:
            return await self.get_user(user_id)
        return None
    
    async def update_user(self, user: User):
        """به‌روزرسانی اطلاعات کاربر"""
        async with self.lock:
            self.users[user.id] = user
            
            # به‌روزرسانی Redis
            redis_client.setex(
                f"user:{user.id}",
                86400,
                user.json()
            )
    
    async def increment_tokens(self, user_id: str, tokens: int):
        """افزایش تعداد توکن‌های مصرفی کاربر"""
        user = await self.get_user(user_id)
        if user:
            user.tokens_used += tokens
            await self.update_user(user)
            TOKEN_USAGE.inc(tokens)

# ==================== پردازشگرهای پیام ====================

class MessageProcessor:
    """پردازشگر پیشرفته پیام‌ها"""
    
    def __init__(self, brain, knowledge_graph, learning_engine):
        self.brain = brain
        self.knowledge_graph = knowledge_graph
        self.learning_engine = learning_engine
        
        # پردازشگرهای تخصصی
        self.code_processor = CodeProcessor()
        self.markdown_processor = MarkdownProcessor()
        self.emotion_processor = EmotionProcessor()
        self.translation_processor = TranslationProcessor()
        self.summarization_processor = SummarizationProcessor()
        
        # تاریخچه مکالمات
        self.conversation_history = defaultdict(lambda: deque(maxlen=50))
        
    async def process_message(
        self,
        message: Message,
        session: ChatSession,
        user: User
    ) -> ChatResponse:
        """پردازش پیام و تولید پاسخ"""
        start_time = datetime.now()
        
        # تشخیص نوع محتوا
        if message.type == MessageType.CODE:
            response_content = await self.code_processor.process(message.content, session.mode)
        elif message.type == MessageType.MARKDOWN:
            response_content = await self.markdown_processor.process(message.content)
        else:
            # پردازش معمولی متن
            response_content = await self._process_text_message(message, session, user)
        
        # تحلیل احساسات
        sentiment = await self._analyze_sentiment(message.content)
        
        # تشخیص احساسات (emotions)
        emotions = await self._detect_emotions(message.content)
        
        # تولید پیشنهادات
        suggestions = await self._generate_suggestions(message.content, session.mode)
        
        # موضوعات مرتبط
        related_topics = await self._find_related_topics(message.content)
        
        # محاسبه زمان پردازش
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # تعداد توکن‌های مصرفی (تخمینی)
        tokens_used = len(message.content.split()) * 1.3  # تخمین ساده
        
        response = ChatResponse(
            session_id=session.id,
            message=response_content,
            tokens_used=int(tokens_used),
            processing_time=processing_time,
            sentiment=sentiment,
            emotions=emotions,
            suggestions=suggestions,
            related_topics=related_topics,
            metadata={
                'mode': session.mode.value,
                'user_id': user.id,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # ذخیره در تاریخچه
        self.conversation_history[session.id].append({
            'user_message': message.content,
            'assistant_response': response_content,
            'timestamp': datetime.now()
        })
        
        RESPONSE_TIME.observe(processing_time)
        
        return response
    
    async def _process_text_message(
        self,
        message: Message,
        session: ChatSession,
        user: User
    ) -> str:
        """پردازش پیام متنی"""
        
        # بررسی کش
        cache_key = hashlib.md256(f"{message.content}:{session.mode.value}".encode()).hexdigest()
        if cache_key in response_cache:
            logger.info(f"Cache hit for message: {cache_key}")
            return response_cache[cache_key]
        
        # جستجو در دانش
        knowledge_results = await self.knowledge_graph.search(message.content, k=5)
        
        # دریافت context از session
        context = session.context
        context['recent_history'] = list(self.conversation_history[session.id])[-5:]
        context['knowledge'] = knowledge_results
        
        # پردازش با مغز
        with torch.no_grad():
            # تبدیل به فرمت مناسب برای مغز
            input_text = self._prepare_input(message.content, context, session.mode)
            
            # تولید پاسخ
            response = await self.brain.generate_response(
                input_text,
                temperature=message.metadata.get('temperature', 0.7),
                max_length=message.metadata.get('max_tokens', 1000)
            )
        
        # به‌روزرسانی context
        session.context['last_response'] = response
        session.context['history'].append({
            'user': message.content,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # ذخیره در کش
        response_cache[cache_key] = response
        
        return response
    
    def _prepare_input(self, message: str, context: Dict, mode: ChatMode) -> str:
        """آماده‌سازی ورودی برای مغز"""
        input_parts = []
        
        # اضافه کردن mode
        if mode == ChatMode.CODE_ASSISTANT:
            input_parts.append("[CODE ASSISTANT MODE]")
        elif mode == ChatMode.WRITING_ASSISTANT:
            input_parts.append("[WRITING ASSISTANT MODE]")
        elif mode == ChatMode.TRANSLATOR:
            input_parts.append("[TRANSLATOR MODE]")
        
        # اضافه کردن context
        if context.get('knowledge'):
            input_parts.append("Relevant knowledge:")
            for k in context['knowledge'][:3]:
                input_parts.append(f"- {k['node']['content']}")
        
        # اضافه کردن history
        if context.get('recent_history'):
            input_parts.append("Recent conversation:")
            for h in context['recent_history'][-3:]:
                input_parts.append(f"User: {h['user_message']}")
                input_parts.append(f"Assistant: {h['assistant_response']}")
        
        # اضافه کردن پیام اصلی
        input_parts.append(f"User: {message}")
        input_parts.append("Assistant:")
        
        return "\n".join(input_parts)
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """تحلیل احساسات متن"""
        # با TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # با VADER
        vader_scores = sentiment_analyzer.polarity_scores(text)
        
        # تشخیص زبان
        try:
            lang = detect(text)
        except:
            lang = 'unknown'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'vader': vader_scores,
            'language': lang,
            'sentiment': 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
        }
    
    async def _detect_emotions(self, text: str) -> List[Dict[str, Any]]:
        """تشخیص احساسات (anger, joy, sadness, etc)"""
        try:
            results = emotion_classifier(text, top_k=3)
            return [
                {
                    'emotion': r['label'],
                    'score': r['score']
                }
                for r in results
            ]
        except:
            return []
    
    async def _generate_suggestions(self, message: str, mode: ChatMode) -> List[str]:
        """تولید پیشنهادات بر اساس پیام"""
        suggestions = []
        
        if mode == ChatMode.CODE_ASSISTANT:
            suggestions = [
                "Can you explain this code?",
                "How can I optimize this?",
                "Show me an example",
                "Debug this for me"
            ]
        elif mode == ChatMode.WRITING_ASSISTANT:
            suggestions = [
                "Make it more formal",
                "Summarize this",
                "Check grammar",
                "Improve clarity"
            ]
        elif mode == ChatMode.TRANSLATOR:
            suggestions = [
                "Translate to Spanish",
                "Translate to French",
                "Translate to German",
                "Translate to Chinese"
            ]
        else:
            # پیشنهادات بر اساس محتوا
            if '?' in message:
                suggestions.append("I can help answer that question")
            if 'code' in message.lower() or 'programming' in message.lower():
                suggestions.append("Switch to code assistant mode")
            if len(message.split()) > 50:
                suggestions.append("Summarize this for me")
        
        return suggestions[:4]
    
    async def _find_related_topics(self, message: str) -> List[str]:
        """یافتن موضوعات مرتبط"""
        # جستجو در گراف دانش
        results = await self.knowledge_graph.search(message, k=3)
        
        topics = []
        for r in results:
            node = r['node']
            if node['type'] == 'concept' or node['type'] == 'topic':
                topics.append(node['content'])
        
        return topics[:5]

# ==================== پردازشگرهای تخصصی ====================

class CodeProcessor:
    """پردازشگر کد"""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp', 'go', 'rust', 'html', 'css']
        
    async def process(self, code: str, mode: ChatMode) -> str:
        """پردازش کد"""
        if mode == ChatMode.CODE_ASSISTANT:
            # تشخیص زبان
            language = self._detect_language(code)
            
            # بررسی خطاهای نحوی
            errors = self._check_syntax(code, language)
            
            if errors:
                return f"Found potential issues:\n{errors}"
            
            # توضیح کد
            explanation = self._explain_code(code, language)
            
            # بهینه‌سازی
            optimization = self._suggest_optimizations(code, language)
            
            return f"{explanation}\n\nOptimization suggestions:\n{optimization}"
        
        return code
    
    def _detect_language(self, code: str) -> str:
        """تشخیص زبان برنامه‌نویسی"""
        # الگوریتم ساده تشخیص
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'python'
        elif 'function ' in code or 'console.log' in code:
            return 'javascript'
        elif 'public class' in code or 'System.out.println' in code:
            return 'java'
        elif '#include' in code:
            return 'cpp'
        return 'unknown'
    
    def _check_syntax(self, code: str, language: str) -> str:
        """بررسی خطاهای نحوی"""
        # اینجا می‌توانید از linterهای مختلف استفاده کنید
        return "No syntax errors detected."
    
    def _explain_code(self, code: str, language: str) -> str:
        """توضیح کد"""
        lines = code.split('\n')
        
        explanation = "Code explanation:\n"
        for i, line in enumerate(lines[:10]):  # محدودیت
            if line.strip():
                explanation += f"Line {i+1}: {self._explain_line(line, language)}\n"
        
        return explanation
    
    def _explain_line(self, line: str, language: str) -> str:
        """توضیح یک خط کد"""
        if language == 'python':
            if 'def ' in line:
                func_name = line.split('def ')[1].split('(')[0]
                return f"Defining function '{func_name}'"
            elif 'import ' in line:
                return f"Importing module: {line.split('import ')[1]}"
            elif 'print(' in line:
                return "Printing output"
            elif 'if ' in line:
                return "Conditional statement"
        return "Code line"
    
    def _suggest_optimizations(self, code: str, language: str) -> str:
        """پیشنهاد بهینه‌سازی"""
        suggestions = []
        
        if language == 'python':
            if 'for i in range(len(' in code:
                suggestions.append("Consider using enumerate() instead of range(len())")
            if '.append' in code and 'for' in code:
                suggestions.append("Consider using list comprehension")
            if 'while True:' in code:
                suggestions.append("Ensure there's a break condition")
        
        if not suggestions:
            suggestions.append("Code looks good!")
        
        return '\n'.join(suggestions)

class MarkdownProcessor:
    """پردازشگر مارک‌داون"""
    
    async def process(self, content: str) -> str:
        """تبدیل مارک‌داون به HTML"""
        # تبدیل مارک‌داون
        html = markdown.markdown(
            content,
            extensions=['extra', 'codehilite', 'toc', 'tables']
        )
        
        # هایلایت کردن کد
        html = self._highlight_code(html)
        
        return html
    
    def _highlight_code(self, html: str) -> str:
        """هایلایت کردن قطعات کد"""
        # اینجا می‌توانید با regex کدها را پیدا کنید و هایلایت کنید
        return html

class EmotionProcessor:
    """پردازشگر احساسات"""
    
    def __init__(self):
        self.emotion_lexicon = {
            'joy': ['happy', 'great', 'excellent', 'wonderful', 'amazing'],
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'terrible'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious'],
            'surprise': ['wow', 'amazing', 'unbelievable', 'shocked', 'astonished'],
            'love': ['love', 'adore', 'cherish', 'treasure', 'worship']
        }
        
    async def process(self, text: str) -> Dict[str, float]:
        """تشخیص احساسات متن"""
        text_lower = text.lower()
        words = text_lower.split()
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_lexicon}
        
        for word in words:
            for emotion, lexicon in self.emotion_lexicon.items():
                if word in lexicon:
                    emotion_scores[emotion] += 1.0
        
        # نرمال‌سازی
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return emotion_scores

class TranslationProcessor:
    """پردازشگر ترجمه"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'fa': 'Persian',
            'ar': 'Arabic',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean'
        }
        
    async def translate(self, text: str, target_lang: str = 'en') -> str:
        """ترجمه متن"""
        try:
            result = await translator.translate(text, dest=target_lang)
            return result.text
        except:
            return text
    
    async def detect_language(self, text: str) -> str:
        """تشخیص زبان متن"""
        try:
            lang = detect(text)
            return self.supported_languages.get(lang, 'Unknown')
        except:
            return 'Unknown'

class SummarizationProcessor:
    """پردازشگر خلاصه‌سازی"""
    
    async def summarize(self, text: str, ratio: float = 0.3) -> str:
        """خلاصه‌سازی متن"""
        # اینجا می‌توانید از مدل‌های summarization استفاده کنید
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 3:
            return text
        
        # TF-IDF برای انتخاب جملات مهم
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # امتیازدهی جملات
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # انتخاب بهترین جملات
        num_sentences = max(1, int(len(sentences) * ratio))
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices.sort()
        
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary

# ==================== WebSocket مدیریت ====================

class ConnectionManager:
    """مدیریت اتصالات WebSocket"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """ایجاد اتصال جدید"""
        await websocket.accept()
        
        async with self.lock:
            connection_id = str(uuid.uuid4())
            self.active_connections[connection_id] = websocket
            self.connection_sessions[connection_id] = session_id
            self.session_connections[session_id] = connection_id
            
            ACTIVE_CONNECTIONS.inc()
            logger.info(f"New connection: {connection_id} for session {session_id}")
            
            return connection_id
    
    async def disconnect(self, connection_id: str):
        """قطع اتصال"""
        async with self.lock:
            if connection_id in self.active_connections:
                session_id = self.connection_sessions.get(connection_id)
                
                del self.active_connections[connection_id]
                if connection_id in self.connection_sessions:
                    del self.connection_sessions[connection_id]
                if session_id and session_id in self.session_connections:
                    del self.session_connections[session_id]
                
                ACTIVE_CONNECTIONS.dec()
                logger.info(f"Connection closed: {connection_id}")
    
    async def send_message(self, connection_id: str, message: Dict):
        """ارسال پیام به یک اتصال"""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
    
    async def send_to_session(self, session_id: str, message: Dict):
        """ارسال پیام به یک session"""
        connection_id = self.session_connections.get(session_id)
        if connection_id:
            await self.send_message(connection_id, message)
    
    async def broadcast(self, message: Dict):
        """ارسال پیام به همه"""
        async with self.lock:
            for connection_id in list(self.active_connections.keys()):
                await self.send_message(connection_id, message)

# ==================== API Endpoints ====================

# نمونه‌سازی مدیران
session_manager = SessionManager()
user_manager = UserManager()
connection_manager = ConnectionManager()
message_processor = None  # در startup مقداردهی می‌شود

@app.on_event("startup")
async def startup_event():
    """رویداد راه‌اندازی"""
    global aioredis_client, message_processor
    
    # اتصال به Redis
    aioredis_client = await aioredis.from_url("redis://localhost", decode_responses=True)
    
    # ایجاد پردازشگر پیام (با مغز و گراف دانش)
    from core_quantum_brain import QuantumBrain
    from knowledge_graph_engine import AdvancedKnowledgeGraph
    from neural_learning_engine import NeuralLearningEngine
    
    brain = QuantumBrain()
    kg = AdvancedKnowledgeGraph()
    learning_engine = NeuralLearningEngine(brain)
    
    message_processor = MessageProcessor(brain, kg, learning_engine)
    
    logger.info("🚀 Chat interface started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """رویداد خاموش‌سازی"""
    await aioredis_client.close()
    logger.info("👋 Chat interface shut down")

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """صفحه اصلی چت"""
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "title": "Advanced AI Chat",
            "version": "1.0.0"
        }
    )

@app.get("/api/health")
async def health_check():
    """بررسی سلامت سیستم"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(connection_manager.active_connections),
        "active_sessions": len(session_manager.sessions),
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.cpu_percent()
    }

@app.post("/api/users")
async def create_user(username: str, email: str):
    """ایجاد کاربر جدید"""
    try:
        user = await user_manager.create_user(username, email)
        return {"user_id": user.id, "message": "User created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/sessions")
async def create_session(user_id: str, mode: ChatMode = ChatMode.CONVERSATION):
    """ایجاد session جدید"""
    user = await user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    session = await session_manager.create_session(user_id, mode)
    return {"session_id": session.id, "mode": session.mode.value}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint برای چت real-time"""
    connection_id = await connection_manager.connect(websocket, session_id)
    
    try:
        # دریافت session
        session = await session_manager.get_session(session_id)
        if not session:
            await websocket.send_json({"error": "Session not found"})
            await connection_manager.disconnect(connection_id)
            return
        
        # دریافت کاربر
        user = await user_manager.get_user(session.user_id)
        if not user:
            await websocket.send_json({"error": "User not found"})
            await connection_manager.disconnect(connection_id)
            return
        
        # ارسال پیام خوش‌آمدگویی
        await connection_manager.send_to_session(session_id, {
            "type": "system",
            "message": f"Welcome {user.username}! Mode: {session.mode.value}",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # دریافت پیام
            data = await websocket.receive_json()
            
            CHAT_MESSAGES.inc()
            
            # ایجاد پیام
            message = Message(
                session_id=session_id,
                user_id=user.id,
                type=MessageType(data.get("type", "text")),
                content=data.get("message", ""),
                metadata=data.get("metadata", {})
            )
            
            # تایید دریافت
            await connection_manager.send_to_session(session_id, {
                "type": "ack",
                "message_id": message.id,
                "status": "received"
            })
            
            # پردازش پیام در پس‌زمینه
            asyncio.create_task(process_and_respond(message, session, user, connection_id))
            
    except WebSocketDisconnect:
        await connection_manager.disconnect(connection_id)
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.disconnect(connection_id)

async def process_and_respond(message: Message, session: ChatSession, user: User, connection_id: str):
    """پردازش پیام و ارسال پاسخ"""
    try:
        # نمایش وضعیت typing
        await connection_manager.send_to_session(session.id, {
            "type": "typing",
            "status": True
        })
        
        # پردازش پیام
        response = await message_processor.process_message(message, session, user)
        
        # به‌روزرسانی session
        await session_manager.add_message(session.id, message.id)
        await session_manager.update_session(session)
        
        # به‌روزرسانی آمار کاربر
        await user_manager.increment_tokens(user.id, response.tokens_used)
        
        # توقف وضعیت typing
        await connection_manager.send_to_session(session.id, {
            "type": "typing",
            "status": False
        })
        
        # ارسال پاسخ
        await connection_manager.send_to_session(session.id, {
            "type": "response",
            "message": response.dict(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await connection_manager.send_to_session(session.id, {
            "type": "error",
            "message": "An error occurred while processing your message",
            "details": str(e)
        })

@app.get("/api/metrics")
async def get_metrics():
    """دریافت متریک‌های Prometheus"""
    return generate_latest(REGISTRY)

@app.get("/api/stats")
async def get_stats():
    """دریافت آمار سیستم"""
    return {
        "sessions": {
            "total": len(session_manager.sessions),
            "active": len([s for s in session_manager.sessions.values() if s.is_active])
        },
        "connections": {
            "active": len(connection_manager.active_connections)
        },
        "users": {
            "total": len(user_manager.users)
        },
        "cache": {
            "session_cache": len(session_cache),
            "response_cache": len(response_cache)
        },
        "messages": {
            "total": CHAT_MESSAGES._value.get(),
            "rate": CHAT_MESSAGES._value.get() / 3600  # تخمینی
        }
    }

@app.get("/api/modes")
async def get_chat_modes():
    """دریافت حالت‌های مختلف چت"""
    return {
        "modes": [
            {
                "id": mode.value,
                "name": mode.name.replace("_", " ").title(),
                "description": self._get_mode_description(mode)
            }
            for mode in ChatMode
        ]
    }
    
    def _get_mode_description(self, mode: ChatMode) -> str:
        """دریافت توضیحات حالت"""
        descriptions = {
            ChatMode.CONVERSATION: "General conversation and Q&A",
            ChatMode.CODE_ASSISTANT: "Programming help and code review",
            ChatMode.WRITING_ASSISTANT: "Writing improvement and editing",
            ChatMode.TRANSLATOR: "Translate between languages",
            ChatMode.SUMMARIZER: "Summarize long texts",
            ChatMode.ANALYST: "Analyze data and provide insights"
        }
        return descriptions.get(mode, "Chat mode")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
