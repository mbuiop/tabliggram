"""
موتور یادگیری عصبی پیشرفته با قابلیت یادگیری عمیق از اسناد
و بهینه‌سازی مدل با تکنیک‌های meta-learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import asyncio
import pickle
import json
import hashlib
import os
from pathlib import Path
import logging
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import queue
import time
from enum import Enum
import math
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb
from tensorboard import SummaryWriter
import optuna
from hyperopt import fmin, tpe, hp, Trials
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

class LearningPhase(Enum):
    """فازهای یادگیری"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    ACTIVE_LEARNING = "active_learning"
    META_LEARNING = "meta_learning"
    REINFORCEMENT = "reinforcement"
    ONLINE = "online"
    BATCH = "batch"

class OptimizationStrategy(Enum):
    """استراتژی‌های بهینه‌سازی"""
    GRADIENT_DESCENT = "gradient_descent"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    REINFORCEMENT = "reinforcement"
    META = "meta"
    QUANTUM = "quantum"

@dataclass
class LearningConfig:
    """تنظیمات موتور یادگیری"""
    # پارامترهای پایه
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # استراتژی یادگیری
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.GRADIENT_DESCENT
    learning_phase: LearningPhase = LearningPhase.PRETRAIN
    
    # Meta-learning
    meta_learning_rate: float = 1e-3
    meta_batch_size: int = 4
    inner_steps: int = 5
    
    # Active learning
    uncertainty_threshold: float = 0.3
    diversity_threshold: float = 0.7
    max_active_samples: int = 1000
    
    # Reinforcement learning
    rl_gamma: float = 0.99
    rl_lambda: float = 0.95
    rl_epsilon: float = 0.2
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Logging and checkpointing
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    use_wandb: bool = False
    use_tensorboard: bool = True

class DocumentDataset(Dataset):
    """دیتاست اسناد برای آموزش"""
    
    def __init__(self, documents: List[str], tokenizer, max_length: int = 512, config: LearningConfig = None):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config
        self.indices = list(range(len(documents)))
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # توکنایز
        tokens = self.tokenizer(
            doc,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze(),  # برای language modeling
            'index': torch.tensor(idx)
        }

class MetaLearner(nn.Module):
    """یادگیرنده متا برای یادگیری چگونگی یادگیری"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # شبکه متا
        self.meta_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # حافظه متا
        self.meta_memory = nn.Parameter(torch.randn(100, hidden_dim) * 0.02)
        
        # مکانیزم توجه برای حافظه
        self.memory_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        # دریافت پارامترهای بهینه برای task
        meta_params = self.meta_network(task_embedding)
        
        # استفاده از حافظه متا
        memory_query = task_embedding.unsqueeze(1)
        memory_keys = self.meta_memory.unsqueeze(0).expand(task_embedding.size(0), -1, -1)
        attended_memory, _ = self.memory_attention(memory_query, memory_keys, memory_keys)
        
        meta_params = meta_params + attended_memory.squeeze(1)
        
        return meta_params

class ActiveLearner:
    """یادگیرنده فعال برای انتخاب هوشمندانه داده‌ها"""
    
    def __init__(self, model: nn.Module, config: LearningConfig):
        self.model = model
        self.config = config
        self.uncertainty_estimator = self._create_uncertainty_estimator()
        self.diversity_calculator = self._create_diversity_calculator()
        
    def _create_uncertainty_estimator(self):
        """ایجاد تخمین‌گر عدم قطعیت"""
        return nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _create_diversity_calculator(self):
        """ایجاد محاسبه‌گر تنوع"""
        return nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    @torch.no_grad()
    def select_samples(self, unlabeled_data: List[str], k: int = 10) -> List[int]:
        """انتخاب بهترین نمونه‌ها برای برچسب‌زنی"""
        # محاسبه embedding برای همه داده‌ها
        embeddings = self._get_embeddings(unlabeled_data)
        
        # محاسبه عدم قطعیت
        uncertainties = self.uncertainty_estimator(embeddings)
        
        # محاسبه تنوع
        diversity_features = self.diversity_calculator(embeddings)
        
        # ترکیب معیارها
        scores = uncertainties.squeeze() * self.config.uncertainty_threshold
        
        # محاسبه تنوع با حذف نمونه‌های مشابه
        selected_indices = []
        remaining_indices = list(range(len(unlabeled_data)))
        
        for _ in range(min(k, len(unlabeled_data))):
            if not remaining_indices:
                break
            
            # انتخاب بهترین نمونه بر اساس امتیاز
            best_idx = max(remaining_indices, key=lambda i: scores[i])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # کاهش امتیاز نمونه‌های مشابه
            if remaining_indices:
                best_feat = diversity_features[best_idx]
                similarities = F.cosine_similarity(
                    best_feat.unsqueeze(0),
                    diversity_features[remaining_indices]
                )
                
                for i, idx in enumerate(remaining_indices):
                    if similarities[i] > self.config.diversity_threshold:
                        scores[idx] *= 0.5
        
        return selected_indices
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """دریافت embedding برای متون"""
        # اینجا باید از مدل برای دریافت embedding استفاده شود
        # برای سادگی، بردار تصادفی برمی‌گردانیم
        return torch.randn(len(texts), 768)

class ReinforcementLearner:
    """یادگیرنده تقویتی برای بهینه‌سازی استراتژی یادگیری"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.env = self._create_environment()
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=config.learning_rate,
            gamma=config.rl_gamma,
            gae_lambda=config.rl_lambda,
            clip_range=config.rl_epsilon
        )
        
    def _create_environment(self):
        """ایجاد محیط reinforcement learning"""
        
        class LearningEnv(gym.Env):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.action_space = gym.spaces.Box(
                    low=-1, high=1, shape=(5,)
                )  # hyperparameters
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(20,)
                )  # state features
                
            def step(self, action):
                # شبیه‌سازی یک گام یادگیری
                reward = self._simulate_learning(action)
                self.step_count += 1
                done = self.step_count >= 100
                return self._get_state(), reward, done, {}
                
            def reset(self):
                self.step_count = 0
                return self._get_state()
                
            def _get_state(self):
                # دریافت وضعیت فعلی
                return np.random.randn(20)
                
            def _simulate_learning(self, action):
                # شبیه‌سازی پیشرفت یادگیری
                return float(np.random.randn())
        
        return DummyVecEnv([lambda: LearningEnv(self.config)])
    
    def optimize_hyperparameters(self, train_func, n_trials: int = 100):
        """بهینه‌سازی هایپرپارامترها با RL"""
        best_params = None
        best_reward = -float('inf')
        
        for trial in range(n_trials):
            # دریافت action از مدل
            obs = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            
            # آموزش مدل با پارامترهای پیشنهادی
            params = self._action_to_params(action)
            reward = train_func(params)
            
            if reward > best_reward:
                best_reward = reward
                best_params = params
            
            # به‌روزرسانی مدل RL
            self.model.learn(total_timesteps=1000)
        
        return best_params
    
    def _action_to_params(self, action):
        """تبدیل action به هایپرپارامتر"""
        return {
            'learning_rate': float((action[0] + 1) / 2 * 1e-3 + 1e-5),
            'batch_size': int((action[1] + 1) / 2 * 128 + 8),
            'dropout': float((action[2] + 1) / 2 * 0.3 + 0.1),
            'weight_decay': float(10 ** ((action[3] + 1) / 2 * 4 - 5)),
            'warmup_ratio': float((action[4] + 1) / 2 * 0.2)
        }

class EvolutionaryOptimizer:
    """بهینه‌ساز تکاملی برای معماری شبکه"""
    
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self, model_creator, param_space: Dict):
        """ایجاد جمعیت اولیه"""
        for _ in range(self.population_size):
            params = self._sample_params(param_space)
            model = model_creator(params)
            self.population.append({
                'params': params,
                'model': model,
                'fitness': 0.0
            })
    
    def evolve(self, fitness_func, generations: int = 100):
        """اجرای الگوریتم تکاملی"""
        for gen in range(generations):
            # ارزیابی fitness
            for individual in self.population:
                if individual['fitness'] == 0.0:
                    individual['fitness'] = fitness_func(individual['model'])
            
            # مرتب‌سازی بر اساس fitness
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # ثبت بهترین fitness
            self.fitness_history.append(self.population[0]['fitness'])
            
            # انتخاب والدین
            parents = self.population[:self.population_size // 4]
            
            # تولید نسل جدید
            new_population = parents.copy()
            
            # Crossover
            while len(new_population) < self.population_size:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                child_params = self._crossover(
                    parents[p1]['params'],
                    parents[p2]['params']
                )
                child_params = self._mutate(child_params)
                child_model = self._create_model(child_params)
                new_population.append({
                    'params': child_params,
                    'model': child_model,
                    'fitness': 0.0
                })
            
            self.population = new_population
        
        return self.population[0]['model'], self.population[0]['params']
    
    def _sample_params(self, param_space: Dict) -> Dict:
        """نمونه‌گیری تصادفی از فضای پارامتر"""
        params = {}
        for key, space in param_space.items():
            if space['type'] == 'int':
                params[key] = np.random.randint(space['min'], space['max'])
            elif space['type'] == 'float':
                params[key] = np.random.uniform(space['min'], space['max'])
            elif space['type'] == 'categorical':
                params[key] = np.random.choice(space['values'])
        return params
    
    def _crossover(self, params1: Dict, params2: Dict) -> Dict:
        """ترکیب دو فرد"""
        child = {}
        for key in params1:
            if np.random.random() < 0.5:
                child[key] = params1[key]
            else:
                child[key] = params2[key]
        return child
    
    def _mutate(self, params: Dict) -> Dict:
        """اعمال جهش"""
        mutated = params.copy()
        for key in mutated:
            if np.random.random() < self.mutation_rate:
                if isinstance(mutated[key], (int, np.integer)):
                    mutated[key] += np.random.randint(-2, 3)
                elif isinstance(mutated[key], (float, np.floating)):
                    mutated[key] *= np.random.uniform(0.8, 1.2)
        return mutated
    
    def _create_model(self, params):
        """ایجاد مدل با پارامترهای داده شده"""
        # اینجا باید مدل واقعی ساخته شود
        return None

class NeuralLearningEngine:
    """موتور اصلی یادگیری عصبی"""
    
    def __init__(self, brain: nn.Module, config: LearningConfig):
        self.brain = brain
        self.config = config
        self.phase = LearningPhase.PRETRAIN
        
        # بهینه‌ساز
        self.optimizer = torch.optim.AdamW(
            brain.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # scheduler
        self.scheduler = self._create_scheduler()
        
        # mixed precision training
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # ماژول‌های یادگیری
        self.meta_learner = MetaLearner(768, 2048, 100) if config.learning_phase == LearningPhase.META_LEARNING else None
        self.active_learner = ActiveLearner(brain, config) if config.learning_phase == LearningPhase.ACTIVE_LEARNING else None
        self.rl_learner = ReinforcementLearner(config) if config.learning_phase == LearningPhase.REINFORCEMENT else None
        self.evo_optimizer = EvolutionaryOptimizer() if config.optimization_strategy == OptimizationStrategy.EVOLUTIONARY else None
        
        # logging
        self.writer = SummaryWriter('runs/neural_learning') if config.use_tensorboard else None
        if config.use_wandb:
            wandb.init(project="neural-learning-engine", config=config.__dict__)
        
        # آمار و تاریخچه
        self.stats = {
            'epochs': 0,
            'steps': 0,
            'total_loss': 0.0,
            'best_loss': float('inf'),
            'learning_rate': [],
            'train_losses': [],
            'eval_losses': [],
            'perplexities': [],
            'gradient_norms': []
        }
        
        # صف‌ها و threads
        self.training_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Distributed training
        if config.distributed:
            self._setup_distributed()
    
    def _create_scheduler(self):
        """ایجاد scheduler برای learning rate"""
        if self.config.warmup_steps > 0:
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.total_steps
            )
        else:
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.config.total_steps
            )
    
    def _setup_distributed(self):
        """تنظیمات آموزش توزیع‌شده"""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
        
        # بسته‌بندی مدل برای distributed training
        self.brain = torch.nn.parallel.DistributedDataParallel(
            self.brain,
            device_ids=[self.config.rank],
            output_device=self.config.rank
        )
    
    async def train_on_documents(self, documents: List[str], validation_docs: Optional[List[str]] = None):
        """آموزش روی اسناد"""
        self.phase = LearningPhase.PRETRAIN
        
        # ایجاد dataset
        dataset = DocumentDataset(documents, self.brain.tokenizer, config=self.config)
        
        # ایجاد dataloader
        sampler = DistributedSampler(dataset) if self.config.distributed else None
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # حلقه آموزش
        for epoch in range(self.config.total_steps // len(dataloader) + 1):
            epoch_loss = 0.0
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                loss = await self._train_step(batch)
                epoch_loss += loss
                
                self.stats['steps'] += 1
                
                # logging
                if self.stats['steps'] % self.config.log_every == 0:
                    self._log_training(epoch, batch_idx, loss)
                
                # evaluation
                if validation_docs and self.stats['steps'] % self.config.eval_every == 0:
                    eval_loss = await self.evaluate(validation_docs)
                    self.stats['eval_losses'].append(eval_loss)
                    
                    if eval_loss < self.stats['best_loss']:
                        self.stats['best_loss'] = eval_loss
                        self.save_checkpoint('best_model.pt')
                
                # checkpoint
                if self.stats['steps'] % self.config.save_every == 0:
                    self.save_checkpoint(f'checkpoint_{self.stats["steps"]}.pt')
            
            epoch_loss /= len(dataloader)
            self.stats['epochs'] += 1
            self.stats['train_losses'].append(epoch_loss)
            
            # محاسبه perplexity
            perplexity = math.exp(epoch_loss)
            self.stats['perplexities'].append(perplexity)
            
            logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Perplexity = {perplexity:.4f}, Time = {time.time() - epoch_start:.2f}s")
    
    async def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """یک گام آموزش"""
        # انتقال به GPU
        device = next(self.brain.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass با mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.brain(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                loss = self._compute_loss(logits, labels)
            
            # Backward pass با scaler
            self.scaler.scale(loss).backward()
            
            if self.stats['steps'] % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Forward pass معمولی
            outputs = self.brain(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            loss = self._compute_loss(logits, labels)
            
            # Backward pass معمولی
            loss.backward()
            
            if self.stats['steps'] % self.config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # به‌روزرسانی scheduler
        if self.stats['steps'] % self.config.gradient_accumulation_steps == 0:
            self.scheduler.step()
        
        # ذخیره آمار
        self.stats['total_loss'] += loss.item()
        if 'grad_norm' in locals():
            self.stats['gradient_norms'].append(grad_norm.item())
        
        return loss.item()
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """محاسبه loss با label smoothing"""
        vocab_size = logits.shape[-1]
        
        # label smoothing
        smooth_factor = self.config.label_smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        
        if labels.dim() == logits.dim() - 1:
            labels = labels.unsqueeze(-1)
        
        nll_loss = -log_probs.gather(dim=-1, index=labels)
        smooth_loss = -log_probs.mean(dim=-1, keepdim=True)
        
        loss = (1 - smooth_factor) * nll_loss + smooth_factor * smooth_loss
        return loss.mean()
    
    async def evaluate(self, documents: List[str]) -> float:
        """ارزیابی مدل روی داده‌های validation"""
        self.brain.eval()
        
        dataset = DocumentDataset(documents, self.brain.tokenizer, config=self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                device = next(self.brain.parameters()).device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.brain(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                loss = self._compute_loss(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.brain.train()
        return total_loss / num_batches
    
    async def meta_learn(self, tasks: List[Dict]):
        """یادگیری متا روی چندین task"""
        self.phase = LearningPhase.META_LEARNING
        
        for task_batch in self._batch_tasks(tasks, self.config.meta_batch_size):
            # Inner loop - یادگیری روی هر task
            task_grads = []
            
            for task in task_batch:
                # کپی از مدل برای inner loop
                inner_model = copy.deepcopy(self.brain)
                inner_optimizer = torch.optim.SGD(
                    inner_model.parameters(),
                    lr=self.config.meta_learning_rate
                )
                
                # چند گام روی task
                for _ in range(self.config.inner_steps):
                    loss = await self._compute_task_loss(inner_model, task)
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
                
                # محاسبه گرادیان متا
                meta_loss = await self._compute_task_loss(inner_model, task['val'])
                meta_loss.backward()
                
                # جمع‌آوری گرادیان‌ها
                task_grads.append([p.grad.clone() for p in self.brain.parameters()])
            
            # به‌روزرسانی مدل اصلی با گرادیان‌های متا
            self.optimizer.zero_grad()
            for param, grads in zip(self.brain.parameters(), zip(*task_grads)):
                param.grad = torch.stack(grads).mean(dim=0)
            self.optimizer.step()
    
    async def _compute_task_loss(self, model: nn.Module, task: Dict) -> torch.Tensor:
        """محاسبه loss برای یک task خاص"""
        # اینجا باید loss متناسب با task محاسبه شود
        return torch.tensor(0.0)
    
    def _batch_tasks(self, tasks: List, batch_size: int):
        """تقسیم tasks به batch‌ها"""
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    
    def active_learning_cycle(self, unlabeled_data: List[str], label_func, n_rounds: int = 10):
        """چرخه یادگیری فعال"""
        self.phase = LearningPhase.ACTIVE_LEARNING
        
        labeled_data = []
        
        for round in range(n_rounds):
            # انتخاب نمونه‌های نامطمئن
            selected_indices = self.active_learner.select_samples(
                unlabeled_data,
                k=self.config.max_active_samples // n_rounds
            )
            
            # برچسب‌زنی نمونه‌های انتخاب شده
            for idx in selected_indices:
                label = label_func(unlabeled_data[idx])
                labeled_data.append({
                    'text': unlabeled_data[idx],
                    'label': label
                })
            
            # حذف نمونه‌های انتخاب شده از unlabeled
            unlabeled_data = [d for i, d in enumerate(unlabeled_data) if i not in selected_indices]
            
            # آموزش روی داده‌های برچسب‌دار
            if labeled_data:
                texts = [item['text'] for item in labeled_data]
                asyncio.run(self.train_on_documents(texts))
    
    def optimize_with_evolution(self, model_creator, param_space: Dict, fitness_func, generations: int = 100):
        """بهینه‌سازی با الگوریتم تکاملی"""
        self.phase = LearningPhase.META_LEARNING
        
        self.evo_optimizer.initialize_population(model_creator, param_space)
        best_model, best_params = self.evo_optimizer.evolve(fitness_func, generations)
        
        return best_model, best_params
    
    def hyperparameter_optimization(self, train_func, n_trials: int = 100):
        """بهینه‌سازی هایپرپارامترها"""
        if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN:
            return self._bayesian_optimization(train_func, n_trials)
        elif self.config.optimization_strategy == OptimizationStrategy.REINFORCEMENT:
            return self.rl_learner.optimize_hyperparameters(train_func, n_trials)
        else:
            return self._grid_search(train_func)
    
    def _bayesian_optimization(self, train_func, n_trials: int):
        """بهینه‌سازی بیزین با Optuna"""
        
        def objective(trial):
            # تعریف فضای جستجو
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
                'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
                'num_layers': trial.suggest_int('num_layers', 6, 24),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [1024, 2048, 4096])
            }
            
            return train_func(params)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def _grid_search(self, train_func):
        """جستجوی Grid ساده"""
        # اینجا می‌توانید grid search ساده پیاده‌سازی کنید
        pass
    
    def _log_training(self, epoch: int, batch_idx: int, loss: float):
        """ثبت لاگ‌های آموزش"""
        step = self.stats['steps']
        lr = self.scheduler.get_last_lr()[0]
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/train', loss, step)
            self.writer.add_scalar('LR', lr, step)
            if self.stats['gradient_norms']:
                self.writer.add_scalar('Grad/norm', self.stats['gradient_norms'][-1], step)
        
        # WandB
        if self.config.use_wandb:
            wandb.log({
                'train/loss': loss,
                'train/lr': lr,
                'train/epoch': epoch,
                'train/step': step
            })
        
        # Console
        if step % (self.config.log_every * 10) == 0:
            logger.info(
                f"Step {step}: loss = {loss:.4f}, lr = {lr:.6f}, "
                f"grad_norm = {self.stats['gradient_norms'][-1] if self.stats['gradient_norms'] else 0:.4f}"
            )
    
    def save_checkpoint(self, filename: str):
        """ذخیره چک‌پوینت"""
        checkpoint = {
            'model_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stats': self.stats,
            'config': self.config,
            'phase': self.phase.value
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save(checkpoint, f'checkpoints/{filename}')
        logger.info(f"💾 Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """بارگذاری چک‌پوینت"""
        checkpoint = torch.load(f'checkpoints/{filename}')
        
        self.brain.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.stats = checkpoint['stats']
        self.phase = LearningPhase(checkpoint['phase'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"📂 Checkpoint loaded: {filename}")
    
    def get_statistics(self) -> Dict:
        """دریافت آمار"""
        return {
            'phase': self.phase.value,
            'epochs': self.stats['epochs'],
            'steps': self.stats['steps'],
            'total_loss': self.stats['total_loss'],
            'best_loss': self.stats['best_loss'],
            'current_lr': self.scheduler.get_last_lr()[0],
            'perplexity': self.stats['perplexities'][-1] if self.stats['perplexities'] else 0,
            'memory_usage': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }

# نمونه‌سازی و تست
if __name__ == "__main__":
    # ایجاد مغز و موتور یادگیری
    config = LearningConfig()
    
    # تست با داده‌های نمونه
    documents = [
        "هوش مصنوعی شاخه‌ای از علوم کامپیوتر است.",
        "یادگیری عمیق با استفاده از شبکه‌های عصبی انجام می‌شود.",
        "پردازش زبان طبیعی به ماشین‌ها کمک می‌کند متن را بفهمند."
    ]
    
    async def test():
        from core_quantum_brain import QuantumBrain, QuantumConfig
        
        brain_config = QuantumConfig()
        brain = QuantumBrain(brain_config)
        
        engine = NeuralLearningEngine(brain, config)
        
        print("شروع آموزش...")
        await engine.train_on_documents(documents)
        
        print("آموزش کامل شد!")
        print(engine.get_statistics())
    
    # asyncio.run(test())
