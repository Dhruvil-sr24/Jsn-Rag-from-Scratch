## JSN
import numpy as np
import faiss
import pickle
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set
import re
from dataclasses import dataclass, field
import openai  # Using OpenAI for both LLM and embeddings
from datetime import datetime
import time
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from urllib.parse import urlparse


# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# OPENAI EMBEDDING WRAPPER
# ============================================================================

class OpenAIEmbeddingModel:
    """OpenAI embedding model wrapper to replace SentenceTransformer"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY', 'key-here')
        
        )
        self.model_name = model_name
        self.device = 'cpu'  # For compatibility
        
    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = False) -> np.ndarray:
        """Encode texts using OpenAI embeddings API"""
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if show_progress_bar and i % batch_size == 0:
                    logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Normalize if requested
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                embeddings_array = embeddings_array / norms
                
            return embeddings_array
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            # Fallback: return random embeddings with correct shape
            return np.random.rand(len(texts), 1536).astype(np.float32)

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class ChatRequest(BaseModel):
    question: str
    session_id: str = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    conversation_stage: str
    confidence_level: float
    chat_history: List[Dict]
    services: List[Dict] = []
    related_works: List[Dict] = []
    source_documents: List[Dict] = []
    suggested_questions: List[str] = []
    debug_info: Optional[Dict] = None

# ============================================================================
# CONVERSATION CONTEXT AND DATA STRUCTURES
# ============================================================================

@dataclass
class ConversationContext:
    """Rich conversation context similar to commercial chatbots"""

    # User understanding
    user_intent: str = None
    primary_intent: str = "discovery"
    confidence_level: float = 0.0
    conversation_stage: str = "discovery"
    
    pending_confirmation: bool = False
    user_expertise_level: str = "intermediate"  

    # Dynamic context
    active_topics: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_tone: str = "professional"
    emotional_state: str = "neutral"

    # Business context
    business_context: Optional[str] = None
    extracted_needs: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    business_goals: List[str] = field(default_factory=list)
    urgency_level: str = "medium"

    # Memory and history
    key_information: Dict[str, Any] = field(default_factory=dict)
    mentioned_entities: Set[str] = field(default_factory=set)
    conversation_history_summary: str = ""
    previous_questions: List[str] = field(default_factory=list)

    # Adaptive behavior
    response_style: str = "balanced"
    user_expertise_level: str = "intermediate"
    preferred_communication_style: str = "direct"
    response_length_preference: str = "medium"

    # Engagement tracking
    questions_asked: int = 0
    user_engagement_level: str = "medium"

    clarification_requests: int = 0

@dataclass
class Document:
    """Proper Document class that can be pickled"""
    text: str
    metadata: Dict[str, Any]
    source_urls: Set[str]

    def __post_init__(self):
        """Ensure source_urls is a set"""
        if not isinstance(self.source_urls, set):
            self.source_urls = set(self.source_urls) if self.source_urls else set()

# ============================================================================
# ADVANCED TOPIC CHANGE DETECTION SYSTEM
# ============================================================================

class AdvancedTopicDetector:
    """Dynamic topic change detection using semantic analysis"""

    def __init__(self):
        self.semantic_model = None  # Lazy loading
        self.topic_coherence_threshold = 0.6
        self.conversation_window = 3

    def _get_semantic_model(self):
        """Lazy load semantic model - NOW USING OPENAI"""
        if self.semantic_model is None:
            self.semantic_model = OpenAIEmbeddingModel('text-embedding-3-small')
        return self.semantic_model

    def _calculate_semantic_drift(self, current_message: str, history: List[str]) -> float:
        """Calculate semantic drift from recent conversation"""

        try:
            recent_history = history[-self.conversation_window:]

            # Create embeddings using OpenAI
            model = self._get_semantic_model()
            current_embedding = model.encode([current_message])
            history_embeddings = model.encode(recent_history)

            # Calculate average similarity with recent messages
            similarities = cosine_similarity(current_embedding, history_embeddings)[0]
            avg_similarity = np.mean(similarities)

            # Convert similarity to drift score (inverse relationship)
            drift_score = 1.0 - avg_similarity

            return np.clip(drift_score, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Semantic drift calculation error: {e}")
            return 0.0

    # In the AdvancedTopicDetector class, replace the entire _analyze_contextual_coherence function with this new version:

    def _analyze_contextual_coherence(self, message: str, context: ConversationContext) -> float:
        """
        Analyze coherence with established conversation context.
        Returns a DIVERGENCE score (0.0 = perfect coherence, 1.0 = total divergence).
        This version is smarter about contextual keywords.
        """
        if not context.business_context and not context.extracted_needs and not context.active_topics:
            return 0.0  # No context to be incoherent with

        message_lower = message.lower()
        model = self._get_semantic_model()
        message_embedding = model.encode([message_lower])
        divergence_scores = []

        try:
            # --- Semantic Coherence Check ---
            # Combine all existing context into a single string for comparison
            context_keywords = (context.business_context or '').split() + \
                            [need for need in context.extracted_needs] + \
                            [topic for topic in context.active_topics]
            context_text = ' '.join(set(context_keywords))

            if context_text.strip():
                context_embedding = model.encode([context_text.lower()])
                # Calculate the base divergence using semantic similarity
                base_divergence = 1.0 - cosine_similarity(message_embedding, context_embedding)[0][0]
                divergence_scores.append(base_divergence)

                # --- Keyword Coherence Check ---
                # If the user's message contains keywords from the existing context, they are likely elaborating, not changing topics.
                # We will reward this by reducing the divergence score.
                keyword_overlap_count = sum(1 for keyword in context_keywords if keyword in message_lower)
                if keyword_overlap_count > 0:
                    # Apply a "coherence bonus" by reducing the divergence score.
                    # The more keywords overlap, the bigger the reduction.
                    coherence_bonus = min(0.3, keyword_overlap_count * 0.15)
                    base_divergence -= coherence_bonus
                    logger.info(f"✅ Contextual keyword overlap found. Applying coherence bonus of {coherence_bonus:.2f}.")
                
                # Recalculate the final score after the bonus
                final_divergence_score = np.clip(base_divergence, 0.0, 1.0)
                return final_divergence_score

            if not divergence_scores:
                return 0.0

            return np.clip(np.mean(divergence_scores), 0.0, 1.0)

        except Exception as e:
            logger.error(f"Contextual coherence error: {e}")
            return 0.5  # Return a neutral score on error

    def _detect_linguistic_patterns(self, message: str, history: List[str]) -> float:
        """Detect linguistic patterns indicating topic change"""

        message_lower = message.lower()
        linguistic_score = 0.0

        try:
            # Dynamic transition phrase detection
            transition_patterns = {
                'explicit': [
                    # Existing patterns are good
                    r'\b(actually|instead|wait|no|different|change)\b',
                    r'\b(switch to|move to|talk about|focus on)\b',
                    r'\b(forget that|nevermind|new topic|let\'s talk about something else)\b'
                ],
                'implicit': [
                    # Added phrases related to expanding scope
                    r'\b(also|additionally|another|different|what else)\b',
                    r'\b(what about|how about|regarding|in terms of)\b',
                    r'\b(i need|i want|looking for|can you build|do you do)\b',
                    r'\b(can you also|what if we add|and we also need)\b'
                ],
                'conversational': [
                    # Existing patterns are good
                    r'\b(by the way|oh|actually|well)\b',
                    r'\b(speaking of|that reminds me)\b',
                    r'\b(let\'s clarify|to be specific|so just to be clear)\b'
                ],
                'scope_change': [
                    # New category for detecting shifts to budget, timeline, etc.
                    r'\b(how much|cost|pricing|budget|price range)\b',
                    r'\b(timeline|how long|duration|time frame)\b',
                    r'\b(the goal is|our main objective)\b'
                ]
            }

            # Score based on transition strength
            for category, patterns in transition_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        if category == 'explicit':
                            linguistic_score += 0.6
                        elif category == 'implicit':
                            linguistic_score += 0.3
                        else:
                            linguistic_score += 0.2
                        break

            # Analyze sentence structure changes
            if len(history) > 0:
                recent_message = history[-1].lower()

                # Check for sudden formality changes
                formality_shift = self._detect_formality_shift(message_lower, recent_message)
                linguistic_score += formality_shift * 0.2

        except Exception as e:
            logger.error(f"Linguistic pattern detection error: {e}")

        return np.clip(linguistic_score, 0.0, 1.0)

    def _analyze_intent_shift(self, message: str, context: ConversationContext) -> float:
        """Analyze intent shift from previous interactions"""

        message_lower = message.lower()
        intent_score = 0.0

        try:
            # Service category detection
            service_categories = {
                'technical_development': ['development', 'software', 'app development', 'web development', 'game development', 'saas'],
                'ai_and_data': ['ai', 'machine learning', 'data science', 'analytics', 'chatbot', 'ai development'],
                'emerging_tech': ['blockchain', 'web3', 'crypto', 'dapp', 'nft'],
                'design_and_creative': ['design', 'branding', 'ui/ux', 'graphics', 'logo', 'motion graphics', 'canva', 'visual design'],
                'business_and_strategy': ['marketing', 'seo', 'strategy', 'consulting', 'business plan', 'digital marketing'],
                'operations_and_support': ['maintenance', 'automation', 'integration', 'support', 'website maintenance'],
            }

            current_category = None
            for category, keywords in service_categories.items():
                if any(keyword in message_lower for keyword in keywords):
                    current_category = category
                    break

            # Compare with previous context
            if context.business_context and current_category:
                previous_category = self._categorize_business_context(context.business_context)
                if previous_category and previous_category != current_category:
                    intent_score += 0.7

        except Exception as e:
            logger.error(f"Intent shift analysis error: {e}")

        return np.clip(intent_score, 0.0, 1.0)

    def _get_adaptive_threshold(self, context: ConversationContext) -> float:
        """Dynamic threshold based on conversation state"""

        base_threshold = 0.25 #0.5 

        # Adjust based on conversation stage
        stage_adjustments = {
            'discovery': -0.1,
            'qualification': -0.05,
            'recommendation': 0.1,
            'closing': 0.15
        }

        threshold = base_threshold + stage_adjustments.get(context.conversation_stage, 0)

        # Adjust based on conversation depth
        if context.questions_asked > 4:
            threshold += 0.05

        # Adjust based on confidence level
        if context.confidence_level < 0.3:
            threshold -= 0.1

        return np.clip(threshold, 0.2, 0.8)

    def _get_domain_keywords(self, business_context: str) -> List[str]:
        """Get domain-specific keywords for coherence checking"""

        domain_keywords = {
            'seo': ['search', 'ranking', 'optimization', 'google', 'keywords', 'traffic', 'serp', 'on-page', 'off-page'],
            'blockchain': ['crypto', 'web3', 'smart contract', 'defi', 'nft', 'ethereum', 'dapp', 'token', 'coin'],
            'ai': ['artificial intelligence', 'machine learning', 'neural', 'algorithm', 'generative ai', 'chatbot', 'rasa', 'dialogflow', 'data science'],
            'marketing': ['campaign', 'advertising', 'promotion', 'brand', 'engagement', 'smm', 'ppc', 'content marketing'],
            'website': ['web', 'site', 'frontend', 'backend', 'responsive', 'ui', 'ecommerce', 'shopify', 'wordpress', 'laravel', 'php'],
            'app_development': ['mobile app', 'ios', 'android', 'react native', 'flutter', 'hybrid app', 'application'],
            'ui_ux': ['user interface', 'user experience', 'wireframe', 'prototype', 'visual design', 'user journey', 'figma'],
            'game_development': ['game', 'gaming', 'unity', 'unreal engine', '2d', '3d'],
            'ecommerce': ['online store', 'shop', 'marketplace', 'shopify', 'woocommerce', 'payment gateway']
        }

        return domain_keywords.get(business_context.lower(), [])

    def _detect_formality_shift(self, current: str, previous: str) -> float:
        """Detect sudden changes in formality level"""

        formal_indicators = ['please', 'would', 'could', 'thank you', 'appreciate']
        informal_indicators = ['hey', 'yeah', 'ok', 'cool', 'awesome']

        current_formal = sum(1 for word in formal_indicators if word in current)
        current_informal = sum(1 for word in informal_indicators if word in current)

        previous_formal = sum(1 for word in formal_indicators if word in previous)
        previous_informal = sum(1 for word in informal_indicators if word in previous)

        # Calculate formality scores
        current_formality = (current_formal - current_informal) / max(len(current.split()), 1)
        previous_formality = (previous_formal - previous_informal) / max(len(previous.split()), 1)

        return abs(current_formality - previous_formality)

    def _categorize_business_context(self, business_context: str) -> str:
        """Categorize business context into broader categories"""

        categories = {
            'technical': ['seo', 'blockchain', 'ai', 'development', 'software', 'app', 'web', 'game'],
            'creative': ['design', 'branding', 'graphics', 'ui', 'ux', 'motion'],
            'business': ['marketing', 'strategy', 'consulting', 'analytics'],
            'operational': ['automation', 'workflow', 'support', 'maintenance']
        }

        for category, keywords in categories.items():
            if business_context.lower() in keywords:
                return category

        return 'general'

    def _get_change_reason(self, score: float, threshold: float) -> str:
        """Provide human-readable reason for topic change detection"""

        if score > threshold + 0.2:
            return "strong_topic_shift"
        elif score > threshold:
            return "moderate_topic_shift"
        elif score > threshold - 0.1:
            return "potential_topic_shift"
        else:
            return "topic_continuation"

# ============================================================================
# LINK DEDUPLICATION SYSTEM
# ============================================================================

class LinkDeduplicationManager:
    """Manages link deduplication across services and work links"""

    def __init__(self):
        self.seen_urls = set()
        self.url_similarity_threshold = 0.8

    def reset_session(self):
        """Reset for new conversation session"""
        self.seen_urls.clear()

    def is_duplicate_url(self, url: str) -> bool:
        """Check if URL is duplicate or very similar"""

        url_clean = self._normalize_url(url)

        # Direct duplicate check
        if url_clean in self.seen_urls:
            return True

        # Similarity check for near-duplicates
        for seen_url in self.seen_urls:
            if self._calculate_url_similarity(url_clean, seen_url) > self.url_similarity_threshold:
                return True

        return False

    def add_url(self, url: str) -> bool:
        """Add URL if not duplicate. Returns True if added, False if duplicate"""

        if self.is_duplicate_url(url):
            return False

        self.seen_urls.add(self._normalize_url(url))
        return True

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison"""

        # Remove protocol and www
        normalized = url.lower()
        normalized = re.sub(r'^https?://', '', normalized)
        normalized = re.sub(r'^www\.', '', normalized)

        # Remove trailing slash
        normalized = normalized.rstrip('/')

        # Remove query parameters and fragments
        normalized = re.sub(r'[?#].*$', '', normalized)

        return normalized

    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """Calculate similarity between two URLs"""

        # Simple similarity based on common parts
        parts1 = set(url1.split('/'))
        parts2 = set(url2.split('/'))

        if not parts1 or not parts2:
            return 0.0

        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))

        return intersection / union if union > 0 else 0.0

# ============================================================================
# ADVANCED MEMORY SYSTEM
# ============================================================================

class ConversationMemory:
    """Advanced memory system similar to ChatGPT/Claude"""

    def __init__(self):
        self.episodic_memory = []
        self.semantic_memory = {}
        self.working_memory = {}
        self.conversation_patterns = {}

    def update_memory(self, question: str, bot_response: str, context: ConversationContext):
        """Update all memory systems"""

        # Extract and store key information
        key_info = self._extract_key_information(question, bot_response)
        self.semantic_memory.update(key_info)

        # Update episodic memory with conversation flow
        episode = {
            'timestamp': datetime.now(),
            'user_intent': context.primary_intent,
            'key_topics': context.active_topics.copy(),
            'question': question,
            'bot_response': bot_response,
            'emotional_state': context.emotional_state,
            'conversation_stage': context.conversation_stage
        }
        self.episodic_memory.append(episode)

        # Keep only recent episodes for performance
        if len(self.episodic_memory) > 20:
            self.episodic_memory = self.episodic_memory[-20:]

        # Update working memory for immediate context
        self.working_memory = self._update_working_memory(context)

        # Learn user communication patterns
        self._update_communication_patterns(question, context)

    def _extract_key_information(self, question: str, bot_response: str) -> Dict[str, Any]:
        """Extract key information from conversation"""
        key_info = {}

        # Extract business information
        if any(word in question.lower() for word in ['business', 'company', 'startup']):
            words = question.lower().split()
            for i, word in enumerate(words):
                if word in ['business', 'company'] and i < len(words) - 1:
                    key_info['business_type'] = words[i+1]

        # Extract needs and requirements
        need_keywords = ['need', 'want', 'looking for', 'require', 'help with']
        for keyword in need_keywords:
            if keyword in question.lower():
                parts = question.lower().split(keyword)
                if len(parts) > 1:
                    need = parts[1].strip()[:50]
                    key_info['stated_need'] = need
                break

        return key_info

    def _update_working_memory(self, context: ConversationContext) -> Dict[str, Any]:
        """Update working memory with current context"""
        return {
            'current_intent': context.primary_intent,
            'active_topics': context.active_topics[-5:],
            'current_stage': context.conversation_stage,
            'user_mood': context.emotional_state,
            'recent_needs': context.extracted_needs[-5:] if context.extracted_needs else []
        }

    def _update_communication_patterns(self, question: str, context: ConversationContext):
        """Learn user communication patterns"""

        message_length = len(question.split())

        # Track message length preferences
        if 'avg_message_length' not in self.conversation_patterns:
            self.conversation_patterns['avg_message_length'] = message_length
        else:
            current_avg = self.conversation_patterns['avg_message_length']
            self.conversation_patterns['avg_message_length'] = (current_avg + message_length) / 2

        # Track communication style
        if any(word in question.lower() for word in ['please', 'thank you', 'thanks']):
            self.conversation_patterns['politeness_level'] = 'high'
        elif any(word in question for word in ['!', '?', 'urgent', 'asap']):
            self.conversation_patterns['urgency_indicators'] = True

    def get_relevant_context(self, current_query: str, context: ConversationContext) -> str:
        """Retrieve relevant context for response generation"""

        # Find relevant past conversations
        relevant_episodes = self._find_relevant_episodes(current_query)

        # Extract relevant semantic knowledge
        relevant_facts = self._extract_relevant_facts(current_query)

        # Build contextual summary
        return self._build_contextual_summary(relevant_episodes, relevant_facts, context)

    def _find_relevant_episodes(self, query: str) -> List[Dict]:
        """Find relevant episodes from conversation history"""
        query_words = set(query.lower().split())
        relevant = []

        for episode in self.episodic_memory[-10:]:
            episode_words = set(episode['question'].lower().split())
            overlap = len(query_words.intersection(episode_words))

            if overlap > 0:
                relevant.append(episode)

        return relevant[-3:]

    def _extract_relevant_facts(self, query: str) -> Dict[str, Any]:
        """Extract relevant facts from semantic memory"""
        relevant = {}
        query_lower = query.lower()

        for key, value in self.semantic_memory.items():
            if any(word in query_lower for word in key.lower().split('_')):
                relevant[key] = value

        return relevant

    def _build_contextual_summary(self, episodes: List[Dict], facts: Dict, context: ConversationContext) -> str:
        """Build contextual summary for response generation"""

        summary_parts = []

        # Add conversation stage context
        summary_parts.append(f"Conversation stage: {context.conversation_stage}")

        # Add user context
        if context.business_context:
            summary_parts.append(f"User business: {context.business_context}")

        if context.extracted_needs:
            summary_parts.append(f"User needs: {', '.join(context.extracted_needs[-3:])}")

        # Add relevant facts
        if facts:
            fact_summary = ", ".join([f"{k}: {v}" for k, v in facts.items()])
            summary_parts.append(f"Known facts: {fact_summary}")

        # Add recent conversation flow
        if episodes:
            recent_topics = [ep.get('user_intent', 'unknown') for ep in episodes]
            summary_parts.append(f"Recent discussion: {' -> '.join(recent_topics)}")

        return " | ".join(summary_parts)

# ============================================================================
# INTELLIGENT WORK LINKS SYSTEM
# ============================================================================

class IntelligentWorkLinksManager:
    """Intelligent work links manager that analyzes URLs for contextual matching"""

    def __init__(self):
        # Work links database - Add your actual work links here
        self.work_links_database = [
            # "https://10turtle.com/work/e-commerce-tech-software-ux-ui-design-website-development-beawhale-a-whale-of-innovation",
            # "https://10turtle.com/work/automotive-e-commerce-ux-ui-design-website-development-dees-organics-e-commerce-excellence-platform-for-the-lymphatic-brush",
        ]

        # Technology and service keywords for intelligent matching
        self.technology_keywords = {
            # Web & App
            'web_development': ['website', 'web', 'development', 'frontend', 'backend', 'cms', 'php', 'laravel', 'wordpress', 'custom web app'],
            'mobile': ['app', 'mobile', 'ios', 'android', 'application', 'react native', 'flutter', 'hybrid app'],
            'ecommerce': ['e-commerce', 'ecommerce', 'shop', 'store', 'commerce', 'retail', 'shopify', 'woocommerce', 'magento'],
            'game_development': ['game', 'gaming', 'unity', 'unreal engine', '2d game', '3d game'],
            'software': ['software', 'tech', 'technology', 'system', 'platform', 'saas'],

            # Design
            'design': ['ux', 'ui', 'design', 'visual', 'graphics', 'branding', 'wireframe', 'prototype', 'user experience'],
            'canva': ['canva', 'canva design', 'canva templates', 'presentations'],
            'motion_design': ['motion', 'animation', 'lottie', 'video animation'],

            # AI & Data
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'generative ai', 'ai solution'],
            'chatbot_tech': ['chatbot', 'conversational ai', 'dialogflow', 'rasa', 'virtual assistant'],
            'data_science': ['data science', 'analytics', 'predictive model', 'data visualization'],

            # Other Key Areas
            'blockchain': ['blockchain', 'crypto', 'web3', 'smart contract', 'defi', 'dapp', 'nft'],
            'seo': ['seo', 'search engine', 'optimization', 'ranking', 'google ranking'],

            # Industries (as tech context)
            'automotive': ['automotive', 'car', 'vehicle', 'auto', 'transport'],
            'healthcare': ['healthcare', 'medical', 'health', 'clinic', 'patient', 'hipaa'],
            'restaurant': ['restaurant', 'food', 'dining', 'cafe', 'culinary', 'food delivery'],
            'finance': ['finance', 'financial', 'banking', 'fintech', 'payment', 'insurtech'],
        }

        # Industry-specific keywords
        self.industry_keywords = {
                'restaurant': ['food', 'dining', 'menu', 'cafe', 'culinary', 'restaurant', 'qsr', 'food delivery'],
                'automotive': ['automotive', 'car', 'vehicle', 'auto', 'dealer', 'car parts', 'dealership'],
                'healthcare': ['medical', 'health', 'clinic', 'doctor', 'patient', 'wellness', 'telemedicine', 'hipaa'],
                'ecommerce': ['shop', 'store', 'retail', 'commerce', 'selling', 'marketplace', 'd2c', 'b2b'],
                'technology': ['tech', 'software', 'digital', 'innovation', 'system', 'saas', 'startup'],
                'finance': ['financial', 'money', 'investment', 'banking', 'insurance', 'fintech', 'insurtech', 'lending'],
                'blockchain': ['blockchain', 'crypto', 'web3', 'defi', 'nft', 'cryptocurrency', 'token'],
                'education': ['education', 'edtech', 'learning', 'e-learning', 'school', 'university'],
                'real_estate': ['real estate', 'realty', 'property', 'real estate agent'],
                'travel': ['travel', 'hospitality', 'tourism', 'booking', 'hotel'],
                'entertainment': ['entertainment', 'media', 'gaming', 'streaming'],
                'logistics': ['logistics', 'supply chain', 'transportation', 'shipping'],
                'manufacturing': ['manufacturing', 'industrial', 'factory', 'production'],
            }

    def find_relevant_work_links(self, context: ConversationContext, query: str = "") -> List[Dict[str, Any]]:
        """Find relevant work links by analyzing URL content against user context"""

        relevant_works = []

        for work_url in self.work_links_database:
            relevance_score = self._calculate_work_link_relevance(work_url, context, query)

            if relevance_score > 0.3:
                work_info = self._extract_work_info_from_url(work_url, relevance_score, context)
                relevant_works.append(work_info)

        # Sort by relevance score
        relevant_works.sort(key=lambda x: x['relevance_score'], reverse=True)

        return relevant_works[:4]

    def _calculate_work_link_relevance(self, work_url: str, context: ConversationContext, query: str) -> float:
        """Calculate relevance score for work link based on URL analysis"""

        url_lower = work_url.lower()
        relevance_score = 0.0

        # Extract URL components for analysis
        url_parts = url_lower.replace('https://10turtle.com/work/', '').split('-')
        url_text = ' '.join(url_parts)

        # Business context matching (highest weight)
        if context.business_context:
            business_lower = context.business_context.lower()

            # Direct business type match in URL
            if business_lower in url_text:
                relevance_score += 0.8

            # Industry keyword matching
            if business_lower in self.industry_keywords:
                industry_keywords = self.industry_keywords[business_lower]
                for keyword in industry_keywords:
                    if keyword in url_text:
                        relevance_score += 0.6
                        break

        # User needs matching (high weight)
        if context.extracted_needs:
            for i, need in enumerate(context.extracted_needs[:3]):
                need_lower = need.lower()
                weight = 0.7 - (i * 0.1)

                # Direct need match
                if need_lower in url_text:
                    relevance_score += weight

                # Need keyword expansion
                need_words = need_lower.split()
                for word in need_words:
                    if len(word) > 3 and word in url_text:
                        relevance_score += weight * 0.3

        # Technology/service matching
        if context.active_topics:
            for topic in context.active_topics:
                topic_lower = topic.lower()

                # Direct topic match
                if topic_lower in url_text:
                    relevance_score += 0.5

                # Technology keyword matching
                if topic_lower in self.technology_keywords:
                    tech_keywords = self.technology_keywords[topic_lower]
                    for keyword in tech_keywords:
                        if keyword in url_text:
                            relevance_score += 0.4
                            break

        # Query terms matching
        if query:
            query_words = query.lower().split()
            for word in query_words:
                if len(word) > 3 and word in url_text:
                    relevance_score += 0.3

        return min(relevance_score, 1.0)

    def _extract_work_info_from_url(self, work_url: str, relevance_score: float, context: ConversationContext) -> Dict[str, Any]:
        """Extract meaningful work information from URL structure"""

        # Extract project components from URL
        url_path = work_url.replace('https://10turtle.com/work/', '')
        url_parts = url_path.split('-')

        # Extract project name
        project_name = "Featured Project"
        if len(url_parts) > 5:
            potential_names = []
            for part in url_parts:
                if len(part) > 4 and part not in ['website', 'development', 'design', 'ecommerce', 'software']:
                    potential_names.append(part.title())

            if potential_names:
                project_name = ' '.join(potential_names[-3:])

        # Extract technologies
        technologies = []
        for tech_category, keywords in self.technology_keywords.items():
            for keyword in keywords:
                if keyword in url_path.lower():
                    tech_display = keyword.replace('-', ' ').title()
                    if tech_display not in technologies:
                        technologies.append(tech_display)

        # Extract industry
        industry = "General Business"
        for industry_type, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in url_path.lower():
                    industry = industry_type.title()
                    break
            if industry != "General Business":
                break

        # Generate contextual description
        description = self._generate_work_description(project_name, technologies, industry, context)

        return {
            'title': project_name,
            'url': work_url,
            'technologies': technologies[:4],
            'industry': industry,
            'description': description,
            'relevance_score': round(relevance_score, 3)
        }

    def _generate_work_description(self, project_name: str, technologies: List[str],
                                 industry: str, context: ConversationContext) -> str:
        """Generate contextual description for work project"""

        tech_str = ", ".join(technologies[:3])

        # Context-aware descriptions
        if context.business_context:
            if context.business_context.lower() == industry.lower():
                return f"Perfect match for your {context.business_context} business - {project_name} showcases {tech_str} expertise"
            else:
                return f"{project_name} - {industry} project featuring {tech_str} solutions"

        if context.extracted_needs:
            primary_need = context.extracted_needs[0].lower()
            if any(tech.lower() in primary_need for tech in technologies):
                return f"Relevant to your {primary_need} needs - {project_name} demonstrates {tech_str}"

        return f"{project_name} - {industry} project showcasing {tech_str}"

# ============================================================================
# ADVANCED QUALIFICATION ENGINE WITH TOPIC DETECTION
# ============================================================================

class AdvancedQualificationEngine:
    """Intent-driven qualification engine with robust topic change detection"""

    def __init__(self, llm_model: str = "gpt-4o"):
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY', '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')
        )
        self.model_name = llm_model

        self.memory = ConversationMemory()
        self.topic_detector = AdvancedTopicDetector()
        

    def _is_positive_confirmation(self, user_message: str, confirmation_question: str) -> bool:
        """Uses the LLM to determine if the user's message is a positive confirmation."""
        
        prompt = f"""
        A chatbot asked the following confirmation question:
        "{confirmation_question}"

        The user responded with:
        "{user_message}"

        Is the user's response a positive confirmation (e.g., 'yes', 'correct', 'that's right', 'both of them')?
        Respond with only the word "YES" or "NO".
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # Zero temperature for reliable classification
                max_tokens=5
            )
            decision = response.choices[0].message.content.strip().upper()
            logger.info(f"🧠 Confirmation check: Is '{user_message}' a YES? -> {decision}")
            return "YES" in decision
        except Exception as e:
            logger.error(f"Confirmation check failed: {e}")
            # Default to a safe 'no' on error
            return False

    def _generate_smart_search_query(self, context: ConversationContext, last_user_query: str) -> str:
        """
        Uses the LLM to synthesize the complete conversation context into a high-relevance search query,
        now with more robust handling for multiple needs.
        """
        
        # This line correctly joins all accumulated needs.
        needs_summary = ', '.join(context.extracted_needs)
        
        # --- THIS IS THE UPGRADED PROMPT ---
        prompt = f"""
        You are an expert services or needs search query creator for a vector database. Your task is to synthesize the user's full conversation into a concise, keyword-rich search query.

        **Full Conversational Context:**
        - Business Type: "{context.business_context}"
        - Stated Needs & Topics (a cumulative list): "{needs_summary}"
        - Latest User Message: "{last_user_query}"

        **CRITICAL INSTRUCTIONS (Follow these steps exactly):**
        1.  **Synthesize ALL Needs:** Your primary goal is to combine the **Business Type** with ALL of the **Stated Needs & Topics**. Do not ignore any of them.
        2.  **Use Latest Message for Detail Only:** Use the "Latest User Message" only if it adds a new, specific detail. If it's a generic word (like 'yes', 'correct', 'no'), you MUST IGNORE IT and rely only on the other context.
        3.  **Construct a Coherent Query:** Combine all the important keywords into a single, logical search phrase.

        **Example 1 (Multiple Needs):**
        - Business Type: "ecommerce"
        - Stated Needs: "website design, logo design, stripe integration"
        - Latest User Message: "yes that is correct"
        - Your Output: ecommerce website design with logo and stripe payment integration

        **Example 2 (Adding Detail):**
        - Business Type: "restaurant"
        - Stated Needs: "website design"
        - Latest User Message: "i also need a menu display feature"
        - Your Output: restaurant website design with menu display feature

        **Your Final Output (a simple string of keywords):**
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            smart_query = response.choices[0].message.content.strip().replace('"', '')
            if not smart_query: # Fallback if LLM returns empty string
                raise ValueError("LLM returned an empty search query.")
            logger.info(f"🧠 Generated Smart Search Query: '{smart_query}'")
            return smart_query
        except Exception as e:
            logger.error(f"Failed to generate smart search query: {e}")
            # A robust fallback combining all important context
            important_keywords = [
                kw for kw in (context.extracted_needs + last_user_query.split())
                if kw not in ['help', 'yes', 'no', 'correct', 'information']
            ]
            return ' '.join([context.business_context or ''] + important_keywords)
    # In the AdvancedQualificationEngine class:
# DELETE the old generate_smart_response function.
# ADD or REPLACE with these three definitive functions.

    # def generate_qualification_question(self, question: str, context: ConversationContext) -> str:
    #     """
    #     ALTERNATIVE: Generate qualification question and extract business context using text parsing
    #     """
    #     smart_query = self._generate_smart_search_query(context, question)
    #     retrieved_context = ""
    #     try:
    #         results, _, _ = self.vector_store.contextual_search(smart_query, context, k=2)
    #         if results:
    #             context_parts = [f"Snippet: '{doc.text}...'" for doc, _, _ in results]
    #             retrieved_context = "\n".join(context_parts)
    #     except Exception as e:
    #         logger.error(f"RAG search for qualification failed: {e}")

    #     # Get conversation history
    #     conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages[-4:]])

    #     # First, extract business context
    #     business_context = self._extract_business_with_llm(question, conversation_history, context)
        
    #     # Then generate the question
    #     prompt = f"""
    #     You are an expert consultant chatbot for 10Turtle. Generate the single most insightful question to understand the user's needs better.

    #     **Current Understanding:**
    #     - Business Type: {context.business_context or 'Unknown'}
    #     - Known Needs: {', '.join(context.extracted_needs) if context.extracted_needs else 'None identified'}
        
    #     **Retrieved Service Information:**
    #     {retrieved_context}
        
    #     **Recent Conversation:**
    #     {conversation_history}

    #     **Your Task:**
        
    #     If user greet you then you should start asking question with greeting and without mentioning "business" word use only "needs".
    #     Generate ONE specific, clarifying question that will help you understand:
    #     1. What specific solution they need (if unclear)
    #     2. Any missing requirements or constraints
    #     3. Technical preferences or business goals

    #     **Guidelines:**
    #     - If their need is vague, ask for specifics using service options from the retrieved information
    #     - If business type is unclear, ask about their industry/business
    #     - If you have good context, ask about preferences, timeline, or technical details
    #     - Keep it conversational and professional
    #     - Use HTML format
    #     - Maximum 2 lines

    #     Return only the HTML question.
    #     """
        
    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model_name,
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=0.4,
    #             max_tokens=150
    #         )
    #         return self._clean_llm_output(response.choices[0].message.content)
    #     except Exception as e:
    #         logger.error(f"Qualification question generation failed: {e}")
    #         return self._generate_fallback_qualification_question(context)


    def generate_qualification_question(self, question: str, context: ConversationContext) -> str:
        """
        ALTERNATIVE: Generate qualification question and extract business context using text parsing
        """
        smart_query = self._generate_smart_search_query(context, question)
        retrieved_context = ""
        try:
            results, _, _ = self.vector_store.contextual_search(smart_query, context, k=2)
            if results:
                context_parts = [f"Snippet: '{doc.text}...'" for doc, _, _ in results]
                retrieved_context = "\n".join(context_parts)
        except Exception as e:
            logger.error(f"RAG search for qualification failed: {e}")

        # Get conversation history
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages[-4:]])

        # Extract business context (this returns a boolean)
        extraction_success = self._extract_business_with_llm(question, conversation_history, context)
        
        # Then generate the question (remove the unused business_context variable)
        prompt = f"""
        You are an expert consultant chatbot for 10Turtle. Generate the single most insightful question to understand the user's needs better.

        **Current Understanding:**
        - Business Type: {context.business_context or 'Unknown'}
        - Known Needs: {', '.join(context.extracted_needs) if context.extracted_needs else 'None identified'}
        
        **Retrieved Service Information:**
        {retrieved_context}
        
        **Recent Conversation:**
        {conversation_history}

        **Your Task:**
        If user greet you then you should start asking question with greeting and without mentioning **business** word use only "needs".
        Generate ONE specific, clarifying question that will help you understand:
        1. What specific solution they need (if unclear)
        2. Any missing requirements or constraints
        3. Technical preferences or business goals

        **Guidelines:**
        - If their need is vague, ask for specifics using service options from the retrieved information
        - If business type is unclear, ask about their industry/business
        - If you have good context, ask about preferences, timeline, or technical details
        - Keep it conversational and professional
        - Use HTML format
        - Maximum 2 lines

        Return only the HTML question.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=150
            )
            return self._clean_llm_output(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Qualification question generation failed: {e}")
            return self._generate_fallback_qualification_question(context)


    # def _extract_business_with_llm(self, question: str, conversation_history: str, context: ConversationContext) -> bool:
    #     """
    #     Extract business context using LLM with text parsing instead of JSON
    #     """
    #     prompt = f"""
    #         Analyze this conversation and extract business information. Be very specific and accurate.

    #         **Conversation History:**
    #         {conversation_history}

    #         **Current User Message:** "{question}"

    #         **Current Understanding:**
    #         - Business: {context.business_context or 'Unknown'}
    #         - Needs: {', '.join(context.extracted_needs) if context.extracted_needs else 'None'}

    #         **IMPORTANT INSTRUCTION:**
    #         - If the user uses corrective language like "no only", "just", "instead", or contradicts previous needs, you should REPLACE the existing needs.
    #         - If the user simply says "no" without providing new information, extract NONE for needs (they are rejecting our understanding)
    #         - If the user provides new specific needs, determine if they are adding to or replacing previous needs

    #         **Your Task:**
    #         Extract and analyze the business information. Respond in this exact format:

    #         BUSINESS_TYPE: [specific business type in snake_case like book_selling, restaurant, healthcare, or NONE if unclear]
    #         BUSINESS_DESCRIPTION: [brief description of their business or NONE]
    #         EXTRACTED_NEEDS: [comma-separated list of needs/requirements or NONE]
    #         REPLACEMENT_MODE: [YES if user is correcting/replacing previous needs, NO if adding to them, CLEAR if user rejected without providing alternatives]
    #         CONFIDENCE: [number between 0.0 and 1.0]
    #         REASONING: [brief explanation of your analysis]

    #         **Examples:**
    #         User: "no only I want logo design"
    #         REPLACEMENT_MODE: YES

    #         User: "no" (after being asked for confirmation)
    #         EXTRACTED_NEEDS: NONE
    #         REPLACEMENT_MODE: CLEAR

    #         User: "yes content management"
    #         REPLACEMENT_MODE: YES

    #         Be specific and accurate in your analysis.
    #         """


        
    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model_name,
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=0.2,
    #             max_tokens=200
    #         )
            
    #         analysis_text = response.choices[0].message.content
    #         return self._parse_business_analysis(analysis_text, context)
            
    #     except Exception as e:
    #         logger.error(f"Business extraction failed: {e}")
    #         return False

    # def _parse_business_analysis(self, analysis_text: str, context: ConversationContext) -> bool:
    #     """
    #     Parse the LLM business analysis text and update context
    #     """
    #     try:
    #         lines = analysis_text.strip().split('\n')
    #         analysis = {}
            
    #         for line in lines:
    #             if ':' in line:
    #                 key, value = line.split(':', 1)
    #                 key = key.strip()
    #                 value = value.strip()
    #                 analysis[key] = value
            
    #         # Check if this is replacement mode
    #     #     replacement_mode = analysis.get('REPLACEMENT_MODE', 'NO').upper() == 'YES'
            
    #     #     # Extract business type
    #     #     business_type = analysis.get('BUSINESS_TYPE', '').strip()
    #     #     if business_type and business_type.lower() not in ['none', 'unknown', 'unclear']:
    #     #         if not context.business_context or context.business_context != business_type:
    #     #             context.business_context = business_type
    #     #             logger.info(f"🧠 Extracted business type: {business_type}")
            
    #     #     # Extract needs - HANDLE REPLACEMENT
    #     #     needs_text = analysis.get('EXTRACTED_NEEDS', '').strip()
    #     #     if needs_text and needs_text.lower() != 'none':
    #     #         new_needs = [need.strip() for need in needs_text.split(',')]
                
    #     #         if replacement_mode:
    #     #             # Replace existing needs completely
    #     #             context.extracted_needs = new_needs
    #     #             logger.info(f"🧠 REPLACED needs with: {new_needs}")
    #     #         else:
    #     #             # Add to existing needs (avoid duplicates)
    #     #             for need in new_needs:
    #     #                 if need and need.lower() not in [existing.lower() for existing in context.extracted_needs]:
    #     #                     context.extracted_needs.append(need)
    #     #                     logger.info(f"🧠 Added need: {need}")
            
    #     #     # Store business description
    #     #     business_desc = analysis.get('BUSINESS_DESCRIPTION', '').strip()
    #     #     if business_desc and business_desc.lower() != 'none':
    #     #         context.key_information["business_description"] = business_desc
            
    #     #     # Update confidence
    #     #     confidence_text = analysis.get('CONFIDENCE', '0.5')
    #     #     try:
    #     #         llm_confidence = float(confidence_text)
    #     #         context.confidence_level = max(context.confidence_level, llm_confidence * 0.9)
    #     #     except ValueError:
    #     #         pass
            
    #     #     # Log reasoning
    #     #     reasoning = analysis.get('REASONING', '')
    #     #     if reasoning:
    #     #         logger.info(f"🧠 LLM reasoning: {reasoning}")
            
    #     #     return True
            
    #     # except Exception as e:
    #     #     logger.error(f"Failed to parse business analysis: {e}")
    #     #     return False
    #         replacement_mode = analysis.get('REPLACEMENT_MODE', 'NO').upper()
        
    #     # Extract needs - HANDLE ALL MODES
    #         needs_text = analysis.get('EXTRACTED_NEEDS', '').strip()
    #         if needs_text and needs_text.lower() != 'none':
    #             new_needs = [need.strip() for need in needs_text.split(',')]
                
    #             if replacement_mode == 'YES':
    #                 # Replace existing needs completely
    #                 context.extracted_needs = new_needs
    #                 logger.info(f"🧠 REPLACED needs with: {new_needs}")
    #             elif replacement_mode == 'CLEAR':
    #                 # Clear needs - user rejected without providing alternatives
    #                 context.extracted_needs = []
    #                 logger.info(f"🧠 CLEARED needs due to user rejection")
    #             else:
    #                 # Add to existing needs (avoid duplicates)
    #                 for need in new_needs:
    #                     if need and need.lower() not in [existing.lower() for existing in context.extracted_needs]:
    #                         context.extracted_needs.append(need)
    #                         logger.info(f"🧠 Added need: {need}")
    #         elif replacement_mode == 'CLEAR':
    #             # User said no without providing alternatives
    #             context.extracted_needs = []
    #             logger.info(f"🧠 CLEARED needs due to user rejection")
            
    #         return True
    #     except Exception as e:
    #         logger.error(f"Failed to parse business analysis: {e}")
    #         return False    
    # 
    # 
    # Perplexity 



    def _extract_business_with_llm(self, question: str, conversation_history: str, context: ConversationContext) -> bool:
        """
        FIXED: Context-aware business extraction with proper need preservation
        """
        
        # Check if user said "no" to a clarification question (not confirmation)
        if question.lower().strip() == "no":
            if len(self.messages) >= 2:
                last_bot_message = self.messages[-2].get("content", "").lower()
                confirmation_indicators = [
                    "to confirm", "is that correct", "is that right", 
                    "does this sound right", "have i understood correctly"
                ]
                is_confirmation = any(indicator in last_bot_message for indicator in confirmation_indicators)
                
                if not is_confirmation:
                    # This is just a "no" to a clarification question - preserve everything
                    logger.info(f"🧠 User said 'no' to clarification, preserving context")
                    return True
        
        # Enhanced business extraction prompt
        prompt = f"""
        Analyze this conversation and extract business information. Be very specific and accurate.

        **Conversation History:**
        {conversation_history}

        **Current User Message:** "{question}"

        **Current Understanding:**
        - Business: {context.business_context or 'Unknown'}
        - Needs: {', '.join(context.extracted_needs) if context.extracted_needs else 'None'}

        **CRITICAL INSTRUCTIONS:**
        1. If business type is already known, DO NOT change it unless user explicitly corrects it
        2. If needs already exist, ADD to them, don't replace unless user explicitly corrects
        3. Look for industry keywords: ecommerce, restaurant, healthcare, automotive, etc.
        4. Extract specific services: chatbot, website, app, CRM, etc.

        **Response Format:**
        BUSINESS_TYPE: [specific industry like ecommerce, restaurant, healthcare, or CURRENT if no change]
        BUSINESS_DESCRIPTION: [brief description or CURRENT]
        EXTRACTED_NEEDS: [comma-separated NEW needs only, or NONE if no new needs]
        ACTION: [ADD or REPLACE or KEEP]
        CONFIDENCE: [0.0-1.0]
        REASONING: [brief explanation]

        **Examples:**
        User: "i am in ecommerce industry"
        BUSINESS_TYPE: ecommerce
        ACTION: ADD
        EXTRACTED_NEEDS: NONE
        
        User: "chatbot for customer service" 
        BUSINESS_TYPE: CURRENT
        ACTION: ADD
        EXTRACTED_NEEDS: chatbot for customer service
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            return self._parse_enhanced_business_analysis(response.choices[0].message.content, context)
            
        except Exception as e:
            logger.error(f"Business extraction failed: {e}")
            return False

    def _parse_enhanced_business_analysis(self, analysis_text: str, context: ConversationContext) -> bool:
        """Enhanced parsing with proper context preservation"""
        try:
            lines = analysis_text.strip().split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    analysis[key.strip()] = value.strip()
            
            # Handle business type
            business_type = analysis.get('BUSINESS_TYPE', '').strip()
            if business_type and business_type.upper() not in ['CURRENT', 'NONE', 'UNKNOWN']:
                context.business_context = business_type
                logger.info(f"🧠 Set business type: {business_type}")
            
            # Handle needs with proper action
            action = analysis.get('ACTION', 'ADD').upper()
            needs_text = analysis.get('EXTRACTED_NEEDS', '').strip()
            
            if needs_text and needs_text.upper() not in ['NONE', 'CURRENT']:
                new_needs = [need.strip() for need in needs_text.split(',')]
                
                if action == 'REPLACE':
                    context.extracted_needs = new_needs
                    logger.info(f"🧠 REPLACED needs: {new_needs}")
                elif action == 'ADD':
                    for need in new_needs:
                        if need and need.lower() not in [existing.lower() for existing in context.extracted_needs]:
                            context.extracted_needs.append(need)
                            logger.info(f"🧠 ADDED need: {need}")
                # KEEP means no changes
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to parse enhanced business analysis: {e}")
            return False





    def detect_topic_change_with_llm(self, question: str, context: ConversationContext) -> Dict[str, Any]:
        """
        ALTERNATIVE: Detect topic changes using text parsing instead of JSON

        """
        if question.lower().strip() == "no":
        # Check if last bot message was a confirmation
            if len(self.messages) >= 2:
                last_bot_message = self.messages[-2].get("content", "").lower()
                confirmation_indicators = [
                    "to confirm", "is that correct", "is that right", 
                    "does this sound right", "have i understood correctly"
                ]
                is_confirmation = any(indicator in last_bot_message for indicator in confirmation_indicators)
                
                if not is_confirmation:
                    # This is just a "no" to a clarification question - don't clear needs
                    logger.info(f"🧠 User said 'no' to clarification, keeping existing needs: {context.extracted_needs}")
                    return True
        
        history_summary = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in self.messages[-6:]])
        
        # First extract business context
        self._extract_business_with_llm(question, history_summary, context)
        
        # Then analyze topic change
        prompt = f"""
        Analyze if there's a topic change in this conversation.

        **Conversation History:**
        {history_summary}
        
        **New User Message:** "{question}"

        **Guidelines:**
        - A clarification (e.g., 'ecommerce website' -> 'shopify') is NOT a topic change
        - A completely different service (e.g., 'logo design' -> 'blockchain development') IS a topic change
        - Adding related features is usually NOT a topic change

        **Response Format:**
        TOPIC_CHANGED: [YES or NO]
        NEW_TOPIC: [specific new topic if changed, or NONE]
        REASONING: [brief one-sentence explanation]

        **Examples:**
        User was discussing website, now asks about mobile app: 
        TOPIC_CHANGED: NO
        NEW_TOPIC: NONE  
        REASONING: Mobile app is related to existing website discussion

        User was discussing website, now asks about blockchain:
        TOPIC_CHANGED: YES
        NEW_TOPIC: blockchain_development
        REASONING: Blockchain is completely different from website development

        Analyze the conversation and respond in the exact format above.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_topic_analysis(analysis_text)
            
        except Exception as e:
            logger.error(f"Topic detection failed: {e}")
            return {"topic_changed": False, "reasoning": "Analysis failed"}

    def _parse_topic_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """
        Parse topic change analysis from LLM response
        """
        try:
            lines = analysis_text.strip().split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    analysis[key] = value
            
            topic_changed = analysis.get('TOPIC_CHANGED', 'NO').upper() == 'YES'
            new_topic = analysis.get('NEW_TOPIC', '').strip()
            reasoning = analysis.get('REASONING', 'Topic analysis completed')
            
            if new_topic.upper() == 'NONE':
                new_topic = None
            
            logger.info(f"🧠 Topic Analysis: {reasoning}")
            
            return {
                "topic_changed": topic_changed,
                "new_topic": new_topic,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Failed to parse topic analysis: {e}")
            return {"topic_changed": False, "reasoning": "Parsing failed"}

    def generate_confirmation_question(self, context: ConversationContext) -> str:
        """
        ALTERNATIVE: Generate confirmation using text parsing
        """
        # First refine the business context
        conversation_context = " | ".join([msg['content'] for msg in self.messages[-4:] if msg['role'] == 'user'])
        self._refine_business_context(conversation_context, context)
        
        # Then generate confirmation
        prompt = f"""
        Generate a specific confirmation question that shows clear understanding of the user's needs.

        **Current Understanding:**
        - Business Type: {context.business_context or 'Unknown'}
        - Requirements: {', '.join(context.extracted_needs) if context.extracted_needs else 'None'}
        - Business Description: {context.key_information.get('business_description', 'N/A')}

        **Guidelines:**
        - Be specific about both business type and exact needs
        - Use HTML format
        - Make it sound natural and professional
        - Show you understand their specific situation
        - Keep it concise (1-2 lines max)

        **Example:**
        "So to confirm - you need an ecommerce website and mobile app for your book selling business, with inventory management capabilities. Is that right?"

        Generate only the HTML confirmation question.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            return self._clean_llm_output(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Confirmation generation failed: {e}")
            business_desc = context.business_context.replace('_', ' ').title() if context.business_context else "your business"
            needs_desc = ', '.join(context.extracted_needs) if context.extracted_needs else "your project"
            return f"<p>So you need {needs_desc} for {business_desc}. Is that correct?</p>"

    def _refine_business_context(self, conversation_context: str, context: ConversationContext):
        """
        Refine business context based on full conversation
        """
        if not conversation_context:
            return
        
        prompt = f"""
        Based on the full conversation, refine the business understanding:

        **Current Understanding:**
        - Business: {context.business_context or 'Unknown'}  
        - Needs: {', '.join(context.extracted_needs) if context.extracted_needs else 'None'}

        **Full User Messages:** {conversation_context}

        **Task:** Provide refined/corrected information in this format:
        REFINED_BUSINESS: [final business type or CURRENT if no change needed]
        REFINED_NEEDS: [comma-separated refined needs or CURRENT if no change needed]
        BUSINESS_DESCRIPTION: [brief description or CURRENT]

        Only suggest changes if you're confident they're more accurate.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            self._parse_refined_context(response.choices[0].message.content, context)
        except Exception as e:
            logger.error(f"Context refinement failed: {e}")

    def _parse_refined_context(self, refinement_text: str, context: ConversationContext):
        """
        Parse and apply context refinements
        """
        try:
            lines = refinement_text.strip().split('\n')
            refinements = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    refinements[key.strip()] = value.strip()
            
            # Apply refinements
            refined_business = refinements.get('REFINED_BUSINESS', '').strip()
            if refined_business and refined_business.upper() != 'CURRENT':
                context.business_context = refined_business
                logger.info(f"🧠 Refined business context: {refined_business}")
            
            refined_needs = refinements.get('REFINED_NEEDS', '').strip()
            if refined_needs and refined_needs.upper() != 'CURRENT':
                context.extracted_needs = [need.strip() for need in refined_needs.split(',')]
                logger.info(f"🧠 Refined needs: {context.extracted_needs}")
            
            refined_desc = refinements.get('BUSINESS_DESCRIPTION', '').strip()
            if refined_desc and refined_desc.upper() != 'CURRENT':
                context.key_information["business_description"] = refined_desc
                
        except Exception as e:
            logger.error(f"Failed to parse context refinements: {e}")

            
    def _update_context_from_llm_analysis(self, context: ConversationContext, analysis: Dict):
        """
        Helper method to update context from LLM business analysis
        """
        if not analysis:
            return
        
        # Update business type
        business_type = analysis.get("business_type")
        if business_type and business_type != "null" and business_type.lower() != "unknown":
            if not context.business_context or context.business_context != business_type:
                context.business_context = business_type
                logger.info(f"🧠 Updated business context: {business_type}")
        
        # Update needs (merge with existing, avoid duplicates)
        new_needs = analysis.get("extracted_needs", [])
        for need in new_needs:
            if need and need.lower() not in [existing.lower() for existing in context.extracted_needs]:
                context.extracted_needs.append(need)
                logger.info(f"🧠 Added need: {need}")
        
        # Store business description
        business_desc = analysis.get("business_description", "")
        if business_desc and business_desc.strip():
            context.key_information["business_description"] = business_desc
        
        # Update confidence if provided
        if analysis.get("confidence"):
            llm_confidence = float(analysis["confidence"])
            # Take the maximum of current confidence and LLM confidence
            context.confidence_level = max(context.confidence_level, llm_confidence * 0.9)

    # def _generate_smart_search_query(self, context: ConversationContext, last_user_query: str) -> str:
    #     """
    #     ENHANCED: Smarter search query generation using existing business context
    #     """
    #     query_parts = []
        
    #     # Add business context
    #     if context.business_context:
    #         business_clean = context.business_context.replace('_', ' ')
    #         query_parts.append(business_clean)
        
    #     # Add extracted needs
    #     if context.extracted_needs:
    #         query_parts.extend(context.extracted_needs[:3])  # Top 3 needs
        
    #     # Add meaningful parts from user query (skip generic responses)
    #     if last_user_query:
    #         query_lower = last_user_query.lower().strip()
    #         # Skip generic responses
    #         generic_responses = ['yes', 'no', 'correct', 'right', 'ok', 'okay', 'sure']
    #         if query_lower not in generic_responses and len(query_lower.split()) <= 4:
    #             query_parts.append(last_user_query)
        
    #     # Create final query
    #     final_query = ' '.join(query_parts)
        
    #     # Fallback if empty
    #     if not final_query.strip():
    #         final_query = "business services solutions"
        
    #     logger.info(f"🧠 Generated Smart Search Query: '{final_query}'")
    #     return final_query
    #=====================================================
    # perplexity
    # 
    def _generate_smart_search_query(self, context: ConversationContext, last_user_query: str) -> str:
        """
        FIXED: Robust search query generation with proper fallbacks
        """
        # Build query components
        query_parts = []
        
        # Add business context if available
        if context.business_context:
            query_parts.append(context.business_context.replace('_', ' '))
        
        # Add needs
        if context.extracted_needs:
            # Clean and add needs
            clean_needs = []
            for need in context.extracted_needs[:4]:  # Top 4 needs
                clean_need = need.replace('_', ' ').strip()
                if clean_need not in ['yes', 'no', 'correct', 'right']:
                    clean_needs.append(clean_need)
            query_parts.extend(clean_needs)
        
        # Add meaningful parts from user query
        if last_user_query:
            query_lower = last_user_query.lower().strip()
            # Skip generic responses
            generic_responses = ['yes', 'no', 'correct', 'right', 'ok', 'okay', 'sure']
            if query_lower not in generic_responses:
                # Extract meaningful words
                words = query_lower.split()
                meaningful_words = [w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'with']]
                if meaningful_words:
                    query_parts.extend(meaningful_words[:3])
        
        # Create final query
        final_query = ' '.join(query_parts).strip()
        
        # Robust fallbacks
        if not final_query:
            if context.business_context:
                final_query = f"{context.business_context} services"
            else:
                final_query = "business services solutions"
        
        logger.info(f"🧠 Generated Smart Search Query: '{final_query}'")
        return final_query
 

    def _update_context_from_intent(self, context: ConversationContext, intent_analysis: Dict, question: str) -> ConversationContext:
        """
        This function now only extracts entities and calculates confidence.
        Topic detection is handled by the new smart response generator.
        """
        # We no longer run the old topic detector here.

        # Update basic context fields
        context.primary_intent = intent_analysis.get('primary_intent', context.primary_intent)
        context.emotional_state = intent_analysis.get('emotional_tone', context.emotional_state)
        context.user_expertise_level = intent_analysis.get('user_expertise', context.user_expertise_level)

        # Extract business context and needs
        self._extract_business_context(question, context)
        topics = self._extract_topics_from_message(question)
        context.active_topics.extend(topics)
        if len(context.active_topics) > 5:                          #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            context.active_topics = context.active_topics[-5:]

        # Update confidence level
        context.confidence_level = self._calculate_confidence_level(context, intent_analysis)    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        return context

    def _handle_topic_transition(self, context: ConversationContext, topic_analysis: Dict, question: str):
        """Intelligently handle different types of topic transitions with proper context management."""
        confidence = topic_analysis["confidence"]
        breakdown = topic_analysis.get("breakdown", {})
        semantic_drift = breakdown.get("semantic", 0)
        contextual_divergence = breakdown.get("contextual", 0)

        logger.info(f"🔍 Topic transition detected: confidence={confidence:.2f}, semantic_drift={semantic_drift:.2f}, contextual_divergence={contextual_divergence:.2f}")

        # --- REVISED LOGIC FOR TRANSITION TYPE ---
        transition_type = "minor_adjustment"  # Default

        # Condition 1: High semantic drift AND moderate/high contextual divergence
        is_strong_semantic_shift = semantic_drift > 0.65 and contextual_divergence > 0.45

        # Condition 2: Moderate semantic drift BUT very high contextual divergence (THIS IS YOUR CASE)
        is_strong_contextual_shift = semantic_drift > 0.35 and contextual_divergence > 0.60 #70

        if is_strong_semantic_shift or is_strong_contextual_shift:
            transition_type = "complete_shift"
        elif semantic_drift > 0.4 and contextual_divergence > 0.3:
            transition_type = "related_tangent"

        # --- The rest of the function remains the same ---
        context.key_information["last_topic_transition"] = {
            "confidence": confidence,
            "type": transition_type,
            "semantic_drift": semantic_drift,
            "contextual_divergence": contextual_divergence,
            "question": question,
            "timestamp": datetime.now().isoformat()
        }

        if transition_type == "complete_shift":
            logger.info("🔄 Handling a COMPLETE TOPIC SHIFT. Resetting to discovery stage.")
            preserved_engagement = context.user_engagement_level
            preserved_emotional = context.emotional_state

            context.key_information["previous_topic"] = {
                "business_context": context.business_context,
                "extracted_needs": context.extracted_needs.copy(),
                "stage": context.conversation_stage
            }
            
            context.business_context = None
            context.extracted_needs = []
            context.active_topics = [self._extract_main_topic(question)]
            context.conversation_stage = "discovery"
            context.confidence_level = 0.2
            context.user_engagement_level = preserved_engagement
            context.emotional_state = preserved_emotional

        elif transition_type == "related_tangent":
            logger.info("🔄 Handling a RELATED TANGENT. Adjusting context.")
            context.confidence_level = max(0.3, context.confidence_level - 0.3)
            context.active_topics.append(self._extract_main_topic(question))
        
        else:
            logger.info("🔄 Handling a MINOR ADJUSTMENT. Maintaining context.")
            new_topic = self._extract_main_topic(question)
            if new_topic not in context.active_topics:
                context.active_topics.append(new_topic)
            context.confidence_level = max(0.4, context.confidence_level - 0.05)

    def _extract_main_topic(self, message: str) -> str:
        """Extract the main topic from a user message with fallbacks"""
        message_lower = message.lower()
        
        # Comprehensive business topics with priority weighting
        topic_keywords = {
        # Core Services
        'website_development': ['website', 'web', 'web development', 'web design', 'custom web application', 'frontend', 'backend', 'cms', 'php', 'laravel', 'wordpress'],
        'app_development': ['app', 'mobile app', 'application', 'ios', 'android', 'react native', 'flutter', 'cross-platform', 'hybrid app'],
        'ecommerce': ['ecommerce', 'e-commerce', 'online store', 'shop', 'retail platform', 'shopify', 'woocommerce', 'magento', 'marketplace', 'payment gateway'],
        'ui_ux_design': ['ui', 'ux', 'ui/ux', 'user interface', 'user experience', 'design', 'wireframing', 'prototyping', 'visual design', 'app design', 'website design'],
        'branding_and_logo': ['brand', 'branding', 'logo', 'identity', 'visual identity', 'corporate identity', 'brand strategy', 'logo design', 'stationery'],

        # AI Services
        'ai_development': ['ai', 'artificial intelligence', 'ai development', 'ai solutions', 'ai integration', 'ai-powered'],
        'chatbot': ['chatbot', 'conversational ai', 'virtual assistant', 'ai assistant', 'chatbot development', 'dialogflow', 'rasa', 'whatsapp bot', 'telegram bot'],
        'data_science': ['data science', 'ml', 'machine learning', 'analytics', 'predictive modeling', 'data analysis', 'business intelligence', 'data visualization'],
        'ai_art': ['ai art', 'generative art', 'ai artists', 'midjourney', 'stable diffusion', 'dall-e'],
        'ai_video': ['ai video', 'video avatars', 'music videos', 'video generation', 'synthetic video'],

        # Specialized Services
        'game_development': ['game', 'game development', 'mobile game', 'unity', 'unreal engine', '2d game', '3d game', 'game design'],
        'blockchain': ['blockchain', 'crypto', 'web3', 'decentralized', 'smart contract', 'dapp', 'nft', 'crypto coin', 'token development'],
        'digital_marketing': ['digital marketing', 'marketing', 'seo', 'search engine optimization', 'ppc', 'social media marketing', 'smm', 'content marketing', 'performance marketing'],
        'website_maintenance': ['maintenance', 'update', 'security', 'optimization', 'monitoring', 'website maintenance', 'support'],
        'saas_development': ['saas', 'software as a service', 'cloud solutions', 'saas platform'],
        'motion_design': ['motion graphics', 'animation', 'video animation', 'explainer video', 'lottie'],
        'canva_design': ['canva', 'canva design', 'canva templates', 'canva presentation'],

        # General & Business
        'consultation': ['consultation', 'advice', 'help', 'guidance', 'strategy', 'digital strategy', 'tech partner'],
        'business_strategy': ['business plan', 'growth strategy', 'scaling', 'business model'],
        'customer_experience': ['customer experience', 'user journey', 'cx'],
    }
        
        # Find the most specific match with priority
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return topic
        
        # Fallback: Use first meaningful noun phrase
        words = message_lower.split()
        if len(words) > 2:
            # Skip common starters
            skip_words = ['i', 'we', 'you', 'the', 'a', 'an', 'our', 'your']
            filtered_words = [w for w in words if w not in skip_words and len(w) > 3]
            if filtered_words:
                return ' '.join(filtered_words[:3])
        
        # Final fallback
        return "general_inquiry"

    def _extract_business_context(self, question: str, context: ConversationContext):
        """FIXED: Enhanced business context extraction"""

        message_lower = question.lower()

        # Comprehensive business type indicators
        business_types = {
            'seo': ['seo', 'search engine', 'ranking', 'optimization', 'google'],
            'blockchain': ['blockchain', 'crypto', 'cryptocurrency', 'web3', 'smart contract', 'defi', 'nft'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'chatbot'],
            'mobile': ['mobile', 'app', 'ios', 'android', 'application'],
            'ecommerce': ['ecommerce', 'online store', 'shopping', 'commerce', 'retail', 'marketplace'],
            'website': ['website', 'web development', 'web design', 'site'],
            'marketing': ['marketing', 'digital marketing', 'advertising', 'promotion'],
            'game_development': ['game', 'gaming', 'mobile game'],
            
            # Industries
            'restaurant': ['restaurant', 'cafe', 'food', 'dining', 'food delivery'],
            'construction': ['construction', 'building', 'contractor'],
            'healthcare': ['healthcare', 'medical', 'clinic', 'doctor', 'telemedicine'],
            'automotive': ['automotive', 'car', 'vehicle', 'auto', 'dealership'],
            'education': ['education', 'edtech', 'e-learning', 'school'],
            'finance': ['finance', 'fintech', 'banking', 'insurance', 'lending'],
            'real_estate': ['real estate', 'property', 'realty'],
            'travel': ['travel', 'tourism', 'hospitality', 'booking'],
            'logistics': ['logistics', 'shipping', 'supply chain'],
        }

        # Find the most specific match
        for business_type, keywords in business_types.items():
            if any(keyword in message_lower for keyword in keywords):
                context.business_context = business_type
                break

        # FIXED: Extract needs with proper string handling
        need_indicators = ['need', 'want', 'looking for', 'require', 'help with', 'interested in', 'want to']
        for indicator in need_indicators:
            if indicator in message_lower:
                parts = message_lower.split(indicator, 1)
                if len(parts) > 1:
                    # FIXED: Use parts[1] which is the string portion after the split
                    need_text = parts[1].strip()  # This is now a string, not a list
                    potential_need = need_text.split()[0:3]  # First 3 words
                    need = ' '.join(potential_need)
                    if need and need not in context.extracted_needs:
                        context.extracted_needs.append(need)
                break

    def _determine_conversation_stage(self, context: ConversationContext, intent_analysis: Dict) -> str:
        """Determine conversation stage based on context and intent"""

        current_stage = context.conversation_stage
        primary_intent = intent_analysis.get('primary_intent', 'question')

        # Stage progression logic with topic change awareness
        if current_stage == "discovery":
            if context.business_context and context.extracted_needs:
                return "qualification"
        elif current_stage == "qualification":
            if context.confidence_level > 0.7 or context.questions_asked > 3:
                return "recommendation"
        elif current_stage == "recommendation":
            if primary_intent == "request" or "ready" in context.conversation_history_summary.lower():
                return "closing"

        return current_stage

    def _extract_topics_from_message(self, message: str) -> List[str]:
        """Extract topics from user message"""

        topics = []
        message_lower = message.lower()

        # Comprehensive business topics
        topic_keywords = {
            # Core Services
            'website_development': ['website', 'web', 'web development', 'web design', 'custom web application', 'frontend', 'backend', 'cms', 'php', 'laravel', 'wordpress'],
            'app_development': ['app', 'mobile app', 'application', 'ios', 'android', 'react native', 'flutter', 'cross-platform', 'hybrid app'],
            'ecommerce': ['ecommerce', 'e-commerce', 'online store', 'shop', 'retail platform', 'shopify', 'woocommerce', 'magento', 'marketplace', 'payment gateway'],
            'ui_ux_design': ['ui', 'ux', 'ui/ux', 'user interface', 'user experience', 'design', 'wireframing', 'prototyping', 'visual design', 'app design', 'website design'],
            'branding_and_logo': ['brand', 'branding', 'logo', 'identity', 'visual identity', 'corporate identity', 'brand strategy', 'logo design', 'stationery'],

            # AI Services
            'ai_development': ['ai', 'artificial intelligence', 'ai development', 'ai solutions', 'ai integration', 'ai-powered'],
            'chatbot': ['chatbot', 'conversational ai', 'virtual assistant', 'ai assistant', 'chatbot development', 'dialogflow', 'rasa', 'whatsapp bot', 'telegram bot'],
            'data_science': ['data science', 'ml', 'machine learning', 'analytics', 'predictive modeling', 'data analysis', 'business intelligence', 'data visualization'],
            'ai_art': ['ai art', 'generative art', 'ai artists', 'midjourney', 'stable diffusion', 'dall-e'],
            'ai_video': ['ai video', 'video avatars', 'music videos', 'video generation', 'synthetic video'],

            # Specialized Services
            'game_development': ['game', 'game development', 'mobile game', 'unity', 'unreal engine', '2d game', '3d game', 'game design'],
            'blockchain': ['blockchain', 'crypto', 'web3', 'decentralized', 'smart contract', 'dapp', 'nft', 'crypto coin', 'token development'],
            'digital_marketing': ['digital marketing', 'marketing', 'seo', 'search engine optimization', 'ppc', 'social media marketing', 'smm', 'content marketing', 'performance marketing'],
            'website_maintenance': ['maintenance', 'update', 'security', 'optimization', 'monitoring', 'website maintenance', 'support'],
            'saas_development': ['saas', 'software as a service', 'cloud solutions', 'saas platform'],
            'motion_design': ['motion graphics', 'animation', 'video animation', 'explainer video', 'lottie'],
            'canva_design': ['canva', 'canva design', 'canva templates', 'canva presentation'],

            # General & Business
            'consultation': ['consultation', 'advice', 'help', 'guidance', 'strategy', 'digital strategy', 'tech partner'],
            'business_strategy': ['business plan', 'growth strategy', 'scaling', 'business model'],
            'customer_experience': ['customer experience', 'user journey', 'cx'],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _calculate_confidence_level(self, context: ConversationContext, intent_analysis: Dict) -> float:
        """More realistic confidence calculation with awareness of topic transitions."""

        print("USing _calculate_confidence_level")
        last_transition = context.key_information.get("last_topic_transition", {})
        transition_type = last_transition.get("type")

        # If we just had a major topic shift, confidence is inherently low.
        if transition_type == "complete_shift":
            # Start low and build up again as new context is gathered.
            base_confidence = 0.15
            if context.business_context: base_confidence += 0.2
            if context.extracted_needs: base_confidence += min(0.3, len(context.extracted_needs) * 0.15)
            return min(0.9, base_confidence)

        # Confidence calculation for continuing conversations
        confidence = 0.15
        if context.business_context:
            confidence += 0.3
        if context.extracted_needs:
            confidence += min(0.45, len(context.extracted_needs) * 0.15)
        if context.questions_asked > 2:
            confidence += min(0.15, (context.questions_asked - 2) * 0.05)
        
        engagement_boost = {'high': 0.1, 'medium': 0.05, 'low': 0.0}
        confidence += engagement_boost.get(context.user_engagement_level, 0.0)
        
        return min(1.0, confidence)

    def _determine_response_parameters(self, context: ConversationContext, intent_analysis: Dict) -> Dict:
        """Determine response style and parameters"""

        # Base parameters
        params = {
            'style': 'balanced',
            'length': 'medium',
            'temperature': 0.1,
            'max_tokens': 200
        }

        # Adapt based on emotional state
        emotional_tone = intent_analysis.get('emotional_tone', 'professional')
        if emotional_tone in ['excited', 'curious']:
            params['temperature'] = 0.4
        elif emotional_tone in ['frustrated', 'uncertain']:
            params['style'] = 'supportive'
            params['temperature'] = 0.2

        return params

    def _build_adaptive_system_prompt(self, context: ConversationContext, intent_analysis: Dict, memory_context: str) -> str:
        """Build adaptive system prompt with nuanced topic transition awareness"""
        base_prompt = "You are an expert business consultant chatbot for 10Turtle. Your goal is to identify a user's business challenges and demonstrate how 10Turtle's services can solve them."
         
        # Check for topic transition
        last_transition = context.key_information.get("last_topic_transition", {})
        requires_qualification = last_transition.get("requires_qualification", False)
        transition_type = last_transition.get("type", "minor_adjustment")
        previous_topic = context.key_information.get("previous_topic", {})
        
        if requires_qualification:
            if transition_type == "complete_shift":
                base_prompt += " The user has completely shifted topics. Focus on understanding their NEW requirements through specific questions rather than providing immediate recommendations. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
                if previous_topic and (previous_topic.get("business_context") or previous_topic.get("extracted_needs")):
                    base_prompt += " Briefly acknowledge the topic change without dwelling on previous discussion. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
            elif transition_type == "related_tangent":
                base_prompt += " The user is exploring a related tangent to the current topic. Connect this new direction to our previous discussion while gathering more details. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
            else:  # subtopic_expansion or minor_adjustment
                base_prompt += " The user is expanding on the current topic. Gather more specific details about this aspect while maintaining conversation flow. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        else:
            base_prompt += " Continue the conversation based on the established context. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        
        # # Add context-specific instructions
        # if context.conversation_stage == "discovery":
        #     base_prompt += " Focus on understanding the user's business and needs through natural, specific questions. Ask one question at a time for clarity."
        # elif context.conversation_stage == "qualification":
        #     if requires_qualification:
        #         base_prompt += " Ask specific, targeted questions about their new topic to build understanding. Don't rush to recommendations."
        #     else:
        #         base_prompt += " Ask clarifying questions to better understand their specific requirements and constraints."
        # elif context.conversation_stage == "recommendation":
        #     base_prompt += " Provide helpful recommendations and solutions based on their well-understood needs. Be specific and actionable."
        # Add context-specific instructions


        # Add emotional adaptation
        emotional_tone = intent_analysis.get('emotional_tone', 'professional')
        if emotional_tone == 'frustrated':
            base_prompt += " Be extra patient and understanding. Acknowledge any potential confusion. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        elif emotional_tone == 'excited':
            base_prompt += " Match their enthusiasm while staying professional and focused. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        elif emotional_tone == 'confused':
            base_prompt += " Provide clear, simple explanations. Break down complex concepts. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        
        if context.conversation_stage == "discovery" or context.conversation_stage == "qualification":
            base_prompt += " Your primary goal is to understand the user's needs. Do not provide solutions or recommendations yet. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
            
            # --- NEW INSTRUCTION FOR AMBIGUITY ---
            base_prompt += "**Crucial Rule for Ambiguity: If the user's request is broad (e.g., 'I need marketing' or 'help with design'), you MUST resolve this ambiguity by offering 2-4 specific options to choose from. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
            base_prompt += "Do not ask a generic open-ended question like 'How can I help?'. "
            base_prompt += "Example for 'marketing': 'We can definitely help. To narrow it down, are you focused on improving your Google ranking (SEO), engaging on social media, or running paid ad campaigns?'** and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."

            base_prompt += "**Crucial Rule for 'How-To' Questions: If a user asks 'how' to do something (e.g., 'how do I automate excel?'), DO NOT give a technical answer. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
            base_prompt += "Instead, acknowledge their goal and ask a clarifying question to understand the business context. Pivot from 'how' to 'why'.**  and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
            
            base_prompt += "Your response MUST end with a single, clear, and relevant question. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        
        elif context.conversation_stage == "recommendation":
            base_prompt += " You have sufficient information. Provide helpful recommendations and solutions based on their well-understood needs. Be specific and actionable. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."


        # --- NEW: Add persona adaptation based on user expertise ---
        if context.user_expertise_level == 'non_technical':
            base_prompt += " Your communication style should be simple, clear, and business-focused. Avoid technical jargon. Use analogies to explain complex topics. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        elif context.user_expertise_level == 'technical':
            base_prompt += " You are speaking with a technical user. You can be direct and use industry-standard technical terms and acronyms where appropriate. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."
        else: # intermediate
            base_prompt += " Your communication style should be balanced, professional, and clear. and MUST PROVIDE RESPONSE IN **HTML** format and in a 1-2 Lines **MAX**."




        return base_prompt

    def _clean_llm_output(self, response: str) -> str:
        """
        A robust function to clean LLM responses, removing markdown code blocks,
        preambles, and other unwanted artifacts, returning only the core HTML.
        """
        # 1. Use regex to strip the markdown code block wrapper if it exists.
        # This pattern handles optional language identifiers (like 'html') and surrounding whitespace.
        code_block_pattern = r'^\s*```(?:[a-zA-Z]*)?\s*(.*?)\s*```\s*$'
        match = re.search(code_block_pattern, response, re.DOTALL)

        if match:
            # If a markdown block is found, use its captured content as the response.
            cleaned_response = match.group(1)
        else:
            # Otherwise, use the original response (it might not have been wrapped).
            cleaned_response = response

        # 2. Remove common AI conversational preambles.
        patterns_to_remove = [
            r"^(Here's the HTML you requested|Certainly, here is the HTML|Okay, here's the response):?\s*",
            r"^(Okay,|Alright,|Sure,|I understand).*?\.\s*",
        ]
        for pattern in patterns_to_remove:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE | re.MULTILINE).strip()

        # 3. Final whitespace cleanup
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()

        return cleaned_response

    def _clean_and_enhance_response(self, response: str, context: ConversationContext, question: str) -> str:
        cleaned_response = self._clean_llm_output(response)

        return cleaned_response

    def _basic_clean_response(self, response: str) -> str:
            """Basic response cleaning without topic awareness"""
            # Remove AI thinking patterns
            thinking_patterns = [
                r"^(Okay,|Alright,|Sure,|I understand|I'm ready to assist).*?\.\s*",
                r"^.*?as a.*?chatbot.*?\.\s*",
                r"html\s*?",
            ]
            for pattern in thinking_patterns:
                response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.MULTILINE)
            
            # Remove markdown code blocks properly
            response = response.replace('```json', '').replace('```','')
            response = response.replace('```html', '').replace('```', '')
            
            # Clean up whitespace
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Remove literal escape characters
            response = response.replace('\\\\', ' ').replace('\\', '')

            # 1b) Or, if you really just want to change **every** single backslash to a space:
            response = response.replace('\\', ' ')            
            return response
    def _generate_fallback_response(self, context: ConversationContext, intent_analysis: Dict) -> str:
            """Generate fallback response when AI generation fails"""

            if context.conversation_stage == "discovery":
                return "I'd love to learn more about your business! What industry are you in? 🏢"
            elif context.conversation_stage == "qualification":
                return "That's helpful! Can you tell me more about your specific goals? 🎯"
            elif context.conversation_stage == "recommendation":
                return "Based on what you've shared, I think we can definitely help! Let me suggest some solutions. 🚀"
            else:
                return "I'm here to help with your business needs! What can I assist you with today? 💼"

# ============================================================================
# ENHANCED VECTOR STORE WITH ROBUST ERROR HANDLING
# ============================================================================

class IntelligentVectorStore:
    """Enhanced vector store with service URLs, work links, and robust error handling"""


    def __init__(self, cache_dir: str = "./vector_cache", embedding_dimension: int = 1536):  # CHANGED: Updated to 1536 for OpenAI
        self.cache_dir = cache_dir
        self.dimension = embedding_dimension
        self.qualification_engine = None

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize FAISS
        self.index = faiss.IndexFlatIP(embedding_dimension)
        faiss.omp_set_num_threads(4)

        self.documents: List[Document] = []
        self.search_cache = {}
        self.contextual_cache = {}

        # Enhanced caching
        self.embeddings_file = os.path.join(cache_dir, "embeddings.npy")
        self.documents_file = os.path.join(cache_dir, "documents.pkl")
        self.faiss_index_file = os.path.join(cache_dir, "faiss.index")
        self.smart_cache = os.path.join(cache_dir, "smart_cache.pkl")

        # Service URL mapping
        self.service_keywords_map = {
            'website': ['web', 'site', 'online', 'digital', 'wordpress', 'laravel', 'php'],
            'development': ['dev', 'app', 'software', 'coding', 'react', 'flutter', 'xamarin'],
            'ecommerce': ['shop', 'store', 'retail', 'commerce', 'shopify', 'woocommerce'],
            'design': ['graphic', 'visual', 'ui', 'ux', 'wireframe', 'prototype', 'logo'],
            'canva': ['canva', 'sketch', 'draw', 'design', 'presentation'],
            'game': ['game', 'gaming', 'unity', 'unreal'],
            'ios': ['ios', 'iphone', 'app'],
            'android': ['android', 'google', 'app'],
            'branding': ['logo', 'identity', 'brand', 'stationery'],
            'marketing': ['marketing', 'advertising', 'promotion', 'smm', 'ppc'],
            'seo': ['seo', 'search', 'optimization', 'ranking'],
            'blockchain': ['blockchain', 'crypto', 'web3', 'dapp', 'nft'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'chatbot', 'generative', 'data'],
            'maintenance': ['maintenance', 'support', 'update', 'security']
        }

        # Initialize work links manager
        self.work_links_manager = IntelligentWorkLinksManager()

        self._load_smart_cache()

    def contextual_search(self, query: str, context: ConversationContext, k: int = 6) -> Tuple[List, Set[str], List[Dict]]:
        """FIXED: Contextual search with stage-aware caching and improved relevance"""

        try:
            # FIXED: Build cache key that includes conversation stage for proper invalidation
            # enhanced_query = self._build_contextual_search_query(query, context)
            enhanced_query = query
            cache_key = f"{enhanced_query}_{context.conversation_stage}"


            # FIXED: Clear cache when entering recommendation stage to ensure fresh results
            if context.conversation_stage == "recommendation" and hasattr(self, '_last_stage'):
                if self._last_stage != "recommendation":
                    logger.info("🔄 Clearing cache for recommendation stage")
                    self.contextual_cache.clear()
            
            self._last_stage = context.conversation_stage

            # Check cache only if not in recommendation stage or cache is fresh
            if cache_key in self.contextual_cache and context.conversation_stage != "recommendation":
                logger.info("📦 Using cached search results")
                return self.contextual_cache[cache_key]

            # Initialize embedder - NOW USING OPENAI
            if not hasattr(self, 'embedder'):
                self.embedder = OpenAIEmbeddingModel('text-embedding-3-small')

            query_embedding = self.embedder.encode([enhanced_query], normalize_embeddings=True)
            search_k = min(k * 4, self.index.ntotal)

            if search_k == 0:
                work_links = self.work_links_manager.find_relevant_work_links(context, query)
                return [], set(), work_links

            # FAISS search
            similarities, indices = self.index.search(query_embedding.astype(np.float32), search_k)

            filtered_results = []
            collected_urls = set()

            # FIXED: Process search results with improved error handling
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                try:
                    # Skip invalid indices
                    if idx < 0 or idx >= len(self.documents):
                        continue

                    doc = self.documents[int(idx)]

                    # FIXED: More lenient contextual relevance for recommendation stage
                    contextual_score = self._calculate_contextual_relevance(
                        doc, context, float(sim), enhanced_query
                    )

                    # FIXED: Lower threshold for recommendation stage
                    threshold = 0.2 if context.conversation_stage == "recommendation" else 0.3
                    
                    if contextual_score > threshold:
                        filtered_results.append((doc, contextual_score, 'contextual'))

                        # FIXED: Enhanced service URL collection with lower threshold
                        for url in doc.source_urls:
                            try:
                                url_relevance_score = self._calculate_service_url_relevance(url, context, query)
                                # FIXED: Much lower threshold for service URLs in recommendation stage
                                url_threshold = 0.15 if context.conversation_stage == "recommendation" else 0.2
                                
                                if url_relevance_score > url_threshold:
                                    collected_urls.add(url)
                                    logger.info(f"✅ Added service URL: {url} (score: {url_relevance_score:.3f})")
                            
                            except Exception as e:
                                logger.error(f"Error processing URL {url}: {e}")
                                continue
                                
                except Exception as e:
                    logger.error(f"Error processing search result for index {idx}: {e}")
                    continue

            # Find relevant work links
            work_links = self.work_links_manager.find_relevant_work_links(context, query)

            # Sort by relevance
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            final_results = filtered_results[:k]
            # lowered_list = [item.lower() for item in my_list]
            result = (final_results, collected_urls, work_links)
            # print(f'CONtextualsearch{result}')
            
            # FIXED: Only cache if not in recommendation stage to ensure fresh results
            if context.conversation_stage != "recommendation":
                self.contextual_cache[cache_key] = result

            logger.info(f"🔍 Search completed: {len(final_results)} docs, {len(collected_urls)} service URLs, {len(work_links)} work links")
            return result

        except Exception as overall_error:
            logger.error(f"Contextual search failed: {overall_error}")
            work_links = self.work_links_manager.find_relevant_work_links(context, query)
            return [], set(), work_links

    def _calculate_service_url_relevance(self, url: str, context: ConversationContext, query: str) -> float:
        """FIXED: Much more generous service URL relevance calculation"""

        url_lower = url.lower()
        relevance_score = 0.0

        # FIXED: Higher base score for any service-related URL
        service_indicators = ['service', 'solution', 'offering', 'expertise', 'specialization']
        if any(word in url_lower for word in service_indicators):
            relevance_score += 0.4  # Increased from 0.4

        # FIXED: More generous business context matching
        if context.business_context:
            business_lower = context.business_context.lower()

            # Direct business type in URL
            if business_lower in url_lower:
                relevance_score += 0.6  # Strong match

            # FIXED: Enhanced keyword matching with broader coverage
            if business_lower in self.service_keywords_map:
                keywords = self.service_keywords_map[business_lower]
                for keyword in keywords:
                    if keyword in url_lower:
                        relevance_score += 0.4  # Increased from 0.3
                        break
            
            # FIXED: Fallback matching for common business terms
            business_terms = ['business', 'company', 'corporate', 'professional', 'digital']
            if any(term in url_lower for term in business_terms):
                relevance_score += 0.3

        # FIXED: More generous needs matching
        if context.extracted_needs:
            for i, need in enumerate(context.extracted_needs[:3]):
                need_lower = need.lower()
                weight = 0.5 - (i * 0.1)  # Decreased penalty for later needs

                # Direct need match
                if need_lower in url_lower:
                    relevance_score += weight

                # FIXED: Word-level matching for compound needs
                need_words = need_lower.split()
                matching_words = sum(1 for word in need_words if len(word) > 2 and word in url_lower)
                if matching_words > 0:
                    relevance_score += weight * 0.4 * (matching_words / len(need_words))

        # FIXED: Query term matching with more weight
        if query:
            query_words = [w for w in query.lower().split() if len(w) > 2]
            matching_query_words = sum(1 for word in query_words if word in url_lower)
            if matching_query_words > 0:
                relevance_score += 0.3 * (matching_query_words / len(query_words))

        # FIXED: Bonus for recommendation stage
        if context.conversation_stage == "recommendation":
            relevance_score *= 1.2  # 20% boost

        return min(relevance_score, 1.0)

    def _build_contextual_search_query(self, query: str, context: ConversationContext) -> str:
        """Build enhanced search query considering conversation context"""

        query_parts = [query]

        if context.business_context:
            query_parts.append(context.business_context)

        if context.extracted_needs:
            query_parts.extend(context.extracted_needs[:2])

        if context.active_topics:
            query_parts.extend(context.active_topics[-2:])

        return ' '.join(query_parts)

    def _calculate_contextual_relevance(self, doc: Document, context: ConversationContext,
                                      semantic_score: float, enhanced_query: str) -> float:
        """FIXED: More generous contextual relevance for recommendation stage"""

        # FIXED: Higher base semantic weight
        relevance = semantic_score * 0.6  # Increased from 0.5

        text_lower = doc.text.lower()
        title_lower = doc.metadata.get('title', '').lower()

        # FIXED: More generous business context matching
        if context.business_context:
            business_words = context.business_context.lower().split()
            business_matches = 0
            
            for word in business_words:
                if len(word) > 2:  # Skip short words
                    if word in text_lower:
                        business_matches += 1
                        relevance += 0.2  # Reduced individual boost but cumulative
                    if word in title_lower:
                        business_matches += 1
                        relevance += 0.15
            
            # FIXED: Bonus for multiple business word matches
            if business_matches > 1:
                relevance += 0.1

        # FIXED: Enhanced needs matching
        if context.extracted_needs:
            needs_matches = 0
            for need in context.extracted_needs:
                need_lower = need.lower()
                
                # Direct need match
                if need_lower in text_lower:
                    needs_matches += 1
                    relevance += 0.25  # Reduced from 0.3
                if need_lower in title_lower:
                    needs_matches += 1
                    relevance += 0.2
                
                # FIXED: Word-level matching for compound needs
                need_words = [w for w in need_lower.split() if len(w) > 2]
                word_matches = sum(1 for word in need_words if word in text_lower)
                if word_matches > 0:
                    relevance += 0.15 * (word_matches / len(need_words))
            
            # FIXED: Bonus for multiple need matches
            if needs_matches > 1:
                relevance += 0.1

        # FIXED: Active topics matching
        if context.active_topics:
            topic_matches = 0
            for topic in context.active_topics:
                topic_lower = topic.lower()
                if topic_lower in text_lower or topic_lower in title_lower:
                    topic_matches += 1
                    relevance += 0.1
            
            # Bonus for multiple topic matches
            if topic_matches > 1:
                relevance += 0.05

        # FIXED: Stage-based bonus
        if context.conversation_stage == "recommendation":
            relevance *= 1.15  # 15% boost for recommendation stage

        return min(relevance, 1.0)
    def _load_smart_cache(self):
        """Load enhanced cache"""
        try:
            if os.path.exists(self.smart_cache):
                with open(self.smart_cache, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.search_cache = cache_data.get('search_cache', {})
                    self.contextual_cache = cache_data.get('contextual_cache', {})
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            self.search_cache = {}
            self.contextual_cache = {}


    def load_cache(self) -> bool:
        """Load cache"""
        try:
            files_exist = all(os.path.exists(f) for f in [
                self.embeddings_file, self.documents_file, self.faiss_index_file
            ])

            if not files_exist:
                return False

            with open(self.documents_file, 'rb') as f:
                self.documents = pickle.load(f)

            self.index = faiss.read_index(self.faiss_index_file)

            logger.info(f"✅ Cache loaded: {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"❌ Cache load failed: {e}")
            return False

    def add_documents(self, documents: List[Document], embeddings: np.ndarray, save_cache: bool = True):
        """Add documents with proper error handling"""
        try:
            if len(embeddings) == 0:
                return

            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            self.documents.extend(documents)
            self.index.add(embeddings.astype(np.float32))

            if save_cache:
                with open(self.documents_file, 'wb') as f:
                    pickle.dump(self.documents, f)
                faiss.write_index(self.index, self.faiss_index_file)
                np.save(self.embeddings_file, embeddings)

            logger.info(f"✅ Added {len(documents)} documents to index")

        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")

# ============================================================================
# MAIN CHATBOT CLASS WITH ALL FEATURES INTEGRATED
# ============================================================================

class AdvancedRAGChatbot:
    """Production-grade RAG chatbot with all advanced features"""

    def __init__(self, cache_dir: str = "./smart_cache"):
        logger.info("🚀 Initializing Advanced RAG Chatbot...")

        try:
            # Initialize components - NOW USING OPENAI FOR EMBEDDINGS
            self.embedding_model = OpenAIEmbeddingModel('text-embedding-3-small')  # CHANGED: OpenAI embeddings
            self.vector_store = IntelligentVectorStore(cache_dir, 1536)  # CHANGED: Updated dimension to 1536
            self.qualification_engine = AdvancedQualificationEngine()
            self.qualification_engine.vector_store = self.vector_store
            self.vector_store.qualification_engine = self.qualification_engine
            

            # Link deduplication
            self.link_deduplicator = LinkDeduplicationManager()

            # State management
            self.messages: List = []
            self.context = ConversationContext()
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Store messages reference in qualification engine for topic detection
            self.qualification_engine.messages = self.messages

            logger.info("✅ Advanced RAG Chatbot initialized!")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise


    def chat(self, question: str) -> Dict[str, Any]:
        """
        ENHANCED: Main chat method with integrated business extraction
        """
        start_time = time.time()
        logger.info(f"⚡ Processing: {question[:50]}...")

        try:
            # --- Initialization for the turn ---
            self.link_deduplicator.reset_session()
            self.messages.append({"role": "user", "content": question})
            self.context.questions_asked += 1
            response_data = {}

            # --- Step 1: Topic Change Detection (with business extraction) ---
            if len(self.messages) > 1:
                topic_analysis = self.qualification_engine.detect_topic_change_with_llm(question, self.context)
                if topic_analysis.get("topic_changed"):
                    logger.info(f"🧠 Topic Agent detected a shift to '{topic_analysis.get('new_topic')}'. Resetting context.")
                    # Reset but preserve some context
                    preserved_engagement = self.context.user_engagement_level
                    preserved_questions_asked = self.context.questions_asked
                    # Keep any business info that was just extracted
                    preserved_business = self.context.business_context
                    preserved_needs = self.context.extracted_needs.copy()
                    
                    self.context = ConversationContext(
                        user_engagement_level=preserved_engagement,
                        questions_asked=preserved_questions_asked,
                        business_context=preserved_business,
                        extracted_needs=[]      # Clear extracted needs completely
  # Keep some context
                    )
                    self.context.conversation_stage = "discovery"

            # --- Step 2: State Machine Logic ---

            # STATE 1: PENDING CONFIRMATION (Highest Priority)
            if self.context.pending_confirmation:
                last_bot_question = self.messages[-2].get("content", "")
                if self.qualification_engine._is_positive_confirmation(question, last_bot_question):
                    # User confirmed - proceed to recommendation
                    self.context.pending_confirmation = False
                    self.context.conversation_stage = "recommendation"
                    logger.info("✅ User confirmed. Transitioning to recommendation stage.")
                    response_data = self._generate_intelligent_recommendation(question)
                else:
                    # User rejected - CLEAR NEEDS and ask what they actually want
                    logger.info("❌ User rejected confirmation. Clearing needs and asking clarifying question.")
                    self.context.pending_confirmation = False
                    self.context.confidence_level = 0.3
                    
                    # CLEAR EXTRACTED NEEDS when user rejects confirmation
                    self.context.extracted_needs = []
                    
                    response = self.qualification_engine.generate_qualification_question(question, self.context)
                    response_data = self._format_qualification_response(question, response)

            # STATE 2: RECOMMENDATION
            elif self.context.conversation_stage == "recommendation":
                response_data = self._generate_intelligent_recommendation(question)

            # STATE 3: QUALIFICATION (Default state)
            else:
                self.context.conversation_stage = "qualification"
                
                # ENHANCED: Use improved confidence calculation
                self.context.confidence_level = self._calculate_enhanced_confidence_level(self.context)
                
                # Check if ready for confirmation (more conservative approach)
                has_solid_context = (
                    self.context.business_context and 
                    len(self.context.extracted_needs) >= 1 and
                    self.context.confidence_level > 0.65
                )
                
                is_extensive_conversation = self.context.questions_asked >= 3
                not_recently_rejected = not self._just_rejected_confirmation()
                
                should_confirm = (has_solid_context or is_extensive_conversation) and not_recently_rejected

                if should_confirm:
                    logger.info("🎯 Qualification complete. Generating confirmation summary.")
                    self.context.pending_confirmation = True
                    # Enhanced confirmation (which will also refine business context)
                    response = self.qualification_engine.generate_confirmation_question(self.context)
                    response_data = self._format_qualification_response(question, response)
                else:
                    # Continue qualifying (this will extract business context automatically)
                    response = self.qualification_engine.generate_qualification_question(question, self.context)
                    response_data = self._format_qualification_response(question, response)

            # --- Finalize the turn ---
            self.messages.append({"role": "assistant", "content": response_data["answer"]})
            self._update_conversation_summary()
            
            # Log extracted context for debugging
            logger.info(f"📊 Context Update - Business: {self.context.business_context}, Needs: {self.context.extracted_needs}, Confidence: {self.context.confidence_level:.2f}")
            
            process_time = time.time() - start_time
            logger.info(f"✅ Response generated in {process_time:.2f}s")
            return response_data

        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            return self._format_error_response(question)

    def _calculate_enhanced_confidence_level(self, context: ConversationContext) -> float:
        """
        ENHANCED: Better confidence calculation using LLM-extracted context
        """
        confidence = 0.15  # Base confidence
        
        # Business context confidence (higher weight for specific business types)
        if context.business_context:
            confidence += 0.3
            
            # Bonus for specific business types (not just "website" or "mobile")
            specific_business_types = [
                'book_selling', 'restaurant', 'healthcare', 'ecommerce', 'real_estate',
                'automotive', 'education', 'finance', 'travel', 'logistics'
            ]
            if context.business_context in specific_business_types:
                confidence += 0.1
        
        # Needs confidence - quality over quantity
        if context.extracted_needs:
            # Base needs score
            needs_score = min(0.35, len(context.extracted_needs) * 0.12)
            
            # Bonus for specific, detailed needs
            specificity_bonus = 0
            for need in context.extracted_needs:
                words = need.split()
                if len(words) >= 2:  # Multi-word needs are more specific
                    specificity_bonus += 0.08
                elif words[0] in ['website', 'app', 'system', 'platform', 'solution']:
                    specificity_bonus += 0.05
            
            confidence += needs_score + min(specificity_bonus, 0.2)
        
        # Conversation engagement and depth
        if context.questions_asked > 2:
            confidence += min(0.15, (context.questions_asked - 2) * 0.04)
        
        # Business description bonus (from LLM extraction)
        if context.key_information.get("business_description"):
            confidence += 0.08
        
        return min(1.0, confidence)

    def _just_rejected_confirmation(self) -> bool:
        """
        ENHANCED: Better detection of recent confirmation rejection
        """
        if len(self.messages) < 4:
            return False
        
        # Look for recent confirmation questions and negative responses
        recent_exchanges = []
        for i in range(len(self.messages) - 6, len(self.messages), 2):
            if i >= 0 and i + 1 < len(self.messages):
                bot_msg = self.messages[i].get("content", "").lower()
                user_msg = self.messages[i + 1].get("content", "").lower()
                recent_exchanges.append((bot_msg, user_msg))
        
        # Check for confirmation patterns followed by negative responses
        confirmation_patterns = ["is that correct", "is that right", "does this sound right", "to confirm"]
        negative_responses = ["no", "not really", "not quite", "that's not right", "incorrect"]
        
        for bot_msg, user_msg in recent_exchanges[-2:]:  # Check last 2 exchanges
            has_confirmation = any(pattern in bot_msg for pattern in confirmation_patterns)
            has_negative = any(response in user_msg for response in negative_responses)
            
            if has_confirmation and has_negative:
                return True
        
        return False
    def _generate_clarifying_question_after_rejection(self, user_response: str) -> str:
        """Generate a clarifying question when user rejects our understanding"""
        
        prompt = f"""
        The user has rejected our understanding of their needs. Generate a clarifying question to better understand what they actually want.

        Current understanding:
        - Business: {self.context.business_context or 'Unknown'}
        - Needs: {', '.join(self.context.extracted_needs) if self.context.extracted_needs else 'None identified'}
        
        User's rejection: "{user_response}"
        
        Generate a single, specific question to clarify what they actually need. 
        Be direct and ask for specific clarification about their business or requirements.
        
        Return only the HTML response.
        """
        
        try:
            response = self.qualification_engine.client.chat.completions.create(
                model=self.qualification_engine.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            return self.qualification_engine._clean_llm_output(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Clarifying question generation failed: {e}")
            return "<p>I apologize for the confusion. Could you please tell me more specifically what you're looking for?</p>"

    

    # You will also need to add these two small helper functions to the AdvancedRAGChatbot class:
    def _format_qualification_response(self, question: str, response: str) -> Dict[str, Any]:
        """Formats a standard qualification response dictionary."""
        return {
            "question": question,
            "answer": response,
            "conversation_stage": self.context.conversation_stage,
            "confidence_level": round(self.context.confidence_level, 2),
            "chat_history": self.messages[-6:],
            "services": [], "related_works": [], "source_documents": [], "suggested_questions": [],
            "debug_info": { "pending_confirmation": self.context.pending_confirmation }
        }

    def _format_error_response(self, question: str) -> Dict[str, Any]:
        """Formats a standard error response dictionary."""
        return {
            "question": question,
            "answer": "<p>I apologize, but I encountered an error. Please try again.</p>",
            "conversation_stage": self.context.conversation_stage,
            "confidence_level": 0.0,
            "chat_history": self.messages,
            "services": [], "related_works": [], "source_documents": [], "suggested_questions": []
        }

    def build_index(self, site_tree_or_path, force_rebuild: bool = False):
        """Build search index"""

        try:
            if not force_rebuild and self.vector_store.load_cache():
                logger.info("⚡ Using cached index!")
                return

            logger.info("📄 Building search index...")

            # Load data
            if isinstance(site_tree_or_path, str):
                with open(site_tree_or_path, 'r', encoding='utf-8') as f:
                    site_tree = json.load(f)
            else:
                site_tree = site_tree_or_path

            # Process documents
            documents = self._process_site_tree(site_tree)

            if not documents:
                logger.warning("⚠️ No documents found to index")
                return

            # Generate embeddings - NOW USING OPENAI
            texts = [doc.text for doc in documents if doc.text.strip()]

            if not texts:
                logger.warning("⚠️ No valid text found in documents")
                return

            logger.info(f"🔄 Processing {len(texts)} documents...")
            embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)

            # Add to vector store
            self.vector_store.add_documents(documents, embeddings)

            logger.info(f"✅ Index built with {len(documents)} documents")

        except Exception as e:
            logger.error(f"❌ Index building failed: {e}")
            logger.info("🔄 Continuing with empty index...")

    def _process_site_tree(self, node, parent_url=None) -> List[Document]:
        """Process site tree into Document objects"""

        documents = []
        stack = [(node, parent_url)]

        while stack:
            current, parent = stack.pop()

            if not current:
                continue

            url = current.get('url', '')
            text = current.get('text', '').strip()
            title = current.get('title', '')

            if text:
                doc = Document(
                    text=text,
                    metadata={
                        'url': url,
                        'title': title,
                        'parent_url': parent
                    },
                    source_urls={url} if url else set()
                )

                documents.append(doc)

            # Add children
            for child in current.get('children', []):
                stack.append((child, url or parent))

        return documents

    def _handle_qualification(self, question: str) -> Dict[str, Any]:
        """
        Handles the qualification stage with a deterministic state machine,
        using the LLM for reasoning and language generation.
        """

        # === STATE 1: PENDING CONFIRMATION ===
        # This block has the highest priority. If we are waiting for a 'yes' or 'no', handle it first.
        if self.context.pending_confirmation:
            # Get the last bot message, which was the confirmation question
            last_bot_question = self.messages[-1].get("content", "")

            # Use the intelligent confirmation checker
            if self.qualification_engine._is_positive_confirmation(question, last_bot_question):
                self.context.pending_confirmation = False
                self.context.conversation_stage = "recommendation"
                logger.info("✅ User confirmed understanding. Transitioning to recommendation stage.")
                # Immediately proceed to generate the final recommendation and exit.
                return self._generate_intelligent_recommendation(question)
            else:
                    
                # --- THIS IS THE FIX ---
                # If the user corrects the bot, we MUST reset the state fully.
                self.context.pending_confirmation = False
                self.context.confidence_level = 0.4 # Keep some confidence
                logger.info("User provided a correction. Forcing re-qualification.")
                
                # We will now call the Brain with the user's correction to get a NEW question.
                self.context, response, _ = self.qualification_engine.generate_smart_response(
                    question, self.context
                )

        # === STATE 2: STANDARD QUALIFICATION ===
        # If we are not waiting for confirmation, run the normal qualification process.
        
        # 1. Generate the response and get the "is_ready" decision from the "Brain"
        self.context, response, is_ready_for_confirmation = self.qualification_engine.generate_smart_response(
            question, self.context
        )

        # 2. If the "Brain" decided it's time to confirm, this function will now generate
        #    the confirmation question for the user's current turn.
        if is_ready_for_confirmation:
            logger.info("🤖 Brain has determined qualification is complete. Generating confirmation summary.")
            self.context.pending_confirmation = True # Set the flag for the NEXT turn.
 
            pass # The response is already the confirmation question.
        else:
            logger.info(f"❓ Staying in qualification - Confidence: {self.context.confidence_level:.2f}")

        # 3. Format and return the final response dictionary
        return {
            "question": question,
            "answer": response,
            "conversation_stage": self.context.conversation_stage,
            "confidence_level": round(self.context.confidence_level, 2),
            "chat_history": self.messages[-6:],
            "services": [],
            "related_works": [],
            "source_documents": [],
            "suggested_questions": [],
            "debug_info": { "pending_confirmation": self.context.pending_confirmation }
        }

    def _handle_recommendation(self, question: str) -> Dict[str, Any]:
        """Handle recommendation stage"""

        # # Update context
        # self.context, _ = self.qualification_engine.generate_contextual_response(
        #     question, self.context
        # )

        return self._generate_intelligent_recommendation(question)

    def _generate_intelligent_recommendation(self, user_query: str = None) -> Dict[str, Any]:
        """Generate intelligent recommendations with unique links and suggested questions"""

        search_query = self.qualification_engine._generate_smart_search_query(self.context, user_query)
        
        results, service_urls, work_links = self.vector_store.contextual_search(
            search_query, self.context, k=4
        )
        
        recommendation_answer, suggested_questions = self._generate_contextual_recommendation(results, service_urls, work_links, user_query)


        # --- MODIFIED CALL TO HANDLE TUPLE RETURN ---
        # This function now returns both the answer and the suggested questions
        # recommendation_answer, suggested_questions  = self._generate_contextual_recommendation(results, service_urls, work_links, user_query)
        # print(f"_generate_intelligent_recommendation  {results}")
        services = self._extract_unique_services(results)
        related_works = self._extract_unique_work_links(work_links)
        source_docs = self._extract_source_documents(results)

        return {
            "question": user_query or "recommendation request",
            "answer": recommendation_answer,
            
            # --- ADD THE NEW KEY TO THE FINAL DICTIONARY ---
            "suggested_questions ": suggested_questions ,
            
            # "source_documents": source_docs,    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "related_works": related_works,
            "source_documents": services,
            "conversation_stage": self.context.conversation_stage,
            "confidence_level": round(self.context.confidence_level, 2),
            "chat_history": self.messages[-6:]
        }

    def _extract_unique_services(self, results: List) -> List[Dict[str, str]]:
        """
        Extracts unique, relevant service links by leveraging RAG scores
        and filtering for URL specificity.
        """
        scored_services = {}

        for doc, relevance, method in results:
            if relevance < 0.4:
                continue

            for url in doc.source_urls:
                # Basic check to ensure it's a service page
                if '/service/' not in url:
                    continue

                # --- NEW: SPECIFICITY FILTER (from your old code) ---
                try:
                    parsed = urlparse(url)
                    path_parts = parsed.path.strip("/").split("/")
                    
                    # We filter for URLs that are "deep" in the service hierarchy.
                    # A path like /service/uiux/website-design has 3 parts.
                    # This prevents high-level, generic pages from being shown.
                    if len(path_parts) < 3:
                        logger.info(f"🚮 Filtering out high-level service URL: {url}")
                        continue
                except Exception:
                    # If URL parsing fails for any reason, safely skip it
                    continue
                # --- END OF NEW FILTER ---

                # If the URL is specific enough, add it to our list
                if url in scored_services:
                    scored_services[url] = max(scored_services[url], relevance)
                else:
                    scored_services[url] = relevance

        # Sort the collected service URLs by their inherited relevance score
        sorted_urls = sorted(scored_services.items(), key=lambda item: item[1], reverse=True)

        # Build the final list of unique services
        unique_services = []
        for url, score in sorted_urls:
            if self.link_deduplicator.add_url(url):
                service_info = self._generate_service_info(url, score)
                unique_services.append(service_info)
                if len(unique_services) >= 4:
                    break

        return unique_services

    def _extract_unique_work_links(self, work_links: List[Dict]) -> List[Dict[str, str]]:
        """Extract unique work links without duplicates"""

        unique_works = []

        for work in work_links:
            work_url = work['url']

            if self.link_deduplicator.add_url(work_url):
                unique_works.append({
                    "title": work['title'],
                    "url": work_url,
                    # "description": work['description'],
                    # "technologies": ', '.join(work['technologies']) if work['technologies'] else '',
                    # "industry": work['industry'],
                    # "relevance": work['relevance_score']
                })

                if len(unique_works) >= 4:
                    break

        return unique_works

    def _generate_service_info(self, url: str, relevance_score: float) -> Dict[str, str]:
        """FIXED: Generate contextual service information from URL"""

        url_lower = url
        url_parts = url.split('/')

        # Extract meaningful name from URL
        service_name = "Our Services"

        # Try to get service name from URL path
        for part in reversed(url_parts):
            if part and len(part) > 2 and part not in ['service', 'services', 'www', 'http:', 'https:']:
                clean_name = part.replace('-', ' ').replace('_', ' ')
                service_name = clean_name.title()
                break

        # Context-aware service naming
        if self.context.business_context:
            business = self.context.business_context

            if business == 'restaurant' and any(word in url_lower for word in ['web', 'digital']):
                service_name = "Restaurant Digital Solutions"
            elif business == 'healthcare' and 'web' in url_lower:
                service_name = "Healthcare Web Development"
            elif business == 'blockchain' and any(word in url_lower for word in ['blockchain', 'crypto']):
                service_name = "Blockchain Development Services"
            elif business == 'seo' and any(word in url_lower for word in ['seo', 'search']):
                service_name = "SEO & Search Optimization"

        # FIXED: Need-specific naming
        if self.context.extracted_needs:
            primary_need = self.context.extracted_needs if self.context.extracted_needs else ""

            if 'marketing' in primary_need and 'digital' in url_lower:
                service_name = "Digital Marketing Solutions"
            elif 'website' in primary_need and 'design' in url_lower:
                service_name = "Website Design Services"
            elif 'blockchain' in primary_need:
                service_name = "Blockchain Development Services"
            elif 'seo' in primary_need:
                service_name = "SEO & Search Optimization"

        return {
            "title": service_name,
            "url": url,
            "relevance": round(relevance_score, 2)
        }

    def _build_recommendation_query(self, user_query: str = None) -> str:
        """Build recommendation search query"""

        components = []

        if user_query:
            components.append(user_query)

        if self.context.business_context:
            components.append(self.context.business_context)

        if self.context.extracted_needs:
            components.extend(self.context.extracted_needs[:3])

        return " ".join(components) or "business services solutions"


    def _generate_contextual_recommendation(self, results: List, service_urls: Set[str], work_links: List[Dict], user_query: str = None) -> tuple[str, list[str]]:
        """
        Generates a recommendation and suggested questions via two separate, robust LLM calls.
        Returns a tuple: (html_answer, list_of_questions)
        """
        # --- Part 1: Generate the Main Answer ---
        
        # Build context from search results, work links, and memory
        context_parts = [f"[Relevance: {relevance:.2f}] {doc.text}" for doc, relevance, _ in results[:3]]
        search_context = "\n---\n".join(context_parts)
        work_context = ""
        if work_links:
            work_examples = [f"{work['title']} ({work['industry']})" for work in work_links[:2]]
            work_context = f"\nRelevant work examples: {' | '.join(work_examples)}"
        memory_context = self.qualification_engine.memory.get_relevant_context(user_query or "recommendation", self.context)

        answer_prompt = f"""
    You are a professional business consultant for 10Turtle.
    Based on the context below, generate a helpful, conversational HTML response in a 1-2 Lines **MAX**.
    Address the user's needs and connect them to 10Turtle's solutions. Keep it concise.

    User Context:
    - Business: {self.context.business_context or 'General business'}
    - Needs: {', '.join(self.context.extracted_needs) if self.context.extracted_needs else 'Business growth'}

    Memory Context: {memory_context}
    Search Results: {search_context}
    {work_context}
    User Query: {user_query or 'Looking for recommendations'}

    Respond with ONLY the raw HTML content in a 1-2 Lines **MAX**.
    """
        try:
            answer_response = self.qualification_engine.client.chat.completions.create(
                model=self.qualification_engine.model_name,
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            main_answer = self.qualification_engine._clean_llm_output(answer_response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Failed to generate main answer: {e}")
            main_answer = self._generate_fallback_recommendation()

        # --- Part 2: Generate the Suggested Questions ---
         #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=================================================== adede context
        questions_prompt = f"""
        Based on the user's needs and the bot's answer, generate a JSON object.
        The object must have a single key called "questions" which contains an array of 2-3 concise string questions based on Context:{context_parts[:200]}. 
        These questions should help the user explore how 10Turtle's services can solve their specific problems.
        Phrase them from the user's perspective.

        User's Need: "I have an '{self.context.business_context}' business and I want to '{', '.join(self.context.extracted_needs)}'."
        Bot's Answer: "{main_answer}"

        Example Format:
        {{
        "questions": ["How long does a website redesign typically take?", "What kind of analytics can I get for my sales?", "Can you integrate my existing payment gateway?"]
        }}

        Return ONLY the raw JSON object and nothing else.
        """
        
        suggested_questions = [] # Default to empty list
        try:
            questions_response = self.qualification_engine.client.chat.completions.create(
                model=self.qualification_engine.model_name,
                messages=[{"role": "user", "content": questions_prompt}],
                temperature=0.5,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            response_content = questions_response.choices[0].message.content
            
            if response_content:
                response_data = json.loads(response_content)
                # The parsing is now simpler because we expect a consistent object format
                if isinstance(response_data, dict):
                    suggested_questions = response_data.get("questions", [])

            # Final type check to ensure it's always a list
            if not isinstance(suggested_questions, list):
                suggested_questions = []
                
        except Exception as e:
            logger.error(f"Failed to generate suggested questions: {e}")
            suggested_questions = []

        return main_answer, suggested_questions

    def _clean_response(self, response: str) -> str:
        """Clean response for production use"""

        patterns = [
            r"^(Okay,|Alright,|Sure,|I understand).*?\.\s*",
            r"^.*?consultant.*?\.\s*",
            r"html\s*",
        ]

        for pattern in patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.MULTILINE)

        # Remove markdown code blocks
        response = response.replace('``````', '')
        response = response.replace('``````', '')

        # Clean whitespace
        response = re.sub(r'\s+', ' ', response).strip()

        # Remove escape characters
        response = response.replace('\\n', ' ').replace('\\', '')

        return response

    def _generate_fallback_recommendation(self) -> str:
        """Generate fallback recommendation"""

        emoji = "🚀"
        business = self.context.business_context or "business"
        needs = ', '.join(self.context.extracted_needs[:2]) if self.context.extracted_needs else "growth"

        return f"{emoji} I'd love to help your {business} with {needs}! Let's discuss how we can create the perfect solution for you."

    def _extract_source_documents(self, results: List) -> List[Dict[str, Any]]:
        """Extract source document information"""
        docs = []
        for doc, relevance, method in results[:4]:
            docs.append({
                # "title": doc.metadata.get('title', 'Unknown'),
                "source": doc.metadata.get('url', ''),
                # "relevance_score": round(relevance, 3),
                # "preview": doc.text[:150] + "..." if len(doc.text) > 150 else doc.text
            })

        return docs

    # In the AdvancedRAGChatbot class, replace the entire function with this version:

    # def _extract_source_documents(self, results: List) -> List[Dict[str, str]]:
    #     """
    #     Extracts unique, relevant service links by leveraging RAG scores
    #     and filtering for URL specificity.
    #     """
    #     scored_services = {}

    #     for doc, relevance, method in results:
    #         if relevance < 0.4:
    #             continue

    #         for url in doc.metadata.get('url', ''):
    #             # Basic check to ensure it's a service page
    #             if '/service/' not in url:
    #                 continue

    #             # --- NEW: SPECIFICITY FILTER (from your old code) ---
    #             try:
    #                 parsed = urlparse(url)
    #                 path_parts = parsed.path.strip("/").split("/")
                    
    #                 # We filter for URLs that are "deep" in the service hierarchy.
    #                 # A path like /service/uiux/website-design has 3 parts.
    #                 # This prevents high-level, generic pages from being shown.
    #                 if len(path_parts) < 3:
    #                     logger.info(f"🚮 Filtering out high-level service URL: {url}")
    #                     continue
    #             except Exception:
    #                 # If URL parsing fails for any reason, safely skip it
    #                 continue
    #             # --- END OF NEW FILTER ---

    #             # If the URL is specific enough, add it to our list
    #             if url in scored_services:
    #                 scored_services[url] = max(scored_services[url], relevance)
    #             else:
    #                 scored_services[url] = relevance

    #     # Sort the collected service URLs by their inherited relevance score
    #     sorted_urls = sorted(scored_services.items(), key=lambda item: item[1], reverse=True)

    #     # Build the final list of unique services
    #     unique_services = []
    #     for url, score in sorted_urls:
    #         if self.link_deduplicator.add_url(url):
    #             service_info = self._generate_service_info(url, score)
    #             unique_services.append(service_info)
    #             if len(unique_services) >= 4:
    #                 break

    #     return unique_services

    def _update_conversation_summary(self):
        """Update conversation history summary"""

        recent_messages = self.messages[-6:]
        summary_parts = []

        for msg in recent_messages:
            role = msg['role']
            content = msg['content'][:150]
            summary_parts.append(f"{role}: {content}")

        self.context.conversation_history_summary = " | ".join(summary_parts)

    def reset(self):
        """Reset chatbot state"""
        self.messages.clear()
        self.context = ConversationContext()
        self.link_deduplicator.reset_session()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("🔄 Chatbot reset complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            'session_id': self.session_id,
            'conversation_stage': self.context.conversation_stage,
            'confidence_level': round(self.context.confidence_level, 2),
            'business_context': self.context.business_context,
            'extracted_needs': self.context.extracted_needs,
            'questions_asked': self.context.questions_asked,
            'user_engagement': self.context.user_engagement_level,
            'emotional_state': self.context.emotional_state
        }

    def add_work_links(self, work_links: List[str]):
        """Add additional work links to the database"""
        self.vector_store.work_links_manager.work_links_database.extend(work_links)
        logger.info(f"✅ Added {len(work_links)} work links to database")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG Chatbot API",
    description="Production-grade RAG chatbot with intelligent conversation management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    try:
        chatbot = AdvancedRAGChatbot(cache_dir="./production_cache")

        # Add work links if you have them
        work_links = [
            "https://10turtle.com/work/e-commerce-tech-software-ux-ui-design-website-development-beawhale-a-whale-of-innovation",

    "https://10turtle.com/work/automotive-e-commerce-ux-ui-design-website-development-dees-organics-e-commerce-excellence-platform-for-the-lymphatic-brush",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-print-design-mentes-expertas-motiva-transforma-impacta",

    "https://10turtle.com/work/food-hospitality-graphic-branding-el-fogn-menu-a-global-culinary-experience",

    "https://10turtle.com/work/ai-technology-graphic-branding-marketing-branding-print-design-tech-software-ec04-sustainable-energy-solutions",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-marketing-branding-tech-software-proxcath-medical-precision-in-medical-tubing-balloons-engineered-for-excellence",

    "https://10turtle.com/work/ai-technology-ux-ui-design-website-development-pines-edge-a-serene-and-immersive-retreat-experience-blending-nature-with-luxury",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-print-design-pete-evans-unlocking-ancestral-wisdom-for-modern-health",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-pleasant-paws-pet-center-where-exceptional-care-meets-pet-happiness",

    "https://10turtle.com/work/ai-technology-automotive-health-wellness-tech-software-ux-ui-design-website-development-houston-mind-brain-helping-you-on-the-path-to-wellness",

    "https://10turtle.com/work/ai-technology-automotive-travel-tourism-ux-ui-design-website-development-escaped-the-9-to-5-discover-your-next-adventure-with-ease",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-rose-wallis-atelier-timeless-elegance-in-bespoke-marble-travertine-furniture",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-marketing-branding-print-design-uxbridge-vets-caring-for-your-pets-always",

    "https://10turtle.com/work/automotive-finance-legal-graphic-branding-marketing-branding-mex-insurance-your-best-insurance-partners",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-la-2e-classe-stickers-design",

    "https://10turtle.com/work/ai-technology-mobile-app-tech-software-ux-ui-design-ask-gpt-revolutionizing-conversations-with-ai-powered-assistance",

    "https://10turtle.com/work/graphic-branding-jemma-russo-a-fashion-magazine",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-pure-organic-vitamins-enhancing-supplement-absorption",

    "https://10turtle.com/work/ai-technology-health-wellness-ux-ui-design-website-development-promille-get-ahead-with-promille-goals-growth-and-mastery",

    "https://10turtle.com/work/ai-technology-food-hospitality-graphic-branding-the-grind-burgers-chopped-cheese-fries-etc",

    "https://10turtle.com/work/ai-technology-graphic-branding-print-design-ajanta-fire-work-colours-of-celebration",

    "https://10turtle.com/work/automotive-ux-ui-design-website-development-classic-garage-drive-your-dreams-with-classic-cars",

    "https://10turtle.com/work/education-training-ux-ui-design-website-development-stem-discovery-boxes-transforming-the-stem-discovery-boxes-homepage",

    "https://10turtle.com/work/finance-legal-tech-software-ux-ui-design-website-development-crypto-trading-empowering-traders-with-a-modern-user-centric-platform",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-prerra-elegance-in-every-bead",

    "https://10turtle.com/work/sports-gaming-ux-ui-design-website-development-playglobalgame-turn-your-passion-into-winning",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-zabrina-cox-guiding-you-through-every-contraction",

    "https://10turtle.com/work/automotive-food-hospitality-graphic-branding-marketing-branding-packaging-design-steamysips-anytime-anywhere",

    "https://10turtle.com/work/ai-technology-mobile-app-ux-ui-design-house-ai-revolutionize-your-space-ai-powered-interior-design",

    "https://10turtle.com/work/ai-technology-education-training-studymate-ai-homework-helper",

    "https://10turtle.com/work/ai-technology-education-training-tech-software-xora-ai-ai-powered-video-creation",

    "https://10turtle.com/work/ai-technology-smart-cleanup-optimize-your-ios-storage",

    "https://10turtle.com/work/education-training-food-hospitality-tech-software-hungry-tables-smart-easy-dining-reservations",

    "https://10turtle.com/work/ux-ui-design-website-development-fotofreude-fun-and-seamless-photo-booth-experience-for-every-event",

    "https://10turtle.com/work/ai-technology-finance-legal-graphic-branding-imalloch-mcclean-more-simpler-smarter-solutions",

    "https://10turtle.com/work/food-hospitality-graphic-branding-print-design-tech-software-preshifter-with-better-service",

    "https://10turtle.com/work/automotive-finance-legal-graphic-branding-marketing-branding-prosperity-finance-group-made-finance-easier",

    "https://10turtle.com/work/ai-technology-finance-legal-graphic-branding-patria-lending-nature-in-flight-framed-by-color",

    "https://10turtle.com/work/ai-technology-marketing-branding-mobile-app-ux-ui-design-dall-redefining-creativity-with-ai",

    "https://10turtle.com/work/print-design-tech-software-casio-where-every-drive-becomes-an-experience",

    "https://10turtle.com/work/food-hospitality-graphic-branding-ichi-ichi-sweeter-he-sweet",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-cinci-420-get-premium-cannabis-flower-delivered-right-to-your-door",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-lmd-beauty-crafting-a-seamless-beauty-shopping-experience",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-tillian-simmons-resume",

    "https://10turtle.com/work/ai-technology-e-commerce-shopify-tech-software-ux-ui-design-website-development-iflytek-building-the-fly-tek-shopify-store",

    "https://10turtle.com/work/ux-ui-design-website-development-mobile-as-a-service-a-redesign-journey",

    "https://10turtle.com/work/ai-technology-automotive-education-training-graphic-branding-health-wellness-print-design-wirebewegen-best-learning-platform-ever",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-marketing-branding-yoga-mind-li-balance-your-mind-body-soul",

    "https://10turtle.com/work/ux-ui-design-website-development-gap-life-redefining-growth-and-purpose-beyond-the-conventional",

    "https://10turtle.com/work/ai-technology-education-training-smartchat-ai-multi-model-intelligence",

    "https://10turtle.com/work/ai-technology-tech-software-animart-ai-ai-anime-generator",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-guide-de-prparation-your-ultimate-prep-companion-for-success-in-english",

    "https://10turtle.com/work/automotive-graphic-branding-splash-in-style-caps-that-talk-colours-that-pop",

    "https://10turtle.com/work/ai-technology-graphic-branding-dreamworks-magazine-inspiring-faith-family-future",

    "https://10turtle.com/work/e-commerce-education-training-ux-ui-design-website-development-cryptaminiatures-llc-crafting-a-grim-dark-digital-experience",

    "https://10turtle.com/work/ai-technology-ux-ui-design-website-development-golden-touch-design-elevating-event-experiences",

    "https://10turtle.com/work/food-hospitality-ux-ui-design-website-development-tomtom-burritos-bringing-authentic-mexican-street-food-online",

    "https://10turtle.com/work/ai-technology-education-training-food-hospitality-fresh-food-connecting-surplus-to-need",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-building-my-leckerly-paradiesly-a-custom-e-commerce-experience",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-la-casa-de-las-capsulas-brew-perfection-with-every-coffee-capsule",

    "https://10turtle.com/work/ai-technology-graphic-branding-marketing-branding-print-design-jewelry-sales-academy-shaping-the-future-of-luxury-jewelry",

    "https://10turtle.com/work/graphic-branding-tech-software-le-collectifs-turn-stories-into-impact",

    "https://10turtle.com/work/graphic-branding-rare-remarkable-naturally-exceptional",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-transform-your-space-with-custom-portraits-by-exhibitiv",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-marketing-branding-schaufelle-optometry",

    "https://10turtle.com/work/ai-technology-graphic-branding-real-estate-construction-issac-advisor-your-next-home-solution",

    "https://10turtle.com/work/ai-technology-education-training-ai-interior-inspiration-for-personalized-spaces",

    "https://10turtle.com/work/ai-technology-automotive-education-training-graphic-branding-print-design-study-in-ireland-discover-multiculturalism-top-education-careers",

    "https://10turtle.com/work/graphic-branding-bold-patterns-everyday-essentials-reimagined",

    "https://10turtle.com/work/food-hospitality-tech-software-ux-ui-design-website-development-cuppa-sip-smarter-choose-local",

    "https://10turtle.com/work/ai-technology-e-commerce-education-training-ux-ui-design-website-development-steezyink-inking-memories-elevating-experiences",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-revitalizing-mad-kannas-digital-presence",

    "https://10turtle.com/work/ai-technology-education-training-ux-ui-design-website-development-narke-hydraulik-a-website-redesign-for-enhanced-service-accessibility",

    "https://10turtle.com/work/automotive-health-wellness-topbright-cradling-deams-one-soothing-night-at-a-time",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-print-design-beauty-tail-product-catalog",

    "https://10turtle.com/work/graphic-branding-print-design-real-estate-construction-tech-software-taylor-sterling-beyond",

    "https://10turtle.com/work/ai-technology-ux-ui-design-website-development-asmblr-simplifying-firearm-builds-customization",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-print-design-priceless-education-invest-in-early-learning",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-kasi-center-empowering-youth-with-knowledge-confidence",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-marketing-branding-fuel-junkie-driven-by-passion",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-grumpy-alpha-boss-when-dominance-meets-desire-sparks-fly",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-the-sugar-papi-a-delightful-online-sweet-shop",

    "https://10turtle.com/work/ai-technology-marketing-branding-shopify-ux-ui-design-ultra-empowering-entrepreneurs-with-seamless-digital-solutions",

    "https://10turtle.com/work/automotive-graphic-branding-marketing-branding-real-estate-construction-bezoz-homes-home-of-ambitions",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-packaging-design-lumore-power-shots",

    "https://10turtle.com/work/food-hospitality-graphic-branding-travel-tourism-lime-grace-blending-elegance-with-every-escape",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-candle-corture-illuminating-moments-crafting-memories",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-marketing-branding-packaging-design-comfofeet-footwear-step-into-comfort-walk-with-confidence",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-halo-notebook-a-smart-personalized-writing-experience",

    "https://10turtle.com/work/ai-technology-health-wellness-ux-ui-design-website-development-yu-nutrition-your-guide-to-a-healthier-balanced-lifestyle",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-dar-pilates-sharjah-a-luxurious-feminine-pilates-experience",

    "https://10turtle.com/work/ai-technology-graphic-branding-print-design-tech-software-allore-studio-keratin-lash-lift-tint",

    "https://10turtle.com/work/arts-culture-graphic-branding-wall-design-nature-in-flight-framed-by-color",

    "https://10turtle.com/work/food-hospitality-ux-ui-design-website-development-taste-my-city-creating-a-culinary-experience-with-taste-my-city",

    "https://10turtle.com/work/automotive-ux-ui-design-website-development-rawad-a-portfolio-redesign-journey",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-marketing-branding-mobile-app-ux-ui-design-logo-ai-transforming-logo-design-with-ai-logo-maker",

    "https://10turtle.com/work/ai-technology-graphic-branding-tech-software-fueling-the-future-mug-design",

    "https://10turtle.com/work/ai-technology-mobile-app-ux-ui-design-voicegen-bring-your-words-to-life-with-iconic-voices",

    "https://10turtle.com/work/health-wellness-ux-ui-design-website-development-its-towotoo-the-digital-journey-of-its-totwotoo",

    "https://10turtle.com/work/graphic-branding-print-design-cfius-advisory-strategic-expertise-for-global-transactions",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-vins-dieux-crafting-a-seamless-wine-experience",

    "https://10turtle.com/work/ai-technology-marketing-branding-ux-ui-design-website-development-the-unda-dog-authentic-digital-marketing-for-real-engagement",

    "https://10turtle.com/work/ai-technology-education-training-tech-software-smart-alarm-wakeup-puzzle",

    "https://10turtle.com/work/automotive-finance-legal-graphic-branding-health-wellness-cureshank-innovating-research-empowering-patients",

    "https://10turtle.com/work/ux-ui-design-website-development-pro-mate-algorithmic-trading-dashboard-trading-analytics-dashboarde",

    "https://10turtle.com/work/automotive-finance-legal-tech-software-travel-tourism-ux-ui-design-website-development-digitaltrvst-redefining-financial-freedom-with-no-limit-spending",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-truelynnco-elevating-user-experience-with-a-modern-redesign",

    "https://10turtle.com/work/automotive-food-hospitality-graphic-branding-marketing-branding-packaging-design-torn-ranch-enticing-and-appetising-drink",

    "https://10turtle.com/work/website-design-amor",

    "https://10turtle.com/work/ux-ui-design-website-development-free-to-love-enhancing-relationships-through-self-discovery",

    "https://10turtle.com/work/ai-technology-education-training-tech-software-mina-your-personal-safety-companion",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-socreatez-dein-business-trifft-genz",

    "https://10turtle.com/work/ai-technology-graphic-branding-marketing-branding-faith-in-fashion-where-style-meets-spirituality",

    "https://10turtle.com/work/graphic-branding-print-design-tech-software-myspy-smart-tracking-anytime-anywhere",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-monkey-tails-spark-imagination-play-and-joy",

    "https://10turtle.com/work/ai-technology-finance-legal-ux-ui-design-website-development-navit-empowering-financial-confidence",

    "https://10turtle.com/work/ai-technology-education-training-docschat-ai-smart-document-assistant",

    "https://10turtle.com/work/ai-technology-graphic-branding-marketing-branding-print-design-tyh-warehousing-solutions-your-trusted-warehouse-partner",

    "https://10turtle.com/work/ai-technology-arts-culture-automotive-graphic-branding-powering-wishing-you-a-warm-and-joyful-holiday-season",

    "https://10turtle.com/work/ai-technology-graphic-branding-marketing-branding-real-estate-construction-iquna-elevating-living-defining-luxury",

    "https://10turtle.com/work/ai-technology-arts-culture-education-training-graphic-branding-reditus-renewing-culture-through-thomistic-thought",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-etf-group-calendar-design",

    "https://10turtle.com/work/ai-technology-automotive-food-hospitality-graphic-branding-packaging-design-parle-packaging-that-speaks-to-your-senses",

    "https://10turtle.com/work/graphic-branding-marketing-branding-wibes-a-creative-space-for-creative-people",

    "https://10turtle.com/work/ai-technology-education-training-tech-software-ai-assistant-smart-business-advice-calls",

    "https://10turtle.com/work/ux-ui-design-website-development-crm-dashboard-enhancing-sales-pipeline-management",

    "https://10turtle.com/work/graphic-branding-tech-software-cyberowl-guarding-the-digital-night",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-marketing-branding-tech-software-geometry-where-precision-meets-aesthetic",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-femme-life-fr-dein-wohlbefinden",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-ev-hipotecas-optimizing-mortgage-solutions-with-expert-guidance",

    "https://10turtle.com/work/travel-tourism-ux-ui-design-website-development-epicbend-epicfrederick-a-travel-tourism-experience",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-kendikids-upendo-eco-friendly-baby-apparel-for-little-ones",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-prishna-elevating-jewelry-elegance",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-print-design-sports-gaming-marvel-sports-catalog-design-elevate-every-game-from-core-to-elite",

    "https://10turtle.com/work/ai-technology-food-hospitality-graphic-branding-print-design-handgemacht-unvergesslich-asiatischer-geschmack-fr-profis",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-tribal-art-where-art-meets-everyday",

    "https://10turtle.com/work/ai-technology-graphic-branding-real-estate-construction-exp-luxury-strategy-sophistication-success",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-m-a-poetry-that-lives-in-moments",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-print-design-tech-software-hilgard-turning-housing-data-into-action",

    "https://10turtle.com/work/graphic-branding-real-estate-construction-exp-luxury-a-memoir-of-letting-go",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-packaging-design-mana-unlock-your-hairs-potentials",

    "https://10turtle.com/work/food-hospitality-health-wellness-ux-ui-design-website-development-psilocybin-chocolate-lucid-awaken-your-senses",

    "https://10turtle.com/work/food-hospitality-ux-ui-design-website-development-revitalizing-caffeine-crashers-a-high-energy-digital-experience",

    "https://10turtle.com/work/ai-technology-education-training-ux-ui-design-website-development-trust-convenience-store-redefining-everyday-shopping-with-quality-convenience",

    "https://10turtle.com/work/ai-technology-ux-ui-design-website-development-indulge-in-natures-richness-one-spoonful-at-a-time",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-tech-software-finalround-ai-confidence-starts-here",

    "https://10turtle.com/work/ai-technology-automotive-finance-legal-graphic-branding-real-estate-construction-movment-mortgage-your-trusted-path-to-homeownership",

    "https://10turtle.com/work/graphic-branding-amara-where-every-wedding-tells-a-story",

    "https://10turtle.com/work/ai-technology-automotive-food-hospitality-graphic-branding-la-cucina-di-nonna-carmelina-a-taste-of-italys-heart",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-mana-brand-guidelines-defining-our-identity",

    "https://10turtle.com/work/ux-ui-design-connecting-businesses-with-world-class-chinese-manufacturing",

    "https://10turtle.com/work/ai-technology-food-hospitality-graphic-branding-harder-day-distilling-co-crafting-tradition-pouring-innovation",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-statement-watches-jewelry-it-offers-timeless-luxury-taste-discerning-individuals",

    "https://10turtle.com/work/ai-technology-graphic-branding-packaging-design-blossom-bunny-pure-simple-hair-detangler",

    "https://10turtle.com/work/ai-technology-graphic-branding-marketing-branding-ai-operations-leader-driving-excellence-with-ai-insights",

    "https://10turtle.com/work/food-hospitality-graphic-branding-h2g-food-co-no-nasties-honest-to-goodness",

    "https://10turtle.com/work/ai-technology-graphic-branding-flipping-fortune-where-every-tip-earns-you-more",

    "https://10turtle.com/work/graphic-branding-marketing-branding-real-estate-construction-andera-a-modern-touch-of-timeless-elegance",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-tech-software-ai-where-complexity-meets-clarity",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-tech-software-psycontech-tech-that-thinks-with-you",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-marketing-branding-packaging-design-mitogenex-fueling-cells-powering-life",

    "https://10turtle.com/work/graphic-branding-tech-software-bps-physical-security-consultants-your-peace-of-mind-partners",

    "https://10turtle.com/work/ai-technology-education-training-graphic-branding-40acreleague-rooted-in-nature-driven-by-equity",

    "https://10turtle.com/work/arts-culture-automotive-graphic-branding-tech-software-lake-city-performance-max-performance-zero-limits",

    "https://10turtle.com/work/ai-technology-graphic-branding-packaging-design-blendi-blend-anywhere",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-packaging-design-stairwedge-comfort-in-every-step",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-print-design-nadeem-sabir-fixed-teeth-in-a-day",

    "https://10turtle.com/work/health-wellness-ux-ui-design-website-development-total-movement-the-home-of-holistic-therapies",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-nfa-bridge-unlock-the-digital-frontier-with-our-exclusive-nfts",

    "https://10turtle.com/work/ai-technology-tech-software-ux-ui-design-website-development-the-mix-button-the-future-of-mixing-is-here-seamless-intelligent-instant",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-seasons-craft-craft-your-quilting-masterpiece",

    "https://10turtle.com/work/automotive-health-wellness-travel-tourism-ux-ui-design-babassu-experience-luxury-spa-services-in-the-comfort-of-your-home",

    "https://10turtle.com/work/real-estate-construction-ux-ui-design-website-development-what-we-do-is-more-than-just-bricks-and-mortar",

    "https://10turtle.com/work/e-commerce-real-estate-construction-ux-ui-design-website-development-case-study-6868b6809534dd45f32c17fc",

    "https://10turtle.com/work/ai-technology-e-commerce-food-hospitality-sports-gaming-ux-ui-design-website-development-football-kitchen-bring-the-taste-of-victory-home",

    "https://10turtle.com/work/ai-technology-marketing-branding-ux-ui-design-website-development-sanvitoagnc-amplifying-creativity-elevating-brands",

    "https://10turtle.com/work/food-hospitality-graphic-branding-marketing-branding-packaging-design-koolers-drink-refresh-repeat",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-packaging-design-ovasave-your-health-personalized",

    "https://10turtle.com/work/ai-technology-graphic-branding-light-digital-where-sophistication-meets-innovation-and-lasting-elegance",

    "https://10turtle.com/work/ai-technology-health-wellness-ux-ui-design-website-development-a-fitness-driven-brand-empowering-individuals-to-overcome-transform-and-succeed",

    "https://10turtle.com/work/ai-technology-automotive-ux-ui-design-website-development-lavender-haze-nail-boutique-website-design",

    "https://10turtle.com/work/ai-technology-health-wellness-ux-ui-design-website-development-empowering-your-journey-through-motherhood",

    "https://10turtle.com/work/ai-technology-ux-ui-design-website-development-atelier-loulou-slow-living-in-the-city-heart",

    "https://10turtle.com/work/automotive-graphic-branding-real-estate-construction-crazy-house-comfort-clarity-care",

    "https://10turtle.com/work/ai-technology-tech-software-ux-ui-design-website-development-blitzware-powering-innovation-one-line-of-code-at-a-time",

    "https://10turtle.com/work/automotive-finance-legal-real-estate-construction-ux-ui-design-website-development-enrich-reality-your-gateway-to-exclusive-properties",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-the-journey-boutique-llc-discover-timeless-elegance",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-explore-a-collection-of-footwear-that-blends-style-for-every-step-you-take",

    "https://10turtle.com/work/ai-technology-finance-legal-sports-gaming-ux-ui-design-colin-chisholm-strategic-financial-advisory-for-the-sports-world-and-beyond",

    "https://10turtle.com/work/marketing-branding-tech-software-ux-ui-design-website-development-wordpress-crafting-a-dynamic-digital-presence",

    "https://10turtle.com/work/e-commerce-ux-ui-design-website-development-kistler-precision-crafted-fishing-gear-for-anglers-who-demand-the-best",

    "https://10turtle.com/work/ai-technology-automotive-education-training-finance-legal-ux-ui-design-website-development-algo26-empowering-growth-through-mentorship-coaching",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-pebbles-pack-stylish-eco-friendly-accessories-for-dogs-who-love-to-stand-out",

    "https://10turtle.com/work/ai-technology-ux-ui-design-moor-gear-up-for-every-journey-survival-and-exploration-with-moor",

    "https://10turtle.com/work/ai-technology-graphic-branding-real-estate-construction-azzure-where-strategy-meets-storytelling",

    "https://10turtle.com/work/ai-technology-automotive-education-training-graphic-branding-health-wellness-la-perle-professional-care-empowered-at-home",

    "https://10turtle.com/work/ai-technology-graphic-branding-print-design-real-estate-construction-tech-software-the-cloud-workforce-flyer-design",

    "https://10turtle.com/work/ai-technology-arts-culture-graphic-branding-fortune-celebrating-diversity-one-stage-at-a-time",

    "https://10turtle.com/work/ai-technology-graphic-branding-print-design-keto-real-guilt-free-indulgence-premium-corporate-gifting",

    "https://10turtle.com/work/ai-technology-graphic-branding-wedgewood-weddings-romantic-occasions-made-fun-flawless-and-easy",

    "https://10turtle.com/work/automotive-food-hospitality-graphic-branding-whole-story-meals-real-food-real-nutrition-anytime-anywhere",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-marketing-branding-packaging-design-mitogenex-fueling-cells-powering-life",

    "https://10turtle.com/work/ai-technology-graphic-branding-tech-software-ai2030-structure-route-scale",

    "https://10turtle.com/work/ai-technology-graphic-branding-packaging-design-the-healing-jar-where-tradition-touches-skin",

    "https://10turtle.com/work/ai-technology-graphic-branding-happy-vibes-for-minis-big-joys-for-little-ones",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-changeainment-transform-with-impact",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-packaging-design-synbio-premium-supplement",

    "https://10turtle.com/work/ai-technology-arts-culture-automotive-graphic-branding-packaging-design-holy-gels-gifting-joy-in-every-pink-toned-drop",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-wuxi-south-petroleum-revolutionizing-the-future-of-lubricants-and-additives-for-a-sustainable-world",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-wrong-to-strong-transform-with-strength",

    "https://10turtle.com/work/education-training-ux-ui-design-website-development-win-tutors-unlock-your-potential-find-your-perfect-tutor-match",

    "https://10turtle.com/work/ai-technology-graphic-branding-health-wellness-packaging-design-repit-usa-unbox-nature-unfold-your-story",

    "https://10turtle.com/work/finance-legal-graphic-branding-fastmarkets-where-structure-meets-growth",

    "https://10turtle.com/work/ai-technology-arts-culture-food-hospitality-graphic-branding-bauer-play-connecting-nature-with-innovation",

    "https://10turtle.com/work/automotive-education-training-travel-tourism-ux-ui-design-website-development-study-key-embrace-the-adventure-of-language-learning",

    "https://10turtle.com/work/automotive-graphic-branding-real-estate-construction-crazy-house-comfort-clarity-care",

    "https://10turtle.com/work/ai-technology-e-commerce-ux-ui-design-website-development-horns-down-raise-your-glass-to-legacy-flavor-and-fire",

    "https://10turtle.com/work/ai-technology-automotive-food-hospitality-graphic-branding-silverland-a-taste-of-the-old-west-served-fresh-daily",

    "https://10turtle.com/work/graphic-branding-tech-software-moneypenny-back-office-accounting-services",

    "https://10turtle.com/work/ai-technology-automotive-graphic-branding-health-wellness-print-design-pharma-desk-advancing-biopharma-through-analytics",

    "https://10turtle.com/work/graphic-branding-packaging-design-ez-labels-design-organizing-made-easy",

    "https://10turtle.com/work/arts-culture-automotive-food-hospitality-graphic-branding-marketing-branding-y-z-b-one-bubble-at-a-time",

    "https://10turtle.com/work/ai-technology-graphic-branding-print-design-real-estate-construction-california-home-company-one-home-at-a-time",

    "https://10turtle.com/work/ai-technology-graphic-branding-imprinted-uw-partner-voor-vastgoedreclame",

    "https://10turtle.com/work/graphic-branding-le-maple-club-rooted-in-canada-styled-for-the-world",

    "https://10turtle.com/work/ecommerce-design-the-sweet-factory",

    "https://10turtle.com/work/ecommerce-design-tcm",

    "https://10turtle.com/work/ai-technology-finance-legal-ux-ui-design-website-development-dovly-ai-powered-credit-managementsmart-simple-and-stress-free",

    "https://10turtle.com/work/automotive-graphic-branding-health-wellness-marketing-branding-packaging-design-calico-everyday-essentials",

    "https://10turtle.com/work/ai-technology-tech-software-ux-ui-design-website-development-the-estimators-streamlining-collision-appraisals-with-precision",
# Add more work links as needed
        ]

        if work_links:
            chatbot.add_work_links(work_links)

        # Try to build index if data file exists
        import os
        if os.path.exists("Jsn_Full_10turtle_crawl_results.json"):
            chatbot.build_index("Jsn_Full_10turtle_crawl_results.json")
            logger.info("✅ Index built from crawl results")
        else:
            logger.warning("⚠️ No crawl results found, running with empty index")

        logger.info("🎉 Advanced RAG Chatbot initialized and ready!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Advanced RAG Chatbot API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global chatbot
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    global chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    try:
        # Process chat request
        response_data = chatbot.chat(request.question)

        # Convert to response model
        return ChatResponse(
            question=response_data["question"],
            answer=response_data["answer"],
            conversation_stage=response_data["conversation_stage"],
            confidence_level=response_data["confidence_level"],
            chat_history=response_data.get("chat_history", []),
            # services=response_data.get("services", []),
            related_works=response_data.get("related_works", []),
            source_documents=response_data.get("source_documents", []),
            suggested_questions=response_data.get("suggested_questions ", []),
            debug_info=response_data.get("debug_info")
        )

    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/reset")
async def reset_conversation():
    """Reset conversation state"""
    global chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    try:
        chatbot.reset()
        return {"message": "Conversation reset successfully"}
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/stats")
async def get_conversation_stats():
    """Get conversation statistics"""
    global chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    try:
        return chatbot.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_index():
    """Rebuild search index (admin endpoint)"""
    global chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    try:
        import os
        if os.path.exists("./10turtle_crawl_results.json"):
            chatbot.build_index("./10turtle_crawl_results.json", force_rebuild=True)
            return {"message": "Index rebuilt successfully"}
        else:
            raise HTTPException(status_code=404, detail="Crawl results file not found")
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")

# ============================================================================
# PRODUCTION DEPLOYMENT HELPERS
# ============================================================================

def create_app():
    """Factory function to create the FastAPI app"""
    return app

# For Gunicorn deployment
if __name__ == "__main__":
    import uvicorn

    # Production configuration
    uvicorn.run(
        "main:app",  # Adjust module name as needed
        host="0.0.0.0",
        port=8000,
        workers=1,  # Adjust based on your server capacity
        log_level="info",
        access_log=True,
        reload=True  # Set to True only in development
    )


