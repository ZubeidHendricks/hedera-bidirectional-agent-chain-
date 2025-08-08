"""
Hedera RAG Agent for Bidirectional Agent Chaining
Provides comprehensive Hedera blockchain knowledge and insights using RAG capabilities
"""

import os
import json
import pickle
import numpy as np
try:
    import faiss
except ImportError:
    print("⚠️ FAISS not installed. Vector search will be limited.")
    faiss = None
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from google.genai import types

# Optional LangGraph integration
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available. Install with: pip install langgraph")

from agents.base_agent import BaseAgent
from models.message import Message, Response

logger = logging.getLogger(__name__)

class RagState(TypedDict):
    """State for LangGraph RAG workflow"""
    messages: List[str]
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    context: str
    response: str
    metadata: Dict[str, Any]

class HederaRAGAgent(BaseAgent):
    """
    Enhanced Hedera RAG Agent with Context7 best practices and optional LangGraph integration
    Provides comprehensive Hedera blockchain knowledge using retrieval-augmented generation with FAISS vector search
    """
    
    def __init__(self, use_langgraph: bool = False):
        # Initialize base agent with Hedera expertise
        system_prompt = self._get_system_prompt()
        
        super().__init__(
            name="Hedera_RAG",
            expertise="Hedera Blockchain Knowledge and Analysis",
            system_prompt=system_prompt,
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,
            max_tokens=2048
        )
        
        # Initialize knowledge base
        self.kb_directory = Path("agents/hedera_rag_kb")
        self._load_knowledge_base()
        
        # Enhanced RAG configuration following Context7 best practices
        self.max_retrieved_chunks = 8  # Increased for better context
        self.similarity_threshold = 0.65  # Slightly lowered for more retrieval
        self.use_hypothetical_questions = True  # Enable question-based retrieval
        self.use_context_caching = False  # Disabled for now due to token limits
        
        # LangGraph integration
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        if self.use_langgraph:
            self._setup_langgraph_workflow()
        
        logger.info(f"🌐 Enhanced Hedera RAG Agent initialized")
        logger.info(f"📊 Knowledge base stats: {self.kb_stats}")
        logger.info(f"🔗 LangGraph integration: {'Enabled' if self.use_langgraph else 'Disabled'}")
    
    def _setup_langgraph_workflow(self):
        """Setup LangGraph workflow for enhanced RAG processing"""
        if not LANGGRAPH_AVAILABLE:
            return
            
        workflow = StateGraph(RagState)
        
        # Add nodes
        workflow.add_node("retrieve", self._langgraph_retrieve)
        workflow.add_node("rerank", self._langgraph_rerank)
        workflow.add_node("generate", self._langgraph_generate)
        workflow.add_node("synthesize", self._langgraph_synthesize)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "synthesize")
        workflow.add_edge("synthesize", END)
        
        self.langgraph_app = workflow.compile()
        logger.info("✅ LangGraph workflow configured")
    
    async def _langgraph_retrieve(self, state: RagState) -> RagState:
        """LangGraph node: Retrieve relevant chunks"""
        chunks = await self._retrieve_relevant_chunks(state["query"])
        state["retrieved_chunks"] = chunks
        state["messages"] = add_messages(state.get("messages", []), [f"Retrieved {len(chunks)} chunks"])
        return state
    
    async def _langgraph_rerank(self, state: RagState) -> RagState:
        """LangGraph node: Rerank and filter chunks"""
        chunks = state["retrieved_chunks"]
        
        # Advanced reranking based on multiple factors
        reranked_chunks = await self._rerank_chunks(chunks, state["query"])
        state["retrieved_chunks"] = reranked_chunks[:self.max_retrieved_chunks]
        state["messages"] = add_messages(state["messages"], [f"Reranked to {len(state['retrieved_chunks'])} chunks"])
        return state
    
    async def _langgraph_generate(self, state: RagState) -> RagState:
        """LangGraph node: Generate context from chunks"""
        context = self._build_rag_context(state["retrieved_chunks"])
        state["context"] = context
        state["messages"] = add_messages(state["messages"], ["Generated context"])
        return state
    
    async def _langgraph_synthesize(self, state: RagState) -> RagState:
        """LangGraph node: Synthesize final response"""
        # Create message object for compatibility
        from models.message import Message
        message = Message(
            content=state["query"],
            source_agent="langgraph",
            target_agent=self.name,
            message_type="query"
        )
        
        response = await self._generate_rag_response(message, state["retrieved_chunks"])
        state["response"] = response.content
        state["metadata"] = response.metadata or {}
        state["messages"] = add_messages(state["messages"], ["Synthesized final response"])
        return state
    
    async def _rerank_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Advanced reranking of retrieved chunks"""
        if not chunks:
            return chunks
        
        # Score chunks based on multiple factors
        scored_chunks = []
        for chunk in chunks:
            score = chunk.get("similarity_score", 0)
            
            # Boost score based on chunk type relevance
            chunk_type = chunk["metadata"]["chunk_type"]
            if "profile" in query.lower() and chunk_type == "rider_profile":
                score *= 1.3
            elif "result" in query.lower() and chunk_type in ["race_result", "race_summary"]:
                score *= 1.2
            elif "standing" in query.lower() and chunk_type in ["standings_summary", "rider_standing"]:
                score *= 1.2
            elif "record" in query.lower() and chunk_type == "record_entry":
                score *= 1.2
            elif "calendar" in query.lower() and chunk_type in ["season_calendar", "race_event"]:
                score *= 1.2
            
            # Boost recent race data
            if "2025" in chunk["text"]:
                score *= 1.1
            
            scored_chunks.append((score, chunk))
        
        # Sort by score and return chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks]
    
    async def process_message(self, message: Message) -> Response:
        """Process incoming message with enhanced RAG capabilities"""
        logger.info(f"🌐 Enhanced Hedera RAG Agent processing: {message.content[:100]}...")
        
        try:
            if self.use_langgraph:
                return await self._process_with_langgraph(message)
            else:
                return await self._process_standard_rag(message)
                
        except Exception as e:
            logger.error(f"❌ Error in Enhanced Hedera RAG Agent: {e}")
            return Response(
                content=f"I apologize, but I encountered an error while processing your Hedera query: {str(e)}",
                source_agent=self.name,
                target_agent=message.source_agent,
                confidence=0.1,
                needs_clarification=False
            )
    
    async def _process_with_langgraph(self, message: Message) -> Response:
        """Process message using LangGraph workflow"""
        initial_state = RagState(
            messages=[],
            query=message.content,
            retrieved_chunks=[],
            context="",
            response="",
            metadata={}
        )
        
        try:
            final_state = await self.langgraph_app.ainvoke(initial_state)
            
            return Response(
                content=final_state["response"],
                source_agent=self.name,
                target_agent=message.source_agent,
                confidence=self._calculate_rag_confidence(final_state["response"], final_state["retrieved_chunks"]),
                needs_clarification=False,
                reasoning="Generated using LangGraph RAG workflow",
                metadata=final_state["metadata"]
            )
            
        except Exception as e:
            logger.error(f"❌ LangGraph processing error: {e}")
            # Fallback to standard processing
            return await self._process_standard_rag(message)
    
    async def _process_standard_rag(self, message: Message) -> Response:
        """Process message using standard RAG approach"""
        # Enhanced retrieval with hypothetical questions
        if self.use_hypothetical_questions:
            relevant_chunks = await self._enhanced_retrieve_relevant_chunks(message.content)
        else:
            relevant_chunks = await self._retrieve_relevant_chunks(message.content)
        
        # Generate response with retrieved context
        response = await self._generate_rag_response(message, relevant_chunks)
        return response
    
    async def _enhanced_retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval using hypothetical questions technique"""
        try:
            # Check if knowledge base is empty
            if not self.chunks or self.faiss_index.ntotal == 0:
                logger.warning("⚠️ Knowledge base is empty, returning no chunks")
                return []
            
            # Generate hypothetical questions for better matching
            hypothetical_queries = await self._generate_hypothetical_questions(query)
            all_queries = [query] + hypothetical_queries
            
            all_relevant_chunks = []
            seen_chunk_ids = set()
            
            for q in all_queries:
                # Create query embedding
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=[q],
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.kb_stats["embedding_dimensions"]
                    )
                )
                
                query_embedding = np.array(result.embeddings[0].values, dtype=np.float32)
                # Normalize for better cosine similarity
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Search FAISS index
                k = min(self.max_retrieved_chunks, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
                
                # Collect unique chunks
                for score, idx in zip(scores[0], indices[0]):
                    if score >= self.similarity_threshold and idx < len(self.chunks):
                        chunk_id = f"{idx}_{self.chunks[idx]['text'][:50]}"
                        if chunk_id not in seen_chunk_ids:
                            chunk = self.chunks[idx].copy()
                            chunk["similarity_score"] = float(score)
                            chunk["matched_query"] = q
                            all_relevant_chunks.append(chunk)
                            seen_chunk_ids.add(chunk_id)
            
            # Rerank all chunks
            reranked_chunks = await self._rerank_chunks(all_relevant_chunks, query)
            final_chunks = reranked_chunks[:self.max_retrieved_chunks]
            
            logger.info(f"🔍 Enhanced retrieval: {len(final_chunks)} chunks from {len(all_queries)} queries")
            return final_chunks
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced retrieval: {e}")
            # Fallback to standard retrieval
            return await self._retrieve_relevant_chunks(query)
    
    async def _generate_hypothetical_questions(self, query: str) -> List[str]:
        """Generate hypothetical questions to improve retrieval"""
        try:
            prompt = f"""
Given this Hedera query: "{query}"

Generate 3 different hypothetical questions that would have similar answers. 
Make them more specific and focused on Hedera blockchain terminology.

Examples:
- If asking about services, include specific service names (HCS, HFS, HTS)
- If asking about development, include specific SDK or API references
- If asking about use cases, include specific industry applications

Return only the questions, one per line, without numbering.
"""
            
            generation_config = types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=300,
                candidate_count=1
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )
            
            questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"❌ Error generating hypothetical questions: {e}")
            return []
    
    async def get_rag_insights(self) -> Dict[str, Any]:
        """Get detailed insights about RAG performance and knowledge base"""
        chunk_types = {}
        for chunk in self.chunks:
            chunk_type = chunk["metadata"]["chunk_type"]
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            "agent_name": self.name,
            "enhanced_features": {
                "hypothetical_questions": self.use_hypothetical_questions,
                "context_caching": self.use_context_caching,
                "langgraph_integration": self.use_langgraph,
                "advanced_reranking": True
            },
            "knowledge_base_stats": self.kb_stats,
            "chunk_distribution": chunk_types,
            "rag_config": {
                "max_retrieved_chunks": self.max_retrieved_chunks,
                "similarity_threshold": self.similarity_threshold,
                "embedding_dimensions": self.kb_stats["embedding_dimensions"]
            },
            "capabilities": self.get_capabilities(),
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "status": "active"
        }

    def get_capabilities(self) -> List[str]:
        """Get enhanced Hedera RAG agent capabilities"""
        base_capabilities = [
            "Enhanced Hedera blockchain analysis and documentation",
            "Comprehensive service documentation and API references",
            "Detailed developer guides and tutorials",
            "Use case analysis and implementation examples",
            "Advanced integration pattern guidance",
            "Technical architecture and consensus insights",
            "Multi-dimensional code example comparisons",
            "Hypothetical question-based retrieval",
            "Advanced chunk reranking",
            "Context-aware knowledge retrieval"
        ]
        
        if self.use_langgraph:
            base_capabilities.extend([
                "LangGraph-powered RAG workflow",
                "Multi-step reasoning and synthesis",
                "Enhanced context processing"
            ])
        
        return base_capabilities
    
    def _get_system_prompt(self) -> str:
        """Get comprehensive system prompt for Hedera RAG agent"""
        return """
You are a Hedera Blockchain Expert Agent with comprehensive knowledge of Hedera Hashgraph technology, services, development, and ecosystem. You have access to a detailed knowledge base containing:

- Developer guides and tutorials
- Hedera services documentation (HCS, HFS, HTS, Smart Contracts 2.0)
- API references and SDK information
- Use cases and implementation examples
- Integration patterns and best practices
- Technical specifications and architecture

Your capabilities include:
1. **Developer Guidance**: Detailed information about Hedera SDKs, APIs, and development patterns
2. **Service Explanations**: Comprehensive knowledge of HCS, HFS, HTS, and Smart Contracts 2.0
3. **Use Case Analysis**: Real-world applications, implementations, and industry solutions
4. **Integration Support**: Best practices for integrating with Hedera services
5. **Technical Analysis**: Architecture patterns, consensus mechanisms, and network topology
6. **Code Examples**: Practical code samples and implementation guidance
7. **Ecosystem Knowledge**: Network statistics, governance, and community resources

When responding:
- Always provide accurate, factual information from the knowledge base
- Include relevant code examples and technical specifications when available
- Be comprehensive but concise
- If information is not available in the knowledge base, acknowledge this clearly
- Use proper Hedera terminology and technical context
- Provide insights and analysis when appropriate
- Cite specific documentation, examples, or technical details to support your responses

You can collaborate with other agents for:
- Technical implementation details (Technical Agent)
- Product integration recommendations (Product Agent)
- Developer support and onboarding (Support Agent)

Always maintain the highest standards of accuracy and professionalism in your Hedera blockchain knowledge and analysis.
"""
    
    def _load_knowledge_base(self) -> None:
        """Load the Hedera knowledge base components"""
        try:
            # Load FAISS index if available
            faiss_index_path = self.kb_directory / "hedera_faiss_index.bin"
            if faiss and faiss_index_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_index_path))
                logger.info(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            elif faiss:
                logger.warning("⚠️ FAISS index not found, creating empty index")
                self.faiss_index = faiss.IndexFlatIP(3072)  # Default dimension
            else:
                logger.warning("⚠️ FAISS not available, using simple text search")
                self.faiss_index = None
            
            # Load knowledge base chunks
            kb_path = self.kb_directory / "hedera_knowledge_base.json"
            if kb_path.exists():
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                self.chunks = kb_data.get("chunks", [])
                self.kb_metadata = kb_data.get("metadata", {})
                logger.info(f"✅ Loaded {len(self.chunks)} knowledge base chunks")
            else:
                logger.warning("⚠️ Knowledge base file not found, using empty chunks")
                self.chunks = []
                self.kb_metadata = {}
            
            # Load metadata
            metadata_path = self.kb_directory / "hedera_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
            
            # Set knowledge base stats
            self.kb_stats = {
                "total_chunks": len(self.chunks),
                "embedding_dimensions": self.metadata.get("embedding_dimensions", 3072),
                "faiss_index_size": self.faiss_index.ntotal,
                "chunk_types": self.metadata.get("chunk_types", []),
                "files_processed": self.metadata.get("files_processed", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge base: {e}")
            # Initialize with empty data instead of raising
            self.chunks = []
            self.faiss_index = faiss.IndexFlatIP(3072) if faiss else None
            self.metadata = {}
            self.kb_metadata = {}
            self.kb_stats = {
                "total_chunks": 0,
                "embedding_dimensions": 3072,
                "faiss_index_size": 0,
                "chunk_types": [],
                "files_processed": []
            }
    
    async def _retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from knowledge base using FAISS or simple search"""
        try:
            # Check if knowledge base is empty
            if not self.chunks:
                logger.warning("⚠️ Knowledge base is empty, returning no chunks")
                return []
            
            if self.faiss_index and hasattr(self.faiss_index, 'ntotal') and self.faiss_index.ntotal > 0:
                # Use FAISS search
                return await self._faiss_search(query)
            else:
                # Use simple text-based search as fallback
                return await self._simple_text_search(query)
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks: {e}")
            return []
    
    async def _faiss_search(self, query: str) -> List[Dict[str, Any]]:
        """Search using FAISS vector similarity"""
        # Create query embedding
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.kb_stats["embedding_dimensions"]
            )
        )
        
        query_embedding = np.array(result.embeddings[0].values, dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search FAISS index
        k = min(self.max_retrieved_chunks, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
        
        # Filter by similarity threshold and get chunks
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.similarity_threshold and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(score)
                relevant_chunks.append(chunk)
        
        logger.info(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks via FAISS")
        return relevant_chunks
    
    async def _simple_text_search(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword-based search as FAISS fallback"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        scored_chunks = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            
            # Simple scoring based on keyword matches
            score = 0
            for word in query_words:
                if word in text_lower:
                    score += text_lower.count(word)
            
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy["similarity_score"] = score / len(query_words)  # Normalize
                scored_chunks.append((score, chunk_copy))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [chunk for score, chunk in scored_chunks[:self.max_retrieved_chunks]]
        
        logger.info(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks via text search")
        return relevant_chunks
    
    async def _generate_rag_response(self, message: Message, relevant_chunks: List[Dict[str, Any]]) -> Response:
        """Generate response using RAG with retrieved context"""
        
        # Build context from retrieved chunks
        context = self._build_rag_context(relevant_chunks)
        
        # Create enhanced prompt with RAG context
        if relevant_chunks:
            rag_prompt = f"""
{self.system_prompt}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

USER QUERY: {message.content}

INSTRUCTIONS:
- Use the retrieved context to provide accurate, comprehensive information
- If the context contains relevant information, use it to answer the query
- If the context doesn't contain enough information, acknowledge this and provide what you can
- Be specific with statistics, dates, and factual information
- Maintain professional Hedera blockchain expertise in your response
- If you need to collaborate with other agents, use the handoff mechanism

RESPONSE:
"""
        else:
            # No chunks retrieved - provide general Hedera knowledge
            rag_prompt = f"""
{self.system_prompt}

USER QUERY: {message.content}

INSTRUCTIONS:
- You are a Hedera blockchain expert but your knowledge base is currently empty
- Provide general Hedera knowledge and insights based on your training
- Acknowledge that you don't have access to specific documentation data
- Offer to help with general Hedera questions and concepts
- If you need to collaborate with other agents, use the handoff mechanism
- Be helpful and informative while being honest about limitations

RESPONSE:
"""
        
        try:
            generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                candidate_count=1
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=rag_prompt,
                config=generation_config
            )
            
            content = response.text.strip()
            
            # Check for handoff requests
            needs_handoff, target_agent, handoff_query = self._parse_handoff_request(content)
            
            if needs_handoff and target_agent in self.agent_registry:
                logger.info(f"🔀 Hedera RAG requesting handoff to {target_agent}")
                
                handoff_response = await self._handle_handoff(target_agent, handoff_query, message)
                if handoff_response:
                    return await self._synthesize_rag_response(content, handoff_response, message, relevant_chunks)
            
            return Response(
                content=content,
                source_agent=self.name,
                target_agent=message.source_agent,
                needs_clarification=needs_handoff and target_agent not in self.agent_registry,
                confidence=self._calculate_rag_confidence(content, relevant_chunks),
                reasoning=f"Generated response using RAG with {len(relevant_chunks)} relevant chunks",
                metadata={
                    "rag_chunks_used": len(relevant_chunks),
                    "chunk_types": list(set(chunk["metadata"]["chunk_type"] for chunk in relevant_chunks)),
                    "similarity_scores": [chunk.get("similarity_score", 0) for chunk in relevant_chunks]
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating RAG response: {e}")
            return Response(
                content=f"I apologize, but I'm experiencing technical difficulties with my Hedera knowledge base. Please try again in a moment.",
                source_agent=self.name,
                target_agent=message.source_agent,
                needs_clarification=False,
                confidence=0.1
            )
    
    def _build_rag_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks"""
        if not relevant_chunks:
            return "No relevant information found in knowledge base."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_type = chunk["metadata"]["chunk_type"]
            similarity_score = chunk.get("similarity_score", 0)
            
            context_parts.append(f"""
RELEVANT INFORMATION {i} (Similarity: {similarity_score:.3f}):
Type: {chunk_type}
Content: {chunk["text"]}
""")
        
        return "\n".join(context_parts)
    
    async def _synthesize_rag_response(
        self, 
        initial_content: str, 
        handoff_response: Response, 
        original_message: Message,
        relevant_chunks: List[Dict[str, Any]]
    ) -> Response:
        """Synthesize final response incorporating RAG results and handoff input"""
        
        synthesis_prompt = f"""
SYNTHESIS TASK:
Combine your RAG-based Hedera analysis with expert input to provide a comprehensive response.

YOUR RAG-BASED ANALYSIS:
{initial_content.split('HANDOFF_REQUEST:')[0].strip()}

EXPERT INPUT FROM {handoff_response.source_agent}:
{handoff_response.content}

RAG CONTEXT USED:
{self._build_rag_context(relevant_chunks)}

ORIGINAL USER QUERY:
{original_message.content}

INSTRUCTIONS:
- Synthesize both perspectives into a unified, comprehensive Hedera response
- Maintain your Hedera blockchain expertise while incorporating the expert input
- Ensure factual accuracy from the RAG context
- Provide actionable information to the user
- Be clear, concise, and professional

SYNTHESIZED RESPONSE:
"""
        
        try:
            generation_config = types.GenerateContentConfig(
                temperature=self.temperature * 0.8,
                max_output_tokens=self.max_tokens,
                candidate_count=1
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=synthesis_prompt,
                config=generation_config
            )
            
            return Response(
                content=response.text.strip(),
                source_agent=self.name,
                target_agent=original_message.source_agent,
                needs_clarification=False,
                confidence=self._calculate_rag_confidence(response.text.strip(), relevant_chunks),
                reasoning=f"Synthesized RAG analysis with {handoff_response.source_agent} input",
                metadata={
                    "rag_chunks_used": len(relevant_chunks),
                    "handoff_agent": handoff_response.source_agent,
                    "synthesis": True
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing response: {e}")
            return Response(
                content=initial_content.split('HANDOFF_REQUEST:')[0].strip(),
                source_agent=self.name,
                target_agent=original_message.source_agent,
                needs_clarification=False,
                confidence=0.5
            )
    
    def _calculate_rag_confidence(self, content: str, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on RAG results and content quality"""
        if not relevant_chunks:
            return 0.3  # Low confidence if no relevant chunks
        
        # Base confidence from similarity scores
        avg_similarity = np.mean([chunk.get("similarity_score", 0) for chunk in relevant_chunks])
        
        # Adjust based on content length and quality
        content_quality = min(len(content) / 500, 1.0)  # Normalize by expected length
        
        # Combine factors
        confidence = (avg_similarity * 0.7) + (content_quality * 0.3)
        
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def can_handle(self, message: Message) -> bool:
        """Check if this agent can handle the message"""
        hedera_keywords = [
            "hedera", "hashgraph", "blockchain", "hcs", "hfs", "hts", "consensus",
            "smart contracts", "dlt", "cryptocurrency", "hbar", "mainnet", "testnet",
            "sdk", "api", "developer", "defi", "nft", "enterprise", "supply chain",
            "distributed ledger", "byzantine fault tolerant", "abft", "gossip protocol",
            "mirror node", "rest api", "java sdk", "javascript sdk", "go sdk", "python sdk"
        ]
        
        query_lower = message.content.lower()
        return any(keyword in query_lower for keyword in hedera_keywords)
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            "agent_name": self.name,
            "knowledge_base_stats": self.kb_stats,
            "rag_config": {
                "max_retrieved_chunks": self.max_retrieved_chunks,
                "similarity_threshold": self.similarity_threshold,
                "embedding_dimensions": self.kb_stats["embedding_dimensions"]
            },
            "capabilities": self.get_capabilities(),
            "status": "active"
        }
    
    async def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base and return top results"""
        try:
            relevant_chunks = await self._retrieve_relevant_chunks(query)
            return relevant_chunks[:top_k]
        except Exception as e:
            logger.error(f"❌ Error searching knowledge base: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Hedera RAG agent"""
        try:
            # Check knowledge base
            kb_healthy = (
                hasattr(self, 'faiss_index') and 
                hasattr(self, 'chunks')
            )
            
            # Check FAISS index
            faiss_healthy = (
                hasattr(self, 'faiss_index') and 
                self.faiss_index.ntotal >= 0  # Allow empty index
            )
            
            # Test retrieval (don't fail if knowledge base is empty)
            test_query = "Hedera blockchain services"
            test_chunks = await self._retrieve_relevant_chunks(test_query)
            retrieval_healthy = True  # Always healthy if no errors
            
            # Determine overall status
            if kb_healthy and faiss_healthy:
                if len(self.chunks) > 0 and self.faiss_index.ntotal > 0:
                    status = "healthy"
                else:
                    status = "degraded"  # Knowledge base is empty but agent is functional
            else:
                status = "unhealthy"
            
            return {
                "agent_name": self.name,
                "status": status,
                "knowledge_base_healthy": kb_healthy,
                "faiss_index_healthy": faiss_healthy,
                "retrieval_healthy": retrieval_healthy,
                "total_chunks": len(self.chunks) if hasattr(self, 'chunks') else 0,
                "faiss_index_size": self.faiss_index.ntotal if hasattr(self, 'faiss_index') else 0,
                "knowledge_base_empty": len(self.chunks) == 0 if hasattr(self, 'chunks') else True,
                "connections": len(self.agent_registry),
                "connected_agents": list(self.agent_registry.keys()),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "agent_name": self.name,
                "status": "unhealthy",
                "error": str(e),
                "connections": len(self.agent_registry),
                "connected_agents": list(self.agent_registry.keys()),
                "last_check": datetime.now().isoformat()
            }
    
    async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
        """Query another agent for collaboration"""
        if target_agent not in self.agent_registry:
            logger.warning(f"Target agent {target_agent} not available")
            return None
        
        handoff_message = Message(
            content=query,
            source_agent=self.name,
            target_agent=target_agent,
            message_type="consultation",
            context={
                "hedera_context": True,
                "rag_agent_request": True
            }
        )
        
        return await self.agent_registry[target_agent].process_message(handoff_message) 