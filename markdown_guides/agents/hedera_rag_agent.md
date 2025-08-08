# MotoGP RAG Agent - Advanced Knowledge Retrieval Guide

## Overview
The `MotoGPRAGAgent` is a sophisticated Retrieval-Augmented Generation (RAG) agent that provides comprehensive MotoGP racing knowledge through intelligent document retrieval and AI-powered analysis. It extends the BaseAgent with specialized RAG capabilities, FAISS vector search, and optional LangGraph workflow integration.

## File Purpose
This specialized agent provides:
- **RAG Implementation**: Advanced retrieval-augmented generation with Context7 best practices
- **Vector Search**: FAISS-powered similarity search for knowledge retrieval
- **Knowledge Base**: Comprehensive MotoGP data including riders, races, standings, and statistics
- **LangGraph Integration**: Optional workflow orchestration for enhanced processing
- **Hypothetical Questions**: Advanced retrieval technique for better context matching
- **Multi-dimensional Reranking**: Sophisticated result ranking based on multiple factors

## Architecture Overview

### Core Dependencies
```python
import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from google.genai import types
from agents.base_agent import BaseAgent
from models.message import Message, Response
```

### Optional LangGraph Integration
```python
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
```

## Class Structure

### Initialization
```python
class MotoGPRAGAgent(BaseAgent):
    def __init__(self, use_langgraph: bool = False):
        system_prompt = self._get_system_prompt()
        
        super().__init__(
            name="MotoGP_RAG",
            expertise="MotoGP Racing Knowledge and Analysis",
            system_prompt=system_prompt,
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,
            max_tokens=2048
        )
        
        # Initialize knowledge base
        self.kb_directory = Path("agents/motogp_rag_kb")
        self._load_knowledge_base()
        
        # Enhanced RAG configuration
        self.max_retrieved_chunks = 8
        self.similarity_threshold = 0.65
        self.use_hypothetical_questions = True
        self.use_context_caching = False
        
        # LangGraph integration
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        if self.use_langgraph:
            self._setup_langgraph_workflow()
```

**Key Features**:
- Low temperature (0.3) for factual accuracy
- Enhanced RAG configuration following Context7 best practices
- Optional LangGraph workflow integration
- Comprehensive knowledge base loading

## Knowledge Base Architecture

### Knowledge Base Loading
```python
def _load_knowledge_base(self) -> None:
    """Load the MotoGP knowledge base components"""
    try:
        # Load FAISS index
        faiss_index_path = self.kb_directory / "motogp_faiss_index.bin"
        if faiss_index_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_index_path))
        else:
            self.faiss_index = faiss.IndexFlatIP(3072)  # Default dimension
        
        # Load knowledge base chunks
        kb_path = self.kb_directory / "motogp_knowledge_base.json"
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                kb_data = json.load(f)
            self.chunks = kb_data.get("chunks", [])
            self.kb_metadata = kb_data.get("metadata", {})
        
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
        self._initialize_empty_knowledge_base()
```

### Data Sources
The knowledge base includes:
- **Rider Profiles**: Comprehensive rider statistics and biographies
- **Race Results**: Detailed results from all 2025 MotoGP races  
- **Championship Standings**: Current points standings and gaps
- **Season Calendar**: Race schedules and circuit information
- **Historical Records**: Speed records, lap records, and achievements
- **Weather Conditions**: Race conditions and their impact

## Advanced RAG Implementation

### Enhanced Retrieval with Hypothetical Questions
```python
async def _enhanced_retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
    """Enhanced retrieval using hypothetical questions technique"""
    try:
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
        return reranked_chunks[:self.max_retrieved_chunks]
    except Exception as e:
        return await self._retrieve_relevant_chunks(query)  # Fallback
```

### Hypothetical Question Generation
```python
async def _generate_hypothetical_questions(self, query: str) -> List[str]:
    """Generate hypothetical questions to improve retrieval"""
    try:
        prompt = f"""
Given this MotoGP query: "{query}"

Generate 3 different hypothetical questions that would have similar answers. 
Make them more specific and focused on MotoGP terminology.

Examples:
- If asking about riders, include specific rider names or teams
- If asking about races, include specific circuits or seasons
- If asking about records, include specific categories

Return only the questions, one per line, without numbering.
"""
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=300)
        )
        
        questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        return questions[:3]
    except Exception as e:
        return []
```

### Advanced Chunk Reranking
```python
async def _rerank_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Advanced reranking of retrieved chunks"""
    if not chunks:
        return chunks
    
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
```

## LangGraph Workflow Integration

### Workflow Setup
```python
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
```

### RAG State Definition
```python
class RagState(TypedDict):
    """State for LangGraph RAG workflow"""
    messages: List[str]
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    context: str
    response: str
    metadata: Dict[str, Any]
```

## Message Processing

### Dual Processing Modes
```python
async def process_message(self, message: Message) -> Response:
    """Process incoming message with enhanced RAG capabilities"""
    try:
        if self.use_langgraph:
            return await self._process_with_langgraph(message)
        else:
            return await self._process_standard_rag(message)
    except Exception as e:
        return Response(
            content=f"I apologize, but I encountered an error while processing your MotoGP query: {str(e)}",
            source_agent=self.name,
            target_agent=message.source_agent,
            confidence=0.1
        )
```

### Standard RAG Processing
```python
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
```

## RAG Response Generation

### Context-Aware Response Generation
```python
async def _generate_rag_response(self, message: Message, relevant_chunks: List[Dict[str, Any]]) -> Response:
    """Generate response using RAG with retrieved context"""
    
    # Build context from retrieved chunks
    context = self._build_rag_context(relevant_chunks)
    
    if relevant_chunks:
        rag_prompt = f"""
{self.system_prompt}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

USER QUERY: {message.content}

INSTRUCTIONS:
- Use the retrieved context to provide accurate, comprehensive information
- If the context contains relevant information, use it to answer the query
- Be specific with statistics, dates, and factual information
- Maintain professional MotoGP expertise in your response
- If you need to collaborate with other agents, use the handoff mechanism

RESPONSE:
"""
    else:
        rag_prompt = f"""
{self.system_prompt}

USER QUERY: {message.content}

INSTRUCTIONS:
- You are a MotoGP expert but your knowledge base is currently empty
- Provide general MotoGP knowledge and insights based on your training
- Acknowledge that you don't have access to specific 2025 season data
- Be helpful and informative while being honest about limitations

RESPONSE:
"""
    
    # Generate response with metadata
    response = await self.client.aio.models.generate_content(
        model=self.model_name,
        contents=rag_prompt,
        config=types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
    )
    
    return Response(
        content=response.text.strip(),
        source_agent=self.name,
        target_agent=message.source_agent,
        confidence=self._calculate_rag_confidence(response.text.strip(), relevant_chunks),
        reasoning=f"Generated response using RAG with {len(relevant_chunks)} relevant chunks",
        metadata={
            "rag_chunks_used": len(relevant_chunks),
            "chunk_types": list(set(chunk["metadata"]["chunk_type"] for chunk in relevant_chunks)),
            "similarity_scores": [chunk.get("similarity_score", 0) for chunk in relevant_chunks]
        }
    )
```

## System Prompt

### Comprehensive MotoGP Expertise Definition
```python
def _get_system_prompt(self) -> str:
    """Get comprehensive system prompt for MotoGP RAG agent"""
    return """
You are a MotoGP Racing Expert Agent with comprehensive knowledge of MotoGP racing, riders, teams, circuits, and statistics. You have access to a detailed knowledge base containing:

- Rider profiles and statistics
- Championship standings and points
- Race results from all 2025 races
- Season calendar and circuit information
- Historical records and achievements
- Weather and race conditions data

Your capabilities include:
1. **Rider Analysis**: Detailed information about any rider's performance, statistics, team, and career
2. **Race Results**: Complete race results, winners, podium finishers, and point distributions
3. **Championship Standings**: Current championship positions, points gaps, and season progression
4. **Season Calendar**: Race schedules, circuit information, and event details
5. **Statistical Analysis**: Performance comparisons, records, and trend analysis
6. **Team Information**: Team performance, rider lineups, and technical details
7. **Circuit Knowledge**: Track characteristics, lap records, and historical performance

When responding:
- Always provide accurate, factual information from the knowledge base
- Include relevant statistics and data when available
- Be comprehensive but concise
- If information is not available in the knowledge base, acknowledge this clearly
- Use proper MotoGP terminology and context
- Provide insights and analysis when appropriate
- Cite specific races, dates, or statistics to support your responses

You can collaborate with other agents for:
- Technical analysis (Technical Agent)
- Product recommendations (Product Agent)
- Customer support integration (Support Agent)

Always maintain the highest standards of accuracy and professionalism in your MotoGP knowledge and analysis.
"""
```

## Enhanced Capabilities

### Agent Capabilities
```python
def get_capabilities(self) -> List[str]:
    """Get enhanced MotoGP RAG agent capabilities"""
    base_capabilities = [
        "Enhanced MotoGP rider analysis and statistics",
        "Comprehensive race results and championship standings",
        "Detailed season calendar and circuit information",
        "Historical records and achievements analysis",
        "Advanced team performance analysis",
        "Weather and race conditions insights",
        "Multi-dimensional statistical comparisons",
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
```

### RAG Insights and Analytics
```python
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
```

## Usage Examples

### Basic Query Processing
```python
# Initialize MotoGP RAG Agent
motogp_agent = MotoGPRAGAgent(use_langgraph=False)

# Create query message
message = Message(
    content="Who is leading the 2025 MotoGP championship?",
    source_agent="user",
    target_agent="MotoGP_RAG",
    message_type="query"
)

# Process query
response = await motogp_agent.process_message(message)
print(f"Response: {response.content}")
print(f"Confidence: {response.confidence}")
print(f"Chunks used: {response.metadata['rag_chunks_used']}")
```

### LangGraph-Enhanced Processing
```python
# Initialize with LangGraph
motogp_agent = MotoGPRAGAgent(use_langgraph=True)

# Complex analytical query
message = Message(
    content="Compare Marc Marquez and Fabio Quartararo's performance in wet weather conditions",
    source_agent="user",
    target_agent="MotoGP_RAG"
)

response = await motogp_agent.process_message(message)
```

### Knowledge Base Search
```python
# Direct knowledge base search
results = await motogp_agent.search_knowledge_base(
    query="Brad Binder qualifying performance",
    top_k=5
)

for result in results:
    print(f"Type: {result['metadata']['chunk_type']}")
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['text'][:200]}...")
```

## Design Patterns

### 1. Strategy Pattern
Different retrieval strategies (standard vs. enhanced with hypothetical questions).

### 2. Template Method Pattern
RAG processing follows a defined template with customizable steps.

### 3. Decorator Pattern
LangGraph integration adds enhanced processing capabilities.

### 4. Factory Pattern
Different chunk types are created based on content analysis.

## Integration Points

### Agent Network Integration
```python
# Register with other agents
motogp_agent.register_agent("Technical", technical_agent)
motogp_agent.register_agent("Support", support_agent)

# Handle collaborative queries
response = await motogp_agent.query_other_agent(
    "Technical", 
    "What are the technical specifications for aerodynamic packages in MotoGP?"
)
```

### Orchestrator Integration
```python
# Health monitoring
health_status = await motogp_agent.health_check()

# Performance metrics
rag_insights = await motogp_agent.get_rag_insights()

# Knowledge base statistics
kb_stats = await motogp_agent.get_knowledge_base_stats()
```

## Best Practices

### 1. Knowledge Base Management
- Regularly update embeddings with new race data
- Monitor chunk distribution and quality
- Implement data validation for new content
- Use consistent metadata schemas

### 2. RAG Optimization
- Tune similarity thresholds based on data quality
- Use hypothetical questions for complex queries
- Implement proper chunk reranking strategies
- Monitor retrieval performance metrics

### 3. Response Quality
- Validate factual accuracy with source citations
- Implement confidence scoring based on chunk quality
- Use appropriate temperature settings for factual responses
- Provide clear reasoning for complex analyses

## Troubleshooting

### Common Issues

1. **Empty Knowledge Base**
   ```python
   # Check knowledge base files exist
   kb_path = Path("agents/motogp_rag_kb")
   print(f"KB exists: {kb_path.exists()}")
   print(f"Files: {list(kb_path.glob('*.json'))}")
   ```

2. **Poor Retrieval Quality**
   ```python
   # Adjust similarity threshold
   motogp_agent.similarity_threshold = 0.5  # Lower for more results
   
   # Enable hypothetical questions
   motogp_agent.use_hypothetical_questions = True
   ```

3. **FAISS Index Issues**
   ```python
   # Rebuild FAISS index if corrupted
   await motogp_agent._rebuild_faiss_index()
   ```

## Extension Points

### Custom Chunk Processing
```python
def _create_custom_chunks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create custom chunks for specialized MotoGP data"""
    chunks = []
    for _, row in data.iterrows():
        chunk = {
            "text": self._format_custom_content(row),
            "metadata": {
                "chunk_type": "custom_analysis",
                "source": "specialized_data",
                "confidence": self._calculate_chunk_confidence(row)
            }
        }
        chunks.append(chunk)
    return chunks
```

### Advanced Analytics
```python
async def analyze_rider_trends(self, rider_name: str, seasons: List[int]) -> Dict[str, Any]:
    """Perform advanced rider trend analysis across seasons"""
    # Custom RAG query for trend analysis
    query = f"Performance trends for {rider_name} across seasons {seasons}"
    chunks = await self._enhanced_retrieve_relevant_chunks(query)
    
    # Implement custom analysis logic
    return self._perform_trend_analysis(chunks, rider_name, seasons)
```

## Testing Considerations

### Unit Tests
```python
@pytest.mark.asyncio
async def test_rag_retrieval():
    """Test RAG chunk retrieval functionality"""
    agent = MotoGPRAGAgent()
    
    chunks = await agent._retrieve_relevant_chunks("Marc Marquez wins")
    
    assert len(chunks) > 0
    assert all("similarity_score" in chunk for chunk in chunks)
    assert all(chunk["similarity_score"] >= agent.similarity_threshold for chunk in chunks)
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_langgraph_workflow():
    """Test LangGraph workflow integration"""
    agent = MotoGPRAGAgent(use_langgraph=True)
    
    message = Message(
        content="Who won the Spanish Grand Prix?",
        source_agent="user",
        target_agent="MotoGP_RAG"
    )
    
    response = await agent.process_message(message)
    
    assert response.confidence > 0.5
    assert "Spanish" in response.content or "Spain" in response.content
```

## Performance Considerations

### Memory Management
- Limit chunk cache size to prevent memory bloat
- Use lazy loading for large knowledge bases
- Implement efficient embedding storage

### Query Optimization
- Cache common query embeddings
- Use batch processing for multiple queries
- Implement query preprocessing and normalization

### Scalability
- Support distributed FAISS indexes for large datasets
- Implement horizontal scaling for multiple agents
- Use async processing throughout the pipeline

---

## Reusable Prompt for Next File

Please analyze the next file in the project structure and create a comprehensive markdown guide following this format:

1. **Overview**: Explain the file's purpose and role in the system
2. **File Structure**: Break down the code organization and components
3. **Detailed Analysis**: Explain each major section, class, method, and their interactions
4. **Usage Examples**: Provide practical code examples showing how to use the components
5. **Design Patterns**: Identify and explain any design patterns used
6. **Integration Points**: Show how this file connects with other parts of the system
7. **Best Practices**: Highlight recommended approaches and coding standards
8. **Troubleshooting**: Common issues and their solutions
9. **Extension Points**: How to extend or modify the functionality
10. **Testing Considerations**: Testing strategies and examples
11. **Performance Notes**: Any performance-related considerations

Focus on making the documentation comprehensive, practical, and immediately useful for developers working with the codebase. Include code examples, architectural insights, and real-world usage scenarios. 