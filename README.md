# Hedera Bidirectional Agent Chaining System

## 🌐 Advanced Multi-Agent AI System with LangGraph & Google Gemini for Hedera Blockchain

*A sophisticated bidirectional agent chaining system that leverages **LangGraph** for orchestration and **Google's Gemini API** for intelligent, collaborative problem-solving through specialized AI agents focused on **Hedera Hashgraph blockchain development and ecosystem support**.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.74+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Google GenAI](https://img.shields.io/badge/Google%20GenAI-0.7.0+-blue.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **🤖 Production-Ready**: Enterprise-grade bidirectional agent chaining with real-time collaboration, RAG capabilities, and comprehensive monitoring.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+ 
- Google AI API key
- Terminal/Command prompt access

### 1. Environment Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

#### Environment Configuration
Create a `.env` file in the project root:
```bash
# Core AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Agent Model Configuration (Optional - defaults provided)
SUPPORT_AGENT_MODEL=gemini-2.0-flash-exp
TECHNICAL_AGENT_MODEL=gemini-2.0-flash-exp
PRODUCT_AGENT_MODEL=gemini-2.0-flash-exp
HEDERA_RAG_AGENT_MODEL=gemini-2.0-flash-exp

# System Configuration
MAX_CHAIN_HOPS=10
CONFIDENCE_THRESHOLD=0.7
AGENT_TIMEOUT=30
MAX_TOKENS_PER_RESPONSE=2048

# Server Configuration
PORT=8000
HOST=0.0.0.0
```

### 2. Hedera RAG Knowledge Base Setup

#### Create Embeddings (First Time Setup)
```bash
# Create the Hedera knowledge base with embeddings
python create_hedera_embeddings.py

# This will:
# - Process Hedera CSV data files
# - Generate 3072-dimensional embeddings using Gemini
# - Create FAISS vector index (if available)
# - Package knowledge base for production use
```

**Expected Output:**
```
🌐 Starting Hedera Knowledge Base Creation...
📊 Processing Hedera data files...
🔮 Generating embeddings (this may take a few minutes)...
💾 Creating knowledge base with 28 knowledge chunks...
✅ Knowledge base created successfully!
   - Embeddings: 28 chunks
   - Index size: 3072 dimensions
   - Location: agents/hedera_rag_kb/
```

### 3. Server Management

#### Start the Application Server
```bash
# Start the FastAPI server
python main.py

# Alternative with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Server Status:**
```
🏗️ Initializing Modern Bidirectional Agent Chain with LangGraph...
✅ Modern Bidirectional Agent Chain initialized successfully!
   - Confidence threshold: 0.7
   - Max chain hops: 10
   - Agent timeout: 30s
   - Dynamic routing: Enabled

🔗 Agent Network Status:
  ✅ Support: healthy (3 connections)
  ✅ Technical: healthy (3 connections)  
  ✅ Product: healthy (3 connections)
  ✅ Hedera_RAG: healthy (3 connections)

🌟 Server running on http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
```

#### Clear All Servers
```bash
# Kill all running Python servers
pkill -f "python main.py"
pkill -f "uvicorn"

# Or find and kill specific processes
lsof -ti:8000 | xargs kill -9

# Verify servers are stopped
curl -f http://localhost:8000/health || echo "✅ Server stopped"
```

### 4. Terminal Client Usage

#### Start Interactive Terminal Client
```bash
# Launch the interactive terminal client
python terminal_client.py
```

#### Single Query Example
```bash
# Example single query interaction:
🤖 Bidirectional Agent Chain Terminal Client
🔗 Server Status: ✅ healthy
🎯 Enter your query (or 'help' for commands): 

> What is Hedera Consensus Service and how does it work?

🔄 Initiating Bidirectional Agent Chain...
📝 Query: What is Hedera Consensus Service and how does it work?
🎯 Priority: medium
⠋ Processing bidirectional agent chain...

🌐 RESPONSE GENERATED
🎯 Confidence: 0.89 | ⏱️ Time: 2.8s
🤖 Agents: Support → Hedera_RAG → Support

📋 Response:
Hedera Consensus Service (HCS) is a decentralized messaging service that 
provides immutable ordering and timestamping of messages. It allows 
applications to achieve consensus on the order of events without needing 
to store the actual data on-ledger, making it cost-effective and scalable 
for high-throughput applications.
```

#### Continuous Chat Experience
```bash
# Start continuous chat mode
> chat

🎭 Continuous Chat Mode Activated
💡 Type your questions, 'back' to return to menu, 'quit' to exit

Chat> Help me troubleshoot a database connection error

🔄 Processing: Support → Technical → Support
🎯 Confidence: 0.82 | Agents: 3 | Time: 4.1s

📋 Technical Support Response:
I'll help you troubleshoot the database connection error. Let me analyze 
the most common causes and provide systematic solutions...

Chat> How do I create a token using Hedera Token Service?

🔄 Processing: Support → Hedera_RAG → Support  
🎯 Confidence: 0.92 | Agents: 2 | Time: 2.6s

📋 Hedera Information:
To create a token using Hedera Token Service (HTS), you'll need to use 
the TokenCreateTransaction with the appropriate SDK...

Chat> back

🎯 Returned to main menu
```

### 5. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Direct API Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I optimize application performance?",
    "user_id": "api_user_001",
    "priority": "high"
  }'
```

#### System Metrics
```bash
curl http://localhost:8000/metrics
```

---

## 📋 Comprehensive Project Overview

### 🎯 System Architecture

This project implements a cutting-edge **bidirectional agent chaining system** that revolutionizes multi-agent AI collaboration. Unlike traditional sequential agent workflows, our system enables true **bidirectional communication** where agents can collaborate, iterate, and refine solutions until optimal results are achieved.

### 🌟 Core Innovation: Bidirectional Agent Chaining

#### Traditional Agent Chains vs. Our Approach

**Traditional Sequential Chaining:**
```
User → Agent A → Agent B → Agent C → Response
```

**Our Bidirectional Chaining:**
```
User Query → Support Agent ┬→ Technical Agent ↘
             (coordinator)  │  (diagnostics)   ↘
                           ↓                   ↘
                         Product Agent → MotoGP RAG Agent
                         (features)      (racing data)
                           ↓                   ↙
                         Support Agent ←←←←←←←↙
                         (final synthesis)
```

#### Key Advantages:
- **Iterative Refinement**: Agents collaborate until confidence threshold (0.7) is reached
- **Dynamic Routing**: Intelligent agent selection based on content analysis
- **Knowledge Synthesis**: Expert insights combined into comprehensive responses
- **Real-time Adaptation**: System adjusts based on query complexity and context

### 🏗️ System Components

#### 1. **Multi-Agent Architecture** (5 Specialized Agents)

##### 🎭 **Support Agent** - Customer Communication Orchestrator
- **Primary Role**: Customer-facing coordinator and conversation manager
- **Capabilities**: Query analysis, agent orchestration, response synthesis
- **Communication Style**: Empathetic, professional, jargon-free
- **Collaboration Pattern**: Coordinates with all agents, translates technical solutions

##### 🔧 **Technical Agent** - System Diagnostics Specialist  
- **Primary Role**: Deep technical analysis and troubleshooting
- **Capabilities**: System diagnostics, error analysis, performance optimization
- **Expertise**: Infrastructure, architecture, security assessment
- **Collaboration Pattern**: Provides technical depth to Support and Product agents

##### 📦 **Product Agent** - Feature and Configuration Expert
- **Primary Role**: Comprehensive product knowledge and feature expertise
- **Capabilities**: Feature analysis, configuration guidance, compatibility assessment
- **Expertise**: Product roadmap, licensing, integration requirements
- **Collaboration Pattern**: Collaborates with Technical for implementation details

##### 🌐 **Hedera RAG Agent** - Blockchain Knowledge Specialist
- **Primary Role**: Advanced RAG-powered Hedera blockchain expertise
- **Capabilities**: 28 knowledge chunks, vector search, 3072-dimensional embeddings
- **Knowledge Base**: Developer guides, services documentation, API examples, use cases, tutorials, SDK information
- **Collaboration Pattern**: Provides blockchain expertise to all agents when needed

##### 🧠 **Base Agent** - Foundation Architecture
- **Primary Role**: Abstract base class defining agent contract
- **Capabilities**: Bidirectional communication, GenAI integration, handoff orchestration
- **Features**: Performance monitoring, health checks, response synthesis

#### 2. **Orchestration Engine** - Dynamic Workflow Management

##### **Chain Orchestrator**
```python
class ChainOrchestrator:
    """
    Dynamic Bidirectional Agent Chain Orchestrator.
    Implements true bidirectional agent chaining with confidence-based iteration.
    """
```

**Key Features:**
- **Confidence-Based Iteration**: Continues until 0.7 threshold reached
- **Dynamic Agent Routing**: Intelligent selection based on content analysis  
- **Performance Monitoring**: Real-time metrics and health tracking
- **Conversation Flow Management**: Complete audit trail of interactions
- **Handoff Orchestration**: Seamless agent-to-agent collaboration

**Network Topology:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Support Agent  │ ←→  │ Technical Agent │ ←→  │ Hedera RAG      │
│   (Customer     │     │  (Diagnostics   │     │ Agent           │
│ Communication)  │     │ & Solutions)    │     │ (Blockchain)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────┬───────────┴───────────┬───────────┘
                     │                       │
                     ↓                       ↓
              ┌─────────────────────────────────────┐
              │           Product Agent             │
              │    (Features & Knowledge)           │
              └─────────────────────────────────────┘
```

#### 3. **RAG Infrastructure** - Advanced Knowledge Retrieval

##### **Hedera Knowledge Base**
- **Embeddings**: 3072-dimensional vectors (Gemini embedding model)
- **Vector Search**: FAISS CPU for efficient similarity search (with fallback to text search)
- **Knowledge Chunks**: 28 structured data points
- **Data Sources**: Developer guides, services documentation, API examples, use cases, tutorials, SDK documentation
- **Processing**: Automatic file type detection, flexible column mapping

##### **RAG Pipeline:**
1. **Query Analysis**: Semantic understanding of user questions
2. **Vector Search**: FAISS similarity search across knowledge base
3. **Context Retrieval**: Relevant chunks with confidence scoring
4. **Response Generation**: Gemini model with retrieved context
5. **Quality Assurance**: Confidence validation and source attribution

#### 4. **Production API** - Enterprise-Ready Interface

##### **FastAPI REST API**
```python
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process user request through bidirectional agent chain"""
```

**Endpoints:**
- `POST /query` - Process queries through agent chain
- `GET /health` - System and agent health status
- `GET /metrics` - Performance analytics and statistics
- `GET /agents` - Agent capabilities and network status
- `GET /docs` - Interactive API documentation

**Features:**
- **Async Processing**: Non-blocking operations throughout
- **Input Validation**: Comprehensive Pydantic models
- **Error Handling**: Robust error recovery and timeout management
- **CORS Support**: Cross-origin resource sharing enabled
- **Real-time Monitoring**: Health checks and performance metrics

### 🔬 Technical Specifications

#### **AI/ML Stack**
| Component | Technology | Version |
|-----------|------------|---------|
| **Language Models** | Google Gemini 2.0 Flash | Cheap |
| **Embeddings** | Gemini Embedding 001 | 3072-dim |
| **Orchestration** | LangGraph | 0.2.74+ |
| **Vector Search** | FAISS CPU | 1.7.4+ |
| **State Management** | LangGraph StateGraph | Latest |

#### **Backend Infrastructure**
| Component | Technology | Version |
|-----------|------------|---------|
| **Web Framework** | FastAPI | 0.104+ |
| **ASGI Server** | Uvicorn | 0.24+ |
| **Data Validation** | Pydantic | 2.5+ |
| **HTTP Client** | aiohttp | 3.9+ |
| **Testing** | pytest + pytest-asyncio | 7.4+ |

#### **Performance Metrics**
| Metric | Value | Description |
|--------|-------|-------------|
| **Response Time** | < 5s average | Full agent chain processing |
| **Confidence Threshold** | 0.7 minimum | Quality assurance metric |
| **Max Chain Hops** | 10 maximum | Prevents infinite loops |
| **Agent Timeout** | 30s per agent | Individual agent timeout |
| **Concurrent Users** | 100+ supported | Async processing capacity |

### 🎯 Advanced Features

#### **1. Confidence-Based Processing**
```python
while (conversation_state["hop_count"] < max_chain_hops and 
       conversation_state["current_confidence"] < confidence_threshold):
    # Continue processing until optimal solution found
```

#### **2. Dynamic Agent Routing**
```python
routing_patterns = {
    "technical": ["error", "bug", "crash", "performance", "system"],
    "product": ["feature", "pricing", "plan", "upgrade", "license"],
    "support": ["help", "question", "how to", "tutorial", "guide"],
    "hedera": ["hedera", "blockchain", "hashgraph", "hcs", "hfs", "hts", "smart contracts"]
}
```

#### **3. Real-Time Flow Tracking**
```
🔄 Dynamic Bidirectional Agent Chain Flow:
==================================================
  Step 1: Support Agent (query analysis)
    📝 Action: processed | 🎯 Confidence: 0.65
    
  Step 2: Technical Agent (handoff)  
    📝 Action: processed | 🎯 Confidence: 0.85
    🔀 HANDOFF: Support → Hedera_RAG
    
📊 Communication Flow: Support → Hedera_RAG → Support
🧠 Flow Type: bidirectional, handoff-enabled agent chaining
```

#### **4. Health Monitoring System**
```python
def check_agent_health(self) -> Dict[str, Any]:
    """Comprehensive agent network health check"""
    return {
        "network_topology": "fully_connected",
        "agents": {
            "Support": {"status": "healthy", "connections": 3},
            "Technical": {"status": "healthy", "connections": 3},
            "Product": {"status": "healthy", "connections": 3},
            "Hedera_RAG": {"status": "healthy", "connections": 3}
        }
    }
```

### 🔧 Advanced Configuration

#### **Environment Variables**
```bash
# AI Model Configuration
GOOGLE_API_KEY=your_api_key
SUPPORT_AGENT_MODEL=gemini-2.0-flash-exp
TECHNICAL_AGENT_MODEL=gemini-2.0-flash-exp
PRODUCT_AGENT_MODEL=gemini-2.0-flash-exp
HEDERA_RAG_AGENT_MODEL=gemini-2.0-flash-exp

# Agent Behavior Tuning
SUPPORT_AGENT_TEMPERATURE=0.7
TECHNICAL_AGENT_TEMPERATURE=0.5
PRODUCT_AGENT_TEMPERATURE=0.6
MOTOGP_RAG_AGENT_TEMPERATURE=0.4

# System Performance
MAX_CHAIN_HOPS=10
CONFIDENCE_THRESHOLD=0.7
AGENT_TIMEOUT=30
MAX_TOKENS_PER_RESPONSE=2048

# RAG Configuration
HEDERA_KB_PATH=agents/hedera_rag_kb
EMBEDDING_DIMENSIONS=3072
FAISS_INDEX_TYPE=IndexFlatIP

# Server Configuration
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=["*"]
LOG_LEVEL=INFO
```

#### **Agent Customization**
```python
# Custom agent creation example
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Custom",
            expertise="specialized_domain",
            system_prompt="Your specialized prompt here...",
            temperature=0.8,
            max_tokens=1024
        )
```

---

## 🧪 Testing & Quality Assurance

### **Comprehensive Test Suite**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test categories
pytest tests/agents/          # Agent-specific tests
pytest tests/chain/           # Orchestration tests  
pytest tests/models/          # Data model tests
pytest tests/test_integration.py  # End-to-end tests
```

### **Test Coverage**
| Module | Coverage | Test Files |
|--------|----------|------------|
| **Agents** | ~60% | 4 test files |
| **Chain** | ~55% | 1 test file |
| **Models** | ~70% | 1 test file |
| **Integration** | ~45% | 1 test file |
| **Overall** | ~53% | 10 test files |

### **Quality Metrics**
```bash
# Code quality analysis
flake8 .                    # Style checking
black .                     # Code formatting
mypy .                      # Type checking
bandit -r .                 # Security analysis
```

---

## 📊 Project Statistics

### **Codebase Metrics**
| Metric | Value | Description |
|--------|-------|-------------|
| **Total Files** | 26 | Python source files |
| **Lines of Code** | 9,715+ | Comprehensive implementation |
| **Classes** | 123+ | Object-oriented architecture |
| **Functions/Methods** | 39+ | Modular design |
| **Test Files** | 10 | Quality assurance |
| **Dependencies** | 20+ | Modern tech stack |

### **Agent Network Analysis**
```
🔗 Agent Network Topology:
┌─────────────────────────────────────────────────────────────┐
│                    Full Mesh Network                        │
│  Each agent maintains bidirectional connections to all     │
│  other agents, enabling seamless collaboration             │
├─────────────────────────────────────────────────────────────┤
│  Support Agent      → [Technical, Product, Hedera_RAG]     │
│  Technical Agent    → [Support, Product, Hedera_RAG]       │
│  Product Agent      → [Support, Technical, Hedera_RAG]     │
│  Hedera_RAG Agent   → [Support, Technical, Product]        │
└─────────────────────────────────────────────────────────────┘

📈 Communication Patterns:
  - Total Possible Connections: 12 (4×3)
  - Network Density: 100% (fully connected)
  - Average Path Length: 1 (direct connections)
  - Collaboration Efficiency: Maximum
```

---

## 🚀 Advanced Usage Examples

### **Complex Multi-Agent Collaboration**
```python
# Example: Technical issue requiring multiple expert opinions
query = """
Our application is experiencing intermittent performance issues. 
Users report slow response times during peak hours, and we're seeing 
database connection timeouts. We're on the Enterprise plan and need 
to maintain 99.9% uptime. What's the best approach to diagnose and 
resolve this issue?
"""

# Expected agent flow:
# 1. Support Agent (query analysis) → Confidence: 0.4
# 2. Technical Agent (system diagnostics) → Confidence: 0.6  
# 3. Product Agent (plan features/limits) → Confidence: 0.8
# 4. Support Agent (final synthesis) → Confidence: 0.9
```

### **RAG-Powered Knowledge Retrieval**
```python
# Example: Hedera-specific query
query = "How do I implement smart contracts on Hedera and what are the gas fees?"

# Expected processing:
# 1. Support Agent identifies blockchain query → Hedera_RAG handoff
# 2. Hedera_RAG Agent performs vector search across 28 knowledge chunks
# 3. Retrieves relevant data: smart contract documentation, fee structure
# 4. Generates comprehensive implementation guide with cost analysis
# 5. Support Agent synthesizes into user-friendly response
```

### **API Integration Example**
```python
import asyncio
import aiohttp

async def query_agent_chain():
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": "How do I optimize my application's database performance?",
            "user_id": "api_user_001",
            "priority": "high",
            "expected_response_format": "structured"
        }
        
        async with session.post(
            "http://localhost:8000/query", 
            json=payload
        ) as response:
            result = await response.json()
            print(f"Confidence: {result['confidence_score']}")
            print(f"Response: {result['response']}")
            print(f"Agents: {result['agents_involved']}")

# Run the example
asyncio.run(query_agent_chain())
```

---

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **1. Server Won't Start**
```bash
# Check if port is in use
lsof -i :8000

# Kill existing processes
pkill -f "python main.py"

# Verify Google API key
echo $GOOGLE_API_KEY
```

#### **2. MotoGP RAG Agent Errors**
```bash
# Recreate knowledge base
python create_hedera_embeddings.py

# Verify knowledge base files
ls -la agents/hedera_rag_kb/
```

#### **3. Agent Communication Issues**
```bash
# Check agent health
curl http://localhost:8000/health

# Verify agent network status
curl http://localhost:8000/agents
```

#### **4. Performance Issues**
```bash
# Check system metrics
curl http://localhost:8000/metrics

# Monitor agent response times
python terminal_client.py --verbose
```

### **Debug Mode**
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python main.py

# Run with profiling
python -m cProfile main.py
```

---

## 📚 Documentation & Resources

### **Architecture Documentation**
- **Agent Guides**: `/markdown_guides/agents/`
- **Orchestration**: `/markdown_guides/chain/`
- **Data Models**: `/markdown_guides/models/`
- **Setup Guides**: `/markdown_guides/`

### **Example Projects**
```bash
# Clone example implementations
git clone https://github.com/your-org/agent-examples.git
cd agent-examples/

# Run example scenarios
python examples/customer_support_scenario.py
python examples/technical_troubleshooting.py
python examples/hedera_blockchain_analysis.py
```

---

## 🤝 Contributing

### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/your-username/bidirectional_agent_chaining.git
cd bidirectional_agent_chaining

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run development server
python main.py --reload --debug
```

### **Code Style Guidelines**
```bash
# Format code
black .
isort .

# Lint code
flake8 .
pylint agents/ chain/ models/

# Type checking
mypy .
```

### **Contribution Workflow**
1. Create feature branch: `git checkout -b feature/amazing-enhancement`
2. Make changes and add tests
3. Run test suite: `pytest`
4. Commit with conventional commits: `feat: add amazing enhancement`
5. Push and create pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


<div align="center">

**🌟 Built with ❤️ Zubeid Hendricks**

[⭐ Star this repo](https://github.com/ZubeidHendricks/hedera-bidirectional-agent-chain-) | [📖 Read the docs](markdown_guides)

</div>

