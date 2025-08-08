# Requirements - Project Dependencies Guide

## Overview
The `requirements.txt` file defines all the Python dependencies required for the bidirectional agent chaining system. It includes core frameworks, AI/ML libraries, RAG components, and development tools necessary for the complete system functionality.

## File Purpose
This requirements file provides:
- **Core Dependencies**: Essential frameworks and libraries for system operation
- **AI/ML Integration**: Google GenAI, LangGraph, and LangChain components
- **RAG Infrastructure**: Vector search, embeddings, and knowledge base tools
- **Development Tools**: Testing, utilities, and development support
- **Version Management**: Specific version requirements for stability

## Dependency Categories

### Core System Dependencies
```python
# Web Framework and API
fastapi>=0.104.0          # Modern, fast web framework for building APIs
uvicorn>=0.24.0           # ASGI server for FastAPI applications
pydantic>=2.5.0           # Data validation and settings management
python-multipart>=0.0.6   # Form and file upload handling

# Environment and Configuration
python-dotenv>=1.0.0      # Environment variable management
```

**Purpose**: Foundation components for the web API, request/response handling, and configuration management.

### AI and Language Model Integration
```python
# Google AI Integration
google-genai>=0.7.0       # Google Gemini AI models and embeddings

# LangGraph and LangChain
langgraph>=0.2.74         # Workflow orchestration and state management
langchain-core>=0.3.0     # Core LangChain components
langgraph-checkpoint>=2.0.0  # State persistence and checkpointing

# HTTP Clients
aiohttp>=3.9.0           # Async HTTP client for AI API calls
httpx[socks]>=0.25.0     # Modern HTTP client with proxy support
```

**Purpose**: AI model integration, workflow orchestration, and robust HTTP communication with external AI services.

### RAG and Vector Search Components
```python
# Vector Database and Search
faiss-cpu>=1.7.4         # Facebook AI Similarity Search - CPU version
numpy>=1.24.0            # Numerical computing for embeddings
pandas>=2.0.0            # Data manipulation and analysis
sentence-transformers>=2.2.0  # Additional embedding model support
```

**Purpose**: Implements the RAG (Retrieval-Augmented Generation) infrastructure with efficient vector similarity search capabilities.

### Development and Testing Tools
```python
# Testing Framework
pytest>=7.4.0           # Testing framework
pytest-asyncio>=0.21.0  # Async testing support

# Utilities
pathlib2>=2.3.7         # Enhanced path handling
```

**Purpose**: Comprehensive testing capabilities and utility functions for development.

## Detailed Dependency Analysis

### FastAPI Ecosystem
```python
fastapi>=0.104.0         # Web framework
uvicorn>=0.24.0          # ASGI server  
pydantic>=2.5.0          # Data validation
python-multipart>=0.0.6  # File uploads
```

**Integration**: Provides the complete web API infrastructure for the bidirectional agent chaining system, handling HTTP requests, response validation, and file operations.

### Google AI Integration
```python
google-genai>=0.7.0      # Latest Google GenAI SDK
```

**Features**:
- Gemini model access (gemini-2.0-flash-exp)
- Embedding generation (gemini-embedding-001)
- Both Developer API and Vertex AI support
- Async/await compatibility

### LangGraph Workflow System
```python
langgraph>=0.2.74            # Core workflow engine
langchain-core>=0.3.0        # Message and state management
langgraph-checkpoint>=2.0.0  # State persistence
```

**Capabilities**:
- State-based workflow orchestration
- Message passing between agents
- Conversation flow management
- Checkpoint and resume functionality

### FAISS Vector Database
```python
faiss-cpu>=1.7.4         # CPU-optimized version
numpy>=1.24.0            # Required for vector operations
```

**Performance**: 
- CPU-optimized for deployment flexibility
- Supports millions of vectors
- Sub-millisecond search times
- Memory-efficient indexing

## Installation and Setup

### Basic Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Or with virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Development Installation
```bash
# Install with development extras
pip install -r requirements.txt
pip install -e .  # If setup.py exists

# Install additional development tools
pip install black flake8 mypy  # Code formatting and linting
```

### Production Deployment
```bash
# Use specific versions for production stability
pip install --no-deps -r requirements.txt

# Verify installation
python -c "import fastapi, google.genai, faiss; print('All core dependencies loaded')"
```

## Environment Configuration

### Required Environment Variables
```bash
# Google AI Configuration (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Vertex AI Configuration
USE_VERTEX_AI=false
GOOGLE_CLOUD_PROJECT_ID=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1

# System Configuration (Optional)
CONFIDENCE_THRESHOLD=0.7
MAX_CHAIN_HOPS=30
AGENT_RESPONSE_TIMEOUT=30
```

### Development Environment
```bash
# Development settings
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_DETAILED_FLOW_LOGGING=true
ENABLE_AGENT_COMMUNICATION_LOGS=true
DEBUG=true
```

## Compatibility and Version Requirements

### Python Version
```python
# Minimum Python version
python_requires=">=3.8"

# Recommended Python version
python_version="3.11"  # Optimal performance
```

### Operating System Support
- **Linux**: Full support (Ubuntu 20.04+, CentOS 8+)
- **macOS**: Full support (macOS 11+)
- **Windows**: Full support (Windows 10+)

### Hardware Requirements
```yaml
minimum:
  ram: "4GB"
  cpu: "2 cores"
  storage: "1GB"

recommended:
  ram: "8GB"
  cpu: "4 cores"
  storage: "5GB"
  gpu: "Optional (for FAISS GPU)"
```

## Dependency-Specific Configuration

### FAISS Configuration
```python
# For GPU support (optional)
# faiss-gpu>=1.7.4  # Replace faiss-cpu

# Environment variables
export OMP_NUM_THREADS=4  # Control CPU threading
export FAISS_ENABLE_GPU=false  # CPU-only mode
```

### Google GenAI Configuration
```python
# Client initialization options
google-genai:
  timeout: 30
  retry_count: 3
  max_tokens: 2048
  temperature: 0.7
```

### LangGraph Configuration
```python
# Optional LangGraph settings
langgraph:
  checkpoint_enabled: true
  max_workflow_steps: 100
  state_timeout: 300
```

## Troubleshooting Dependencies

### Common Installation Issues

1. **FAISS Installation Problems**
   ```bash
   # Alternative installation methods
   conda install faiss-cpu -c conda-forge
   # Or build from source for specific optimization
   ```

2. **Google GenAI SDK Issues**
   ```bash
   # Update to latest version
   pip install --upgrade google-genai
   
   # Check API key configuration
   python -c "import os; print('API Key:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET')"
   ```

3. **LangGraph Compatibility**
   ```bash
   # Ensure compatible versions
   pip install "langgraph>=0.2.74" "langchain-core>=0.3.0"
   ```

### Dependency Conflicts
```bash
# Check for conflicts
pip check

# Resolve conflicts
pip install pip-tools
pip-compile requirements.in  # Generate locked versions
```

### Version Pinning for Production
```python
# Create locked requirements
pip freeze > requirements-lock.txt

# Use in production
pip install -r requirements-lock.txt
```

## Performance Optimization

### Memory Usage
```python
# Optimize for memory usage
faiss-cpu==1.7.4  # CPU version uses less memory
numpy>=1.24.0      # Efficient numerical operations
```

### Load Time Optimization
```python
# Lazy loading for optional dependencies
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
```

### Production Optimizations
```bash
# Install with optimizations
pip install --compile --no-cache-dir -r requirements.txt

# Use system packages where available
apt-get install python3-numpy python3-pandas  # Ubuntu/Debian
```

## Security Considerations

### Dependency Security
```bash
# Check for security vulnerabilities
pip install safety
safety check -r requirements.txt

# Update vulnerable packages
pip install --upgrade package_name
```

### Production Security
```bash
# Use hash checking
pip install --require-hashes -r requirements.txt

# Generate hashes
pip-compile --generate-hashes requirements.in
```

## Extension and Customization

### Adding New Dependencies
```python
# Add to requirements.txt with version constraints
new-package>=1.0.0,<2.0.0  # Compatible version range

# For development-only dependencies
# Create requirements-dev.txt
pytest-cov>=4.0.0
black>=23.0.0
mypy>=1.0.0
```

### Optional Dependencies
```python
# Create optional dependency groups
[extras]
gpu = ["faiss-gpu>=1.7.4"]
dev = ["pytest>=7.4.0", "black>=23.0.0"]
monitoring = ["prometheus-client>=0.16.0"]

# Install with extras
pip install -e ".[gpu,dev]"
```

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