# BaseAgent - Core Agent Architecture Guide

## Overview
The `BaseAgent` class serves as the foundational abstract base class for all agents in the bidirectional agent chaining system. It implements the core functionality for agent communication, Google GenAI integration, and the sophisticated handoff mechanism that enables seamless collaboration between specialized agents.

## File Purpose
This base class provides:
- **Abstract Interface**: Defines the contract all agents must implement
- **Bidirectional Communication**: Enables agents to communicate and collaborate
- **GenAI Integration**: Handles Google Gemini AI model interactions
- **Handoff Orchestration**: Manages intelligent agent-to-agent handoffs
- **Response Synthesis**: Combines expert knowledge from multiple agents
- **Performance Monitoring**: Tracks agent performance and health

## Class Architecture

### Core Dependencies
```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Literal
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END
from models.message import Message, Response
```

### Class Definition
```python
class BaseAgent(ABC):
    """
    Enhanced BaseAgent using latest Google GenAI SDK and LangGraph patterns.
    Implements bidirectional agent communication with modern AI orchestration.
    """
```

## Constructor Analysis

### Initialization Parameters
```python
def __init__(
    self, 
    name: str, 
    expertise: str, 
    system_prompt: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048
):
```

**Parameters Explained**:
- `name`: Unique identifier for the agent
- `expertise`: Domain expertise description
- `system_prompt`: Core instruction set defining agent behavior
- `model_name`: Google Gemini model to use (defaults to environment variable)
- `temperature`: Model creativity level (0.0-1.0)
- `max_tokens`: Maximum response length

### Core Attributes
```python
self.name = name
self.expertise = expertise
self.system_prompt = system_prompt
self.conversation_history: List[Message] = []
self.capabilities = [expertise]
self.agent_registry: Dict[str, 'BaseAgent'] = {}
```

## Google GenAI Integration

### Client Initialization
```python
def _initialize_genai_client(self):
    """Initialize Google GenAI client with environment configuration"""
    try:
        use_vertex_ai = os.getenv("USE_VERTEX_AI", "false").lower() == "true"
        
        if use_vertex_ai:
            # Vertex AI configuration
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
            # Developer API configuration
            api_key = os.getenv("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Error initializing GenAI client: {e}")
        raise
```

**Configuration Options**:
- **Developer API**: Uses `GOOGLE_API_KEY` for direct API access
- **Vertex AI**: Uses `GOOGLE_CLOUD_PROJECT_ID` and `GOOGLE_CLOUD_LOCATION` for enterprise deployment

## Message Processing Architecture

### Main Processing Method
```python
async def process_message(self, message: Message) -> Response:
    """Process incoming message and return response"""
    print(f"\n🔄 {self.name} processing message from {message.source_agent}")
    
    self.conversation_history.append(message)
    
    try:
        response = await self._generate_response(message)
        return response
    except Exception as e:
        return Response(
            content=f"Error: {str(e)}",
            source_agent=self.name,
            target_agent=message.source_agent,
            confidence=0.1
        )
```

### Response Generation
```python
async def _generate_response(self, message: Message) -> Response:
    """Generate response using latest Google GenAI SDK"""
    context = self._build_context()
    prompt = f"""
{self.system_prompt}

AGENT IDENTITY:
- Name: {self.name}
- Expertise: {self.expertise}
- Available Capabilities: {', '.join(self.capabilities)}

CONVERSATION CONTEXT:
{context}

CURRENT MESSAGE FROM {message.source_agent.upper()}:
{message.content}

BIDIRECTIONAL COMMUNICATION INSTRUCTIONS:
- You can collaborate with other agents by requesting handoffs
- If you need specific expertise, use: "HANDOFF_REQUEST: [agent_name] - [specific question]"
- Provide complete answers when you have sufficient information
- Be concise, professional, and accurate

RESPONSE:
"""
```

**Key Features**:
- Dynamic context building from conversation history
- Agent identity and capability awareness
- Structured handoff request mechanism
- Professional communication standards

## Bidirectional Communication System

### Agent Registration
```python
def register_agent(self, agent_name: str, agent_instance: 'BaseAgent'):
    """Register another agent for bidirectional communication"""
    self.agent_registry[agent_name] = agent_instance
    print(f"🔗 {self.name} Agent registered connection to {agent_name}")
```

### Handoff Request Parsing
```python
def _parse_handoff_request(self, content: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Parse content for handoff requests"""
    if "HANDOFF_REQUEST:" not in content:
        return False, None, None
    
    try:
        handoff_part = content.split("HANDOFF_REQUEST:")[1].strip()
        if " - " in handoff_part:
            target_agent, query = handoff_part.split(" - ", 1)
            return True, target_agent.strip(), query.strip()
    except Exception:
        pass
    
    return False, None, None
```

### Handoff Execution
```python
async def _handle_handoff(self, target_agent: str, query: str, original_message: Message) -> Optional[Response]:
    """Handle handoff to another agent"""
    if target_agent not in self.agent_registry:
        return None
    
    handoff_message = Message(
        content=query,
        source_agent=self.name,
        target_agent=target_agent,
        message_type="handoff",
        context={
            "original_query": original_message.content,
            "original_source": original_message.source_agent,
            "handoff_reason": f"Expertise needed in {self.agent_registry[target_agent].expertise}"
        }
    )
    
    response = await self.agent_registry[target_agent].process_message(handoff_message)
    return response
```

## Response Synthesis

### Multi-Agent Response Synthesis
```python
async def _synthesize_response(
    self, 
    initial_content: str, 
    handoff_response: Response, 
    original_message: Message
) -> Response:
    """Synthesize final response incorporating handoff results"""
    synthesis_prompt = f"""
SYNTHESIS TASK:
Combine your initial analysis with expert input to provide a comprehensive response.

YOUR INITIAL ANALYSIS:
{initial_content.split('HANDOFF_REQUEST:')[0].strip()}

EXPERT INPUT FROM {handoff_response.source_agent}:
{handoff_response.content}

ORIGINAL USER QUERY:
{original_message.content}

SYNTHESIZED RESPONSE:
"""
    
    # Generate synthesized response using GenAI
    response = await self.client.aio.models.generate_content(
        model=self.model_name,
        contents=synthesis_prompt,
        config=generation_config
    )
    
    return Response(
        content=response.text.strip(),
        source_agent=self.name,
        target_agent=original_message.source_agent,
        confidence=calculated_confidence,
        reasoning="Synthesized response combining multiple agent expertise"
    )
```

## Confidence Calculation

### Dynamic Confidence Scoring
```python
def _calculate_confidence(self, content: str) -> float:
    """Calculate confidence score based on response characteristics"""
    base_confidence = 0.8
    
    # Adjust based on content characteristics
    if len(content) < 50:
        base_confidence -= 0.2
    if "I don't know" in content.lower() or "uncertain" in content.lower():
        base_confidence -= 0.3
    if "error" in content.lower():
        base_confidence -= 0.2
    if len(content.split('.')) > 3:  # Well-structured response
        base_confidence += 0.1
    
    return max(0.1, min(1.0, base_confidence))
```

## Context Management

### Conversation Context Building
```python
def _build_context(self) -> str:
    """Build context from conversation history"""
    if not self.conversation_history:
        return "No prior conversation in this session."
    
    context_parts = []
    recent_messages = self.conversation_history[-8:]  # Last 8 messages
    
    for msg in recent_messages:
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        context_parts.append(f"[{timestamp}] {msg.source_agent} → {msg.target_agent}: {msg.content}")
    
    return "\n".join(context_parts)
```

## Health Monitoring

### Agent Health Check
```python
async def health_check(self) -> Dict[str, Any]:
    """Perform health check on the agent"""
    try:
        test_response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents="Say 'healthy' if you can respond.",
            config=types.GenerateContentConfig(max_output_tokens=10)
        )
        
        return {
            "status": "healthy",
            "model_responsive": "healthy" in test_response.text.lower(),
            "client_initialized": True,
            "connections": len(self.agent_registry)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_responsive": False,
            "connections": len(self.agent_registry)
        }
```

## Abstract Methods

### Required Implementation
```python
@abstractmethod
async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
    """Query another agent for information - implemented by concrete agents"""
    pass
```

## Usage Examples

### Basic Agent Implementation
```python
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Custom",
            expertise="custom_domain",
            system_prompt="You are a custom domain expert..."
        )
    
    async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
        if target_agent not in self.agent_registry:
            return None
        
        message = Message(
            content=query,
            source_agent=self.name,
            target_agent=target_agent,
            message_type="query"
        )
        
        return await self.agent_registry[target_agent].process_message(message)
```

### Multi-Agent Setup
```python
# Initialize agents
support = SupportAgent()
technical = TechnicalAgent()
product = ProductAgent()

# Setup bidirectional communication
support.register_agent("Technical", technical)
support.register_agent("Product", product)
technical.register_agent("Support", support)
technical.register_agent("Product", product)
product.register_agent("Support", support)
product.register_agent("Technical", technical)

# Process user message
user_message = Message(
    content="My app is crashing during file uploads",
    source_agent="user",
    target_agent="Support",
    message_type="query"
)

response = await support.process_message(user_message)
```

## Design Patterns

### 1. Abstract Factory Pattern
`BaseAgent` defines the interface that all concrete agents must implement.

### 2. Strategy Pattern
Different agents implement different strategies for handling their domain expertise.

### 3. Observer Pattern
Agents can observe and respond to messages from other agents.

### 4. Chain of Responsibility
Handoff mechanism creates a chain where requests can be passed between agents.

## Integration Points

### Message System Integration
- Uses standardized `Message` and `Response` models
- Integrates with LangGraph for workflow orchestration
- Supports various message types (query, handoff, analysis, etc.)

### Orchestrator Integration
- Provides health check endpoints for monitoring
- Exposes agent information for dynamic routing
- Supports performance metrics collection

### Environment Configuration
- Flexible AI model selection via environment variables
- Support for both Gemini Developer API and Vertex AI
- Configurable temperature and token limits

## Best Practices

### 1. Agent Design
- Keep system prompts focused and clear
- Define specific expertise domains
- Implement proper error handling
- Use appropriate confidence scoring

### 2. Communication
- Use structured handoff requests
- Maintain conversation context
- Synthesize responses from multiple sources
- Provide clear reasoning for decisions

### 3. Performance
- Monitor response times and confidence scores
- Implement proper timeout handling
- Cache frequently used information
- Use async/await for non-blocking operations

## Troubleshooting

### Common Issues

1. **GenAI Client Initialization Failures**
   ```python
   # Check environment variables
   GOOGLE_API_KEY=your_api_key
   # Or for Vertex AI
   USE_VERTEX_AI=true
   GOOGLE_CLOUD_PROJECT_ID=your_project
   ```

2. **Handoff Parsing Issues**
   ```python
   # Ensure proper format
   "HANDOFF_REQUEST: TechnicalAgent - Analyze this database error"
   ```

3. **Circular Dependencies**
   ```python
   # Avoid direct agent-to-agent calls in constructors
   # Use post-initialization registration
   ```

## Extension Points

### Custom Capability Implementation
```python
class SpecializedAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Specialized", ...)
        self.capabilities.extend([
            "domain_specific_skill",
            "advanced_analysis",
            "custom_integration"
        ])
    
    def can_handle(self, message: Message) -> bool:
        """Custom logic for determining if agent can handle message"""
        return "specialized_keyword" in message.content.lower()
```

### Custom Synthesis Logic
```python
async def _custom_synthesize_response(self, responses: List[Response]) -> Response:
    """Custom synthesis logic for multiple agent responses"""
    # Implement domain-specific synthesis
    # Weight responses based on confidence and relevance
    # Apply custom validation rules
```

## Testing Considerations

### Unit Tests
```python
@pytest.mark.asyncio
async def test_base_agent_message_processing():
    """Test basic message processing functionality"""
    agent = TestAgent()  # Concrete implementation
    
    message = Message(
        content="Test query",
        source_agent="user",
        target_agent="TestAgent"
    )
    
    response = await agent.process_message(message)
    
    assert response.source_agent == "TestAgent"
    assert response.confidence > 0
    assert len(response.content) > 0
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_agent_handoff_mechanism():
    """Test agent-to-agent handoff functionality"""
    agent1 = TestAgent1()
    agent2 = TestAgent2()
    
    agent1.register_agent("TestAgent2", agent2)
    
    message = Message(
        content="HANDOFF_REQUEST: TestAgent2 - Handle this specialized task",
        source_agent="user",
        target_agent="TestAgent1"
    )
    
    response = await agent1.process_message(message)
    
    assert "TestAgent2" in response.reasoning
    assert response.confidence > 0.5
```

## Performance Considerations

### Memory Management
- Limit conversation history length
- Clean up old agent registrations
- Use memory-efficient context building

### Response Time Optimization
- Implement async processing throughout
- Use connection pooling for API calls
- Cache common responses when appropriate

### Error Resilience
- Implement retry logic for API failures
- Graceful degradation when agents are unavailable
- Proper exception handling and logging

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