# Message Models - Communication Structure Guide

## Overview
The `message.py` file defines the core data models for the bidirectional agent chaining system, providing structured communication between agents, workflow management, and LangGraph integration. These Pydantic models ensure type safety and data validation throughout the system.

## File Purpose
This module provides:
- **Structured Communication**: Type-safe message passing between agents
- **LangGraph Integration**: Compatible with LangGraph workflows and state management
- **Data Validation**: Pydantic-based validation for all data structures
- **Workflow Management**: Models for tracking conversation flows and results
- **Type Safety**: Strong typing for improved development experience

## Core Model Definitions

### Message Model
```python
class Message(BaseModel):
    """Enhanced message structure for inter-agent communication"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str = Field(..., description="The message content")
    source_agent: str = Field(..., description="Agent that sent the message")
    target_agent: str = Field(..., description="Agent that should receive the message")
    message_type: Literal[
        "query", "response", "notification", "command", "handoff",
        "analysis", "consultation", "synthesis", "collaboration",
        "technical_analysis", "product_analysis", "support_analysis"
    ] = "query"
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    requires_response: bool = Field(default=True)
    conversation_id: Optional[str] = Field(default=None)
```

**Key Features**:
- Automatic ID and timestamp generation
- Rich message type system for different communication patterns
- Priority levels for message handling
- Flexible context for additional metadata
- LangGraph conversion capabilities

### Response Model
```python
class Response(BaseModel):
    """Enhanced response structure from agents"""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str = Field(..., description="The response content")
    source_agent: str = Field(..., description="Agent that generated the response")
    target_agent: str = Field(..., description="Agent or user to receive the response")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    needs_clarification: bool = Field(default=False)
    suggested_next_agent: Optional[str] = Field(default=None)
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning process")
    metadata: Optional[Dict[str, Any]] = Field(default=None)
```

**Key Features**:
- Confidence scoring for quality assessment
- Reasoning transparency for debugging
- Suggestion mechanism for agent routing
- Rich metadata support

### UserRequest Model
```python
class UserRequest(BaseModel):
    """Enhanced user request structure"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    query: str = Field(..., description="User's query or request")
    user_id: str = Field(..., description="Unique user identifier")
    context: Optional[Dict[str, Any]] = Field(default=None)
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    expected_response_format: Literal["text", "structured", "json"] = "text"
    max_processing_time: Optional[int] = Field(default=60, description="Max time in seconds")
```

**Key Features**:
- User identification and context tracking
- Processing time limits
- Response format preferences
- Priority-based handling

### ChainResult Model
```python
class ChainResult(BaseModel):
    """Enhanced final result from the agent chain"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str = Field(..., description="Original request ID")
    response: str = Field(..., description="Final response to user")
    agents_involved: List[str] = Field(default_factory=list)
    conversation_flow: List[Dict[str, Any]] = Field(default_factory=list)
    total_processing_time: float = Field(..., description="Total time in seconds")
    success: bool = Field(default=True)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    error_details: Optional[str] = Field(default=None)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
```

**Key Features**:
- Complete conversation tracking
- Performance metrics collection
- Success/failure tracking
- Detailed error information

## LangGraph Integration

### State Management
```python
class MessagesState(TypedDict):
    """LangGraph compatible state that tracks conversation messages"""
    messages: Annotated[List[BaseMessage], add_messages]

class AgentState(TypedDict):
    """Extended state for bidirectional agent chaining"""
    messages: Annotated[List[BaseMessage], add_messages]
    agent_context: Dict[str, Any]
    request_id: str
    user_id: str
    current_agent: str
    next_agent: Optional[str]
    conversation_flow: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]
```

### LangChain Message Conversion
```python
def to_langchain_message(self) -> BaseMessage:
    """Convert to LangChain message format"""
    if self.source_agent == "user":
        return HumanMessage(
            content=self.content,
            additional_kwargs={
                "message_id": self.message_id,
                "source_agent": self.source_agent,
                "target_agent": self.target_agent,
                "context": self.context or {}
            }
        )
    else:
        return AIMessage(
            content=self.content,
            additional_kwargs={
                "message_id": self.message_id,
                "source_agent": self.source_agent,
                "target_agent": self.target_agent,
                "context": self.context or {}
            }
        )
```

## Utility Functions

### Message Conversion
```python
def messages_to_langchain(messages: List[Message]) -> List[BaseMessage]:
    """Convert list of Message objects to LangChain messages"""
    return [msg.to_langchain_message() for msg in messages]

def langchain_to_messages(lc_messages: List[BaseMessage]) -> List[Message]:
    """Convert LangChain messages back to Message objects"""
    messages = []
    for lc_msg in lc_messages:
        if isinstance(lc_msg, HumanMessage):
            source_agent = "user"
        elif isinstance(lc_msg, AIMessage):
            source_agent = lc_msg.additional_kwargs.get("source_agent", "ai")
        else:
            source_agent = "system"
            
        target_agent = lc_msg.additional_kwargs.get("target_agent", "unknown")
        
        message = Message(
            content=lc_msg.content,
            source_agent=source_agent,
            target_agent=target_agent,
            context=lc_msg.additional_kwargs.get("context", {})
        )
        messages.append(message)
    
    return messages
```

### Agent State Creation
```python
def create_agent_state(
    request: UserRequest,
    current_agent: str = "Support",
    additional_context: Optional[Dict[str, Any]] = None
) -> AgentState:
    """Create initial agent state for LangGraph processing"""
    initial_message = request.to_initial_message(current_agent)
    
    return AgentState(
        messages=[initial_message.to_langchain_message()],
        agent_context=additional_context or {},
        request_id=request.request_id,
        user_id=request.user_id,
        current_agent=current_agent,
        next_agent=None,
        conversation_flow=[],
        processing_metadata={
            "start_time": datetime.now(),
            "max_processing_time": request.max_processing_time,
            "priority": request.priority
        }
    )
```

## Usage Examples

### Basic Message Creation
```python
from models.message import Message, Response

# Create inter-agent message
message = Message(
    content="Analyze this database performance issue",
    source_agent="Support",
    target_agent="Technical",
    message_type="handoff",
    priority="high",
    context={
        "customer_id": "12345",
        "issue_type": "performance",
        "database": "postgresql"
    }
)

# Create response
response = Response(
    content="Database analysis completed. Found index optimization opportunities.",
    source_agent="Technical",
    target_agent="Support",
    confidence=0.9,
    reasoning="Identified specific slow queries and missing indexes",
    metadata={
        "analysis_time": 45.2,
        "queries_analyzed": 12,
        "optimization_potential": "high"
    }
)
```

### User Request Processing
```python
from models.message import UserRequest

# Create user request
request = UserRequest(
    query="My application is running slowly during peak hours",
    user_id="customer_789",
    priority="high",
    context={
        "application": "web_app",
        "user_count": 500,
        "peak_hours": "9am-5pm"
    },
    max_processing_time=120
)

# Convert to initial message
initial_message = request.to_initial_message("Support")
```

### LangGraph Integration
```python
from models.message import create_agent_state, AgentState

# Create LangGraph-compatible state
state = create_agent_state(
    request=user_request,
    current_agent="Support",
    additional_context={
        "workflow_type": "customer_support",
        "escalation_level": 1
    }
)

# Use in LangGraph workflow
async def process_workflow(state: AgentState) -> AgentState:
    # Process with current agent
    current_agent = state["current_agent"]
    # ... workflow logic
    return updated_state
```

### Conversation Flow Tracking
```python
from models.message import ChainResult

# Create final result
result = ChainResult(
    request_id=request.request_id,
    response="Your application performance has been optimized...",
    agents_involved=["Support", "Technical", "Product"],
    conversation_flow=[
        {
            "hop": 1,
            "agent": "Support",
            "action": "initial_analysis",
            "confidence": 0.6
        },
        {
            "hop": 2,
            "agent": "Technical",
            "action": "performance_analysis",
            "confidence": 0.9
        }
    ],
    total_processing_time=89.5,
    success=True,
    confidence_score=0.92,
    performance_metrics={
        "total_hops": 2,
        "average_confidence": 0.75,
        "agents_consulted": 2
    }
)
```

## Design Patterns

### 1. Data Transfer Object (DTO)
All models serve as DTOs for structured data exchange.

### 2. Builder Pattern
Models support builder-style construction with optional parameters.

### 3. Factory Pattern
Utility functions act as factories for creating complex models.

### 4. Adapter Pattern
LangChain conversion methods act as adapters between systems.

## Integration Points

### Agent Communication
- Messages structure all inter-agent communication
- Responses provide feedback and confidence scoring
- Context preservation throughout conversation chains

### Orchestrator Integration
- UserRequest models incoming requests
- ChainResult models final outputs
- Conversation flow tracking for analysis

### LangGraph Workflows
- AgentState for workflow state management
- Message conversion for LangChain compatibility
- Structured state transitions

## Best Practices

### 1. Model Design
- Use appropriate field constraints and validation
- Provide meaningful default values
- Include comprehensive docstrings

### 2. Type Safety
- Leverage Pydantic validation
- Use Literal types for constrained values
- Include proper type hints

### 3. Performance
- Keep models lightweight
- Use efficient serialization
- Avoid deep nesting where possible

## Troubleshooting

### Common Issues

1. **Validation Errors**
   ```python
   # Ensure required fields are provided
   message = Message(
       content="Required content",
       source_agent="Required source",
       target_agent="Required target"
   )
   ```

2. **LangChain Conversion Issues**
   ```python
   # Check for proper message types
   if isinstance(lc_msg, (HumanMessage, AIMessage)):
       # Safe to convert
       pass
   ```

3. **Context Preservation**
   ```python
   # Always preserve context in handoffs
   handoff_message = Message(
       content=query,
       source_agent=self.name,
       target_agent=target_agent,
       context={
           **original_message.context,
           "handoff_reason": "specialist_needed"
       }
   )
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