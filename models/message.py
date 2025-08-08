from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal, TypedDict, Annotated
from datetime import datetime
import uuid
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages


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
        "technical_analysis", "product_analysis", "support_analysis",
        "technical_product_analysis", "product_technical_analysis"
    ] = "query"
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    requires_response: bool = Field(default=True)
    conversation_id: Optional[str] = Field(default=None)
    
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
    
    def to_langchain_message(self) -> AIMessage:
        """Convert to LangChain AI message format"""
        return AIMessage(
            content=self.content,
            additional_kwargs={
                "response_id": self.response_id,
                "source_agent": self.source_agent,
                "target_agent": self.target_agent,
                "confidence": self.confidence,
                "needs_clarification": self.needs_clarification,
                "suggested_next_agent": self.suggested_next_agent,
                "reasoning": self.reasoning,
                "metadata": self.metadata or {}
            }
        )

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
    
    def to_initial_message(self, target_agent: str = "Support") -> Message:
        """Convert to initial message for agent processing"""
        return Message(
            content=self.query,
            source_agent="user",
            target_agent=target_agent,
            message_type="query",
            priority=self.priority,
            context={
                "user_id": self.user_id,
                "request_id": self.request_id,
                "expected_format": self.expected_response_format,
                **(self.context or {})
            },
            conversation_id=self.request_id
        )

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


class AgentHandoff(BaseModel):
    """Represents a handoff between agents in LangGraph style"""
    from_agent: str
    to_agent: str
    reason: str
    context_data: Dict[str, Any] = Field(default_factory=dict)
    handoff_type: Literal["query", "clarification", "escalation", "completion"] = "query"
    
class ConversationFlow(BaseModel):
    """Tracks the flow of conversation through agents"""
    step_number: int
    agent_name: str
    action: Literal["received", "processed", "responded", "handed_off", "completed"]
    timestamp: datetime = Field(default_factory=datetime.now)
    message_content: str
    processing_time: Optional[float] = None
    confidence: Optional[float] = None
 
 
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