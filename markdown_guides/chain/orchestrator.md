# Chain Orchestrator - Dynamic Bidirectional Agent Orchestration Guide

## Overview
The `ChainOrchestrator` is the core component that manages dynamic bidirectional agent chaining with confidence-based iteration. It orchestrates intelligent agent communication, tracks conversation flows, and ensures optimal problem resolution through collaborative agent networks.

## File Purpose
The orchestrator provides:
- **Dynamic Agent Routing**: Intelligent selection of agents based on content analysis
- **Bidirectional Communication**: True agent-to-agent collaboration with handoffs
- **Confidence-Based Iteration**: Continues processing until confidence threshold is met
- **Performance Monitoring**: Comprehensive metrics and health tracking
- **Conversation Flow Management**: Detailed tracking of agent interactions

## Class Architecture

### Core Dependencies
```python
import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from agents.support_agent import SupportAgent
from agents.technical_agent import TechnicalAgent
from agents.product_agent import ProductAgent
from agents.motogp_rag_agent import MotoGPRAGAgent
from models.message import Message, UserRequest, ChainResult, Response
```

## Initialization and Configuration

### Constructor and Configuration Loading
```python
def __init__(self):
    print("🏗️ Initializing Modern Bidirectional Agent Chain with LangGraph...")
    
    self._load_configuration()
    self._initialize_agents()
    self._setup_agent_network()
    self._initialize_dynamic_routing()
    
    self.conversation_flows = {}
    self.performance_metrics = {}
    self.active_conversations = {}
```

### Environment Configuration
```python
def _load_configuration(self):
    """Load configuration from environment variables"""
    self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    self.max_chain_hops = int(os.getenv("MAX_CHAIN_HOPS", 30)) 
    self.agent_timeout = int(os.getenv("AGENT_RESPONSE_TIMEOUT", 30))
    self.max_conversation_history = int(os.getenv("MAX_CONVERSATION_HISTORY", 50))
    self.enable_performance_monitoring = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    self.enable_agent_logs = os.getenv("ENABLE_AGENT_COMMUNICATION_LOGS", "true").lower() == "true"
    self.enable_detailed_flow_logging = os.getenv("ENABLE_DETAILED_FLOW_LOGGING", "true").lower() == "true"
    self.enable_dynamic_routing = os.getenv("ENABLE_DYNAMIC_ROUTING", "true").lower() == "true"
```

## Agent Network Setup

### Agent Initialization
```python
def _initialize_agents(self):
    """Initialize all agents with enhanced capabilities"""
    try:
        self.support_agent = SupportAgent()
        self.technical_agent = TechnicalAgent()
        self.product_agent = ProductAgent()
        self.motogp_rag_agent = MotoGPRAGAgent()
        self.agents = {
            "Support": self.support_agent,
            "Technical": self.technical_agent,
            "Product": self.product_agent,
            "MotoGP_RAG": self.motogp_rag_agent
        }
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        raise
```

### Bidirectional Network Setup
```python
def _setup_agent_network(self):
    """Set up the bidirectional agent network with full connectivity"""
    # Register all agents with each other for bidirectional communication
    self.support_agent.register_agent("Technical", self.technical_agent)
    self.support_agent.register_agent("Product", self.product_agent)
    self.support_agent.register_agent("MotoGP_RAG", self.motogp_rag_agent)
    
    self.technical_agent.register_agent("Support", self.support_agent)
    self.technical_agent.register_agent("Product", self.product_agent)
    self.technical_agent.register_agent("MotoGP_RAG", self.motogp_rag_agent)
    
    # Continue for all agent pairs...
```

## Dynamic Routing System

### Content Analysis for Routing
```python
def _analyze_content_for_routing(self, content: str) -> Dict[str, float]:
    """Analyze content to determine optimal agent routing"""
    content_lower = content.lower()
    routing_scores = {
        "Support": 0.1,
        "Technical": 0.1,
        "Product": 0.1,
        "MotoGP_RAG": 0.1
    }
    
    # Analyze for different domains
    for domain, keywords in self.routing_patterns.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if domain == "technical":
            routing_scores["Technical"] += score * 0.2
        elif domain == "product":
            routing_scores["Product"] += score * 0.2
        elif domain == "motogp":
            routing_scores["MotoGP_RAG"] += score * 0.3
    
    return routing_scores
```

### Dynamic Agent Selection
```python
def _select_next_agent(
    self, 
    current_agent: str, 
    content: str, 
    conversation_history: List[Dict],
    confidence_score: float,
    agent_usage_count: Optional[Dict[str, int]] = None
) -> Optional[str]:
    """Dynamically select the next agent based on content and context"""
    
    routing_scores = self._analyze_content_for_routing(content)
    
    # Apply usage penalties to avoid loops
    if agent_usage_count is None:
        used_agents = [step.get("agent") for step in conversation_history]
        agent_usage = {agent: used_agents.count(agent) for agent in self.agents.keys()}
    else:
        agent_usage = agent_usage_count
    
    for agent in routing_scores:
        usage_count = agent_usage.get(agent, 0)
        if usage_count > 1:
            penalty = 1.0 - (usage_count * 0.15)
            routing_scores[agent] *= max(0.3, penalty)
    
    # Select highest scoring agent
    if routing_scores:
        selected_agent = max(routing_scores, key=routing_scores.get)
        if routing_scores[selected_agent] > 0.3: 
            return selected_agent
    
    return None
```

## Core Processing Engine

### Main Request Processing
```python
async def process_request(self, user_request: UserRequest) -> Dict[str, Any]:
    """
    Process user request through the dynamic bidirectional agent chain.
    Agents iterate until confidence threshold (0.7) is reached or max hops exceeded.
    """
    start_time = time.time()
    conversation_id = user_request.request_id
    
    # Initialize conversation state
    conversation_state = {
        "request_id": conversation_id,
        "user_id": user_request.user_id,
        "original_query": user_request.query,
        "conversation_flow": [],
        "current_response": None,
        "current_confidence": 0.0,
        "hop_count": 0,
        "agents_involved": set(),
        "start_time": start_time
    }
    
    current_agent = "Support"
    current_message = Message(
        content=user_request.query,
        source_agent="user",
        target_agent=current_agent,
        context={
            "request_id": conversation_id,
            "user_id": user_request.user_id,
            "priority": user_request.priority
        }
    )
    
    agent_usage_count = {agent: 0 for agent in self.agents.keys()}
    
    # Main processing loop
    while (conversation_state["hop_count"] < self.max_chain_hops and 
           conversation_state["current_confidence"] < self.confidence_threshold):
        
        conversation_state["hop_count"] += 1
        agent_usage_count[current_agent] += 1
        conversation_state["agents_involved"].add(current_agent)
        
        # Process with current agent
        response = await self._process_agent_interaction(
            current_agent, 
            current_message, 
            conversation_state
        )
        
        conversation_state["current_response"] = response
        conversation_state["current_confidence"] = response.confidence
        
        # Check if confidence threshold is met
        if response.confidence >= self.confidence_threshold:
            break
        
        # Select next agent
        next_agent = self._select_next_agent(
            current_agent,
            response.content,
            conversation_state["conversation_flow"],
            response.confidence,
            agent_usage_count
        )
        
        if not next_agent:
            break
        
        # Prepare message for next agent
        current_message = Message(
            content=f"Previous analysis from {current_agent}: {response.content}\n\nOriginal query: {user_request.query}",
            source_agent=current_agent,
            target_agent=next_agent,
            message_type="handoff",
            context={
                "request_id": conversation_id,
                "hop_count": conversation_state["hop_count"],
                "previous_confidence": response.confidence,
                "original_query": user_request.query
            }
        )
        
        current_agent = next_agent
    
    # Generate final result
    return self._format_result_response(
        self._create_chain_result(conversation_state, start_time)
    )
```

## Agent Interaction Processing

### Single Agent Interaction
```python
async def _process_agent_interaction(
    self, 
    agent_name: str, 
    message: Message, 
    conversation_state: Dict[str, Any]
) -> Response:
    """Process a single agent interaction with detailed logging and consultation tracking"""
    start_time = time.time()
    
    try:
        agent_instance = self.agents[agent_name]
        
        # Track consultations through print monitoring
        consultation_log = []
        original_print = print
        
        def tracking_print(*args, **kwargs):
            output = ' '.join(str(arg) for arg in args)
            # Capture collaboration patterns
            if "🔀" in output and "requesting handoff to" in output:
                # Extract consulted agent
                parts = output.split("requesting handoff to")
                if len(parts) > 1:
                    consulted_agent = parts[1].strip()
                    consultation_log.append({
                        "consulted_agent": consulted_agent,
                        "type": "handoff_request"
                    })
            return original_print(*args, **kwargs)
        
        # Temporarily replace print to capture consultations
        import builtins
        builtins.print = tracking_print
        
        try:
            response = await agent_instance.process_message(message)
        finally:
            builtins.print = original_print
        
        processing_time = time.time() - start_time
        
        # Track performance metrics
        if self.enable_performance_monitoring:
            self._track_agent_performance(agent_name, processing_time, response.confidence)
        
        # Add consultation steps to conversation flow
        for consultation in consultation_log:
            if consultation["type"] in ["handoff_request", "collaboration"]:
                consulted_agent = consultation["consulted_agent"]
                conversation_state["agents_involved"].add(consulted_agent)
                
                # Create flow step for consultation
                consultation_step = {
                    "hop": conversation_state["hop_count"] + 0.1,
                    "agent": consulted_agent,
                    "action": "consulted",
                    "timestamp": datetime.now(),
                    "confidence": response.confidence * 0.8,
                    "content_preview": f"Consulted by {agent_name} for specialized knowledge",
                    "processing_time": processing_time * 0.3
                }
                conversation_state["conversation_flow"].append(consultation_step)
        
        return response
        
    except Exception as e:
        return Response(
            content=f"Error processing with {agent_name}: {str(e)}",
            source_agent=agent_name,
            target_agent=message.source_agent,
            confidence=0.1,
            reasoning=f"Error in {agent_name}: {str(e)}"
        )
```

## Performance Monitoring

### Agent Performance Tracking
```python
def _track_agent_performance(self, agent_name: str, processing_time: float, confidence: float):
    """Track agent performance metrics"""
    if agent_name not in self.performance_metrics:
        self.performance_metrics[agent_name] = {
            "total_requests": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "average_confidence": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0
        }
    
    metrics = self.performance_metrics[agent_name]
    metrics["total_requests"] += 1
    metrics["total_time"] += processing_time
    metrics["average_time"] = metrics["total_time"] / metrics["total_requests"]
    metrics["min_time"] = min(metrics["min_time"], processing_time)
    metrics["max_time"] = max(metrics["max_time"], processing_time)
    
    # Update confidence average
    current_avg = metrics["average_confidence"]
    n = metrics["total_requests"]
    metrics["average_confidence"] = ((current_avg * (n - 1)) + confidence) / n
```

## Health and Status Monitoring

### Comprehensive Agent Status
```python
async def get_agent_status(self) -> Dict[str, Any]:
    """Get comprehensive status of all agents and the orchestrator"""
    
    agent_statuses = {}
    for name, agent in self.agents.items():
        try:
            health = await agent.health_check()
            info = agent.get_agent_info()
            agent_statuses[name.lower() + "_agent"] = {
                **info,
                "health": health,
                "performance": self.performance_metrics.get(name, {})
            }
        except Exception as e:
            agent_statuses[name.lower() + "_agent"] = {
                "name": name,
                "status": "error",
                "error": str(e)
            }
    
    orchestrator_status = {
        "version": "2.0",
        "dynamic_routing": self.enable_dynamic_routing,
        "confidence_threshold": self.confidence_threshold,
        "max_chain_hops": self.max_chain_hops,
        "agent_timeout": self.agent_timeout,
        "total_conversations": len(self.conversation_flows),
        "active_conversations": len(self.active_conversations)
    }
    
    return {
        "orchestrator": orchestrator_status,
        "agents": agent_statuses,
        "network_health": all(
            agent_status.get("health", {}).get("status") == "healthy"
            for agent_status in agent_statuses.values()
        )
    }
```

## Usage Examples

### Basic Orchestrator Usage
```python
# Initialize orchestrator
orchestrator = ChainOrchestrator()

# Create user request
request = UserRequest(
    query="My application crashes during file uploads",
    user_id="customer_123",
    priority="high"
)

# Process request through bidirectional agent chain
result = await orchestrator.process_request(request)

print(f"Final response: {result['response']}")
print(f"Agents involved: {result['agents_involved']}")
print(f"Confidence: {result['confidence_score']}")
```

### Health Monitoring
```python
# Check system health
status = await orchestrator.get_agent_status()

print(f"Network health: {status['network_health']}")
print(f"Active conversations: {status['orchestrator']['active_conversations']}")

# Get performance metrics
metrics = orchestrator.get_performance_metrics()
print(f"Average chain length: {metrics['orchestrator_metrics']['average_chain_length']}")
```

### Conversation Flow Analysis
```python
# Get detailed conversation history
conversation_history = orchestrator.get_conversation_history(request_id)

for step in conversation_history:
    print(f"Hop {step['hop']}: {step['agent']} - {step['action']}")
    print(f"Confidence: {step['confidence']:.2f}")
```

## Design Patterns

### 1. Orchestrator Pattern
Central coordination of multiple agent services.

### 2. Chain of Responsibility
Dynamic agent chaining based on confidence and content.

### 3. Observer Pattern
Performance monitoring and conversation tracking.

### 4. Strategy Pattern
Different routing strategies based on content analysis.

## Best Practices

### 1. Configuration Management
- Use environment variables for tunable parameters
- Provide sensible defaults for all configurations
- Monitor and adjust thresholds based on performance

### 2. Performance Optimization
- Track agent performance metrics
- Implement proper timeout handling
- Use async processing throughout

### 3. Error Resilience
- Graceful degradation when agents are unavailable
- Comprehensive error handling and logging
- Fallback mechanisms for critical failures

## Troubleshooting

### Common Issues

1. **Infinite Loops**
   ```python
   # Prevented by usage count penalties and max hop limits
   if usage_count > 1:
       penalty = 1.0 - (usage_count * 0.15)
       routing_scores[agent] *= max(0.3, penalty)
   ```

2. **Low Confidence Scores**
   ```python
   # Monitor confidence thresholds and agent performance
   if response.confidence < self.confidence_threshold:
       # Continue to next agent or improve analysis
   ```

3. **Agent Unavailability**
   ```python
   # Check agent health and provide fallbacks
   health_status = await agent.health_check()
   if health_status["status"] != "healthy":
       # Use alternative agents or provide error response
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