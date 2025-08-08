# Support Agent - Customer Communication Orchestrator Guide

## Overview
The `SupportAgent` serves as the primary customer interface and conversation coordinator in the bidirectional agent chaining system. It specializes in customer communication, intelligent problem analysis, and seamless orchestration of expert agents to provide comprehensive solutions.

## File Purpose
This customer-facing agent provides:
- **Customer Interface**: Primary point of contact for user interactions
- **Problem Analysis**: Intelligent analysis of customer queries and issues
- **Agent Orchestration**: Coordinated handoffs to specialized expert agents
- **Response Synthesis**: Combining expert knowledge into customer-friendly responses
- **Communication Translation**: Converting technical solutions into accessible language
- **Solution Delivery**: Ensuring complete and actionable customer responses

## Class Architecture

### Core Dependencies
```python
from typing import Optional, Literal
from agents.base_agent import BaseAgent
from models.message import Message, Response
```

### Class Definition
```python
class SupportAgent(BaseAgent):
    """
    Customer Support Agent specialized in customer communication and coordination.
    Acts as the primary interface for users and orchestrates other agents.
    """
```

## Initialization and System Prompt

### Constructor
```python
def __init__(self):
    super().__init__(
        name="Support",
        expertise="customer_communication",
        system_prompt="""[Comprehensive customer service prompt]"""
    )
```

### Comprehensive Customer Service Expertise
```python
system_prompt = """
You are an Expert Customer Support Agent with advanced capabilities in:

PRIMARY RESPONSIBILITIES:
- Understanding customer problems and translating them into technical terms
- Communicating complex technical solutions in user-friendly language
- Identifying when to collaborate with technical or product experts
- Providing empathetic, professional, and comprehensive customer service
- Ensuring customer satisfaction with complete and actionable responses

BIDIRECTIONAL AGENT CHAIN ROLE:
- Serve as the primary customer interface and conversation coordinator
- Analyze customer queries to determine required expertise
- Use HANDOFF_REQUEST to collaborate with Technical or Product agents when needed
- Synthesize expert knowledge into customer-friendly responses
- Ensure all customer concerns are fully addressed before concluding

COLLABORATION PATTERNS:
- For technical issues, system problems, or error diagnostics → Technical Agent
- For product features, limitations, configurations, or capabilities → Product Agent
- For complex issues requiring multiple perspectives → coordinate sequential handoffs
- Always provide context about the customer's situation when requesting handoffs

COMMUNICATION STYLE:
- Empathetic and professional tone
- Clear, jargon-free explanations
- Proactive problem-solving approach
- Always acknowledge customer concerns
- Provide step-by-step guidance when appropriate
- Offer alternatives and follow-up options

QUALITY STANDARDS:
- Ensure responses are complete and actionable
- Verify technical accuracy through expert collaboration
- Maintain customer focus throughout the interaction
- Provide clear next steps or resolution paths
- Follow up on complex issues appropriately
"""
```

**Key Elements**:
- Customer-centric approach with empathy and professionalism
- Clear collaboration patterns for different types of issues
- Quality standards ensuring complete and accurate responses
- Emphasis on actionable solutions and follow-up

## Enhanced Agent Collaboration

### Intelligent Expert Consultation
```python
async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
    """
    Query another agent for specialized information.
    Enhanced to provide better context for expert agents.
    """
    if target_agent not in self.agent_registry:
        print(f"❌ Support Agent: {target_agent} not available")
        return None
    
    print(f"🔄 Support Agent collaborating with {target_agent}")
    print(f"   Query: {query}")
    
    handoff_message = Message(
        content=query,
        source_agent=self.name,
        target_agent=target_agent,
        message_type="query",
        context={
            "collaboration_type": "expert_consultation",
            "customer_facing": True,
            "requires_translation": True,
            "urgency": "normal"
        }
    )
    
    try:
        response = await self.agent_registry[target_agent].process_message(handoff_message)
        print(f"📥 Support Agent received expert input from {target_agent}")
        return response
    except Exception as e:
        print(f"❌ Support Agent: Error collaborating with {target_agent}: {e}")
        return None
```

**Key Features**:
- Detailed logging of collaboration activities
- Rich context provided to expert agents
- Proper error handling and fallback mechanisms
- Clear indication of customer-facing nature

## Customer-Centric Message Processing

### Enhanced Response Generation
```python
async def _generate_response(self, message: Message) -> Response:
    """
    Enhanced response generation with intelligent agent coordination.
    Overrides base implementation to handle customer-centric workflows.
    """
    
    if message.source_agent == "user":
        print(f"🎯 Support Agent analyzing customer query...")
        
        initial_response = await super()._generate_response(message)
        
        needs_handoff, target_agent, handoff_query = self._parse_handoff_request(initial_response.content)
        
        if needs_handoff and target_agent in self.agent_registry:
            print(f"🔀 Support Agent coordinating with {target_agent} for expert input")
            
            enhanced_query = self._enhance_handoff_query(handoff_query, message)
            
            expert_response = await self.query_other_agent(target_agent, enhanced_query)
            
            if expert_response:
                final_response = await self._synthesize_customer_response(
                    initial_response.content, 
                    expert_response, 
                    message
                )
                print(f"✅ Support Agent provided comprehensive customer response")
                return final_response
            else:
                print(f"⚠️ Support Agent: Collaboration with {target_agent} failed, using initial response")
        
        return initial_response
    
    elif message.message_type == "handoff":
        print(f"🤝 Support Agent processing handoff from {message.source_agent}")
        return await super()._generate_response(message)
    
    else:
        return await super()._generate_response(message)
```

**Processing Flow**:
1. Analyzes customer queries with specialized handling
2. Generates initial assessment and identifies collaboration needs
3. Coordinates with expert agents when specialized knowledge is required
4. Synthesizes expert input into customer-friendly responses
5. Ensures comprehensive solutions before delivery

## Context Enhancement

### Enhanced Handoff Query Generation
```python
def _enhance_handoff_query(self, query: str, original_message: Message) -> str:
    """Enhance handoff query with additional customer context"""
    context_info = []
    
    if hasattr(original_message, 'context') and original_message.context:
        user_id = original_message.context.get('user_id', 'unknown')
        context_info.append(f"Customer ID: {user_id}")
        
        if 'priority' in original_message.context:
            context_info.append(f"Priority: {original_message.context['priority']}")
    
    enhanced_query = f"""
CUSTOMER CONTEXT:
Original Query: "{original_message.content}"
{' | '.join(context_info) if context_info else 'Standard customer inquiry'}

EXPERT CONSULTATION REQUEST:
{query}

RESPONSE REQUIREMENTS:
- Provide technical accuracy suitable for customer explanation
- Include any relevant limitations or considerations
- Suggest specific steps or solutions where possible
- Note any follow-up actions that might be needed
"""
    
    return enhanced_query
```

**Enhancement Features**:
- Preserves original customer context
- Adds metadata about customer and priority
- Provides clear requirements for expert responses
- Ensures responses are suitable for customer communication

## Customer Response Synthesis

### Advanced Multi-Agent Response Synthesis
```python
async def _synthesize_customer_response(
    self, 
    initial_content: str, 
    expert_response: Response, 
    original_message: Message
) -> Response:
    """
    Synthesize expert input into a comprehensive, customer-friendly response.
    Enhanced version of the base synthesis method for customer communication.
    """
    
    initial_analysis = initial_content.split('HANDOFF_REQUEST:')[0].strip()
    
    synthesis_prompt = f"""
CUSTOMER RESPONSE SYNTHESIS TASK:
Create a comprehensive, customer-friendly response that combines your customer service expertise with expert technical/product knowledge.

CUSTOMER'S ORIGINAL QUESTION:
"{original_message.content}"

YOUR INITIAL CUSTOMER SERVICE ANALYSIS:
{initial_analysis}

EXPERT INPUT FROM {expert_response.source_agent.upper()} AGENT:
{expert_response.content}

SYNTHESIS REQUIREMENTS:
1. Address the customer's specific concern directly and completely
2. Translate any technical information into clear, understandable language
3. Provide actionable steps or solutions the customer can follow
4. Maintain an empathetic and professional tone throughout
5. Include any important limitations, considerations, or warnings
6. Offer follow-up support or next steps if appropriate
7. Ensure the response feels cohesive and complete

CUSTOMER-FOCUSED RESPONSE:
"""
    
    try:
        from google.genai import types
        
        generation_config = types.GenerateContentConfig(
            temperature=max(0.3, self.temperature * 0.7), 
            max_output_tokens=self.max_tokens,
            candidate_count=1
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=synthesis_prompt,
            config=generation_config
        )
        
        synthesized_content = response.text.strip()
        
        confidence_score = min(0.95, (
            self._calculate_confidence(synthesized_content) + 
            expert_response.confidence + 
            0.1 
        ) / 2)
        
        return Response(
            content=synthesized_content,
            source_agent=self.name,
            target_agent=original_message.source_agent,
            confidence=confidence_score,
            reasoning=f"Customer response synthesized from Support analysis and {expert_response.source_agent} expertise",
            metadata={
                "collaboration_agents": [expert_response.source_agent],
                "synthesis_type": "customer_focused",
                "expert_confidence": expert_response.confidence
            }
        )
        
    except Exception as e:
        print(f"❌ Support Agent: Error synthesizing customer response: {e}")
        
        # Graceful fallback response
        fallback_content = f"""
I understand your concern about: {original_message.content}

Based on our analysis: {expert_response.content}

I'm here to help you resolve this issue. Please let me know if you need any clarification or have additional questions.
"""
        
        return Response(
            content=fallback_content,
            source_agent=self.name,
            target_agent=original_message.source_agent,
            confidence=max(0.6, expert_response.confidence * 0.8),
            reasoning="Fallback customer response due to synthesis error"
        )
```

**Synthesis Features**:
- Lower temperature for more focused, professional responses
- Comprehensive customer communication requirements
- Graceful error handling with fallback responses
- Enhanced confidence calculation incorporating expert input
- Rich metadata for tracking collaboration effectiveness

## Performance Tracking

### Collaboration Statistics
```python
def get_collaboration_stats(self) -> dict:
    """Get statistics about agent collaborations"""
    stats = {
        "total_collaborations": 0,
        "collaboration_partners": {},
        "success_rate": 0.0
    }
    
    for msg in self.conversation_history:
        if hasattr(msg, 'message_type') and msg.message_type == "handoff":
            stats["total_collaborations"] += 1
            partner = msg.target_agent
            stats["collaboration_partners"][partner] = stats["collaboration_partners"].get(partner, 0) + 1
    
    return stats
```

## Usage Examples

### Basic Customer Interaction
```python
# Initialize Support Agent
support_agent = SupportAgent()

# Register expert agents
support_agent.register_agent("Technical", technical_agent)
support_agent.register_agent("Product", product_agent)

# Process customer query
customer_message = Message(
    content="My application keeps crashing when I try to upload large files",
    source_agent="user",
    target_agent="Support",
    message_type="query",
    context={
        "user_id": "customer_123",
        "priority": "high",
        "product_tier": "enterprise"
    }
)

response = await support_agent.process_message(customer_message)
```

### Multi-Agent Coordination
```python
# Complex issue requiring multiple expert consultations
complex_query = Message(
    content="Customer reports data synchronization issues between mobile and web apps, with occasional authentication failures",
    source_agent="user",
    target_agent="Support"
)

# Support agent will:
# 1. Analyze the query
# 2. Identify need for technical expertise (sync issues)
# 3. Consult Technical agent
# 4. Potentially consult Product agent for feature-specific guidance
# 5. Synthesize comprehensive customer response

response = await support_agent.process_message(complex_query)
```

### Direct Expert Consultation
```python
# Support agent querying technical expert
tech_response = await support_agent.query_other_agent(
    "Technical",
    "Customer experiencing 500 errors during peak usage - need analysis of potential causes and solutions"
)

# Support agent querying product expert
product_response = await support_agent.query_other_agent(
    "Product",
    "Customer wants to enable advanced reporting but unsure about configuration requirements"
)
```

## Design Patterns

### 1. Facade Pattern
Support Agent acts as a facade, hiding the complexity of multi-agent coordination from customers.

### 2. Mediator Pattern
Coordinates communication between customers and expert agents.

### 3. Template Method Pattern
Structured approach to customer service and expert consultation.

### 4. Strategy Pattern
Different response strategies based on customer query type and complexity.

## Integration Points

### Customer Interface Integration
- Primary entry point for all customer interactions
- Handles various customer communication channels
- Maintains customer context throughout conversations

### Expert Agent Integration
- Seamless handoffs to Technical and Product agents
- Rich context sharing for effective collaboration
- Response synthesis combining multiple expert inputs

### Orchestrator Integration
- Participates in dynamic routing decisions
- Provides customer satisfaction metrics
- Supports conversation flow tracking

## Best Practices

### 1. Customer Communication
- Always maintain empathetic and professional tone
- Translate technical information into customer-friendly language
- Provide clear, actionable guidance
- Acknowledge customer concerns explicitly

### 2. Expert Collaboration
- Provide sufficient context for effective expert consultation
- Use structured handoff requests with clear requirements
- Synthesize expert input maintaining customer focus
- Handle collaboration failures gracefully

### 3. Problem Resolution
- Ensure comprehensive analysis of customer issues
- Coordinate multiple experts when needed
- Provide complete solutions rather than partial fixes
- Offer appropriate follow-up and escalation paths

## Troubleshooting

### Common Issues

1. **Expert Agent Unavailable**
   ```python
   # Graceful handling when expert agents are unavailable
   if expert_response is None:
       fallback_response = await self._provide_general_guidance(message)
       return fallback_response
   ```

2. **Synthesis Failures**
   ```python
   # Fallback to expert response if synthesis fails
   except Exception as synthesis_error:
       return self._create_fallback_response(expert_response, original_message)
   ```

3. **Context Loss**
   ```python
   # Ensure customer context is preserved throughout handoffs
   def _preserve_customer_context(self, message: Message) -> Dict[str, Any]:
       return {
           "original_query": message.content,
           "customer_id": message.context.get("user_id"),
           "priority": message.context.get("priority"),
           "product_context": message.context.get("product_tier")
       }
   ```

## Extension Points

### Custom Communication Styles
```python
class CustomSupportAgent(SupportAgent):
    def __init__(self, communication_style: str = "formal"):
        super().__init__()
        self.communication_style = communication_style
    
    async def _adapt_communication_style(self, content: str, customer_profile: Dict[str, Any]) -> str:
        """Adapt communication style based on customer profile"""
        # Implementation for style adaptation
        return adapted_content
```

### Advanced Customer Analytics
```python
async def analyze_customer_satisfaction(self, conversation_history: List[Message]) -> Dict[str, Any]:
    """Analyze customer satisfaction from conversation patterns"""
    satisfaction_metrics = {
        "resolution_time": self._calculate_resolution_time(conversation_history),
        "handoff_efficiency": self._analyze_handoff_patterns(conversation_history),
        "response_completeness": self._assess_response_quality(conversation_history)
    }
    return satisfaction_metrics
```

## Testing Considerations

### Unit Tests
```python
@pytest.mark.asyncio
async def test_support_agent_customer_synthesis():
    """Test customer response synthesis functionality"""
    support_agent = SupportAgent()
    
    # Mock expert response
    expert_response = Response(
        content="Technical analysis: Database connection pool exhausted",
        source_agent="Technical",
        confidence=0.9
    )
    
    original_message = Message(
        content="My app is running slowly",
        source_agent="user",
        target_agent="Support"
    )
    
    synthesized = await support_agent._synthesize_customer_response(
        "Initial support analysis", expert_response, original_message
    )
    
    assert "database" not in synthesized.content.lower()  # Technical jargon removed
    assert "slowly" in synthesized.content.lower()  # Customer language preserved
    assert synthesized.confidence > 0.7
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_end_to_end_customer_support():
    """Test complete customer support workflow"""
    support_agent = SupportAgent()
    technical_agent = TechnicalAgent()
    
    support_agent.register_agent("Technical", technical_agent)
    
    customer_query = Message(
        content="Getting error 500 when uploading files",
        source_agent="user",
        target_agent="Support"
    )
    
    response = await support_agent.process_message(customer_query)
    
    assert response.source_agent == "Support"
    assert "500" in response.content
    assert response.confidence > 0.6
    assert "Technical" in response.metadata.get("collaboration_agents", [])
```

## Performance Considerations

### Response Time Optimization
- Efficient expert consultation patterns
- Parallel processing where applicable
- Optimized synthesis algorithms
- Proper timeout handling

### Customer Experience
- Minimize handoff overhead
- Provide progress indicators for complex queries
- Ensure response completeness
- Maintain conversation context effectively

### Collaboration Efficiency
- Smart routing to most relevant experts
- Avoid unnecessary multiple handoffs
- Cache common expert responses
- Monitor collaboration success rates

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