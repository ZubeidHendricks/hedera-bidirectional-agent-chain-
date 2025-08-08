# Product Agent - Product Knowledge Expert Guide

## Overview
The `ProductAgent` class is a specialized agent focused on product knowledge, feature expertise, and configuration guidance within the bidirectional agent chaining system. It serves as the authoritative source for all product-related information and collaborates intelligently with technical and support agents to provide comprehensive solutions.

## File Purpose
This specialized agent provides:
- **Product Expertise**: Deep knowledge of product features, capabilities, and limitations
- **Configuration Guidance**: Expert advice on product settings and customization
- **Feature Analysis**: Comprehensive analysis of product functionality
- **Integration Support**: Guidance on product integrations and compatibility
- **Collaborative Problem-Solving**: Intelligent handoffs to technical experts when needed
- **Customer-Focused Solutions**: Product recommendations aligned with customer needs

## Class Architecture

### Core Dependencies
```python
from typing import Optional, Literal
from agents.base_agent import BaseAgent
from models.message import Message, Response
```

### Class Definition
```python
class ProductAgent(BaseAgent):
    """
    Product Expert Agent specialized in product knowledge and feature expertise.
    Provides comprehensive product insights within the bidirectional agent chain.
    """
```

## Initialization and Configuration

### Constructor
```python
def __init__(self):
    super().__init__(
        name="Product",
        expertise="product_knowledge",
        system_prompt="""[Comprehensive product expertise prompt]"""
    )
    
    self.capabilities.extend([
        "feature_analysis",
        "product_configuration",
        "compatibility_assessment",
        "integration_guidance",
        "roadmap_insights",
        "licensing_consultation"
    ])
```

**Key Features**:
- Extends base capabilities with product-specific skills
- Focuses on product knowledge domain
- Configured for collaborative problem-solving

## System Prompt Analysis

### Comprehensive Product Expertise Definition
```python
system_prompt = """
You are an Expert Product Knowledge Agent with comprehensive expertise in:

PRIMARY PRODUCT CAPABILITIES:
- Deep knowledge of all product features, capabilities, and limitations
- Understanding of product architecture and design decisions
- Comprehensive knowledge of product roadmap and development priorities
- Expertise in product configuration, settings, and customization options
- Knowledge of product integrations and compatibility requirements
- Understanding of user workflows and common use cases

BIDIRECTIONAL AGENT CHAIN ROLE:
- Provide product expertise to Support and Technical agents
- Analyze product-related aspects of customer issues
- Offer detailed feature explanations and configuration guidance
- Collaborate with Technical agents for implementation-specific questions
- Ensure product recommendations align with customer needs and constraints

PRODUCT KNOWLEDGE DOMAINS:
- Feature functionality and proper usage patterns
- Product limitations and known constraints
- Configuration options and their implications
- Integration capabilities and requirements
- Version differences and upgrade considerations
- Licensing and subscription tier differences
- Compatibility matrices and system requirements

COLLABORATION PATTERNS:
- When technical implementation details are needed → Technical Agent
- When system-level constraints affect product recommendations → Technical Agent
- Always provide product context that enables optimal technical solutions

PRODUCT COMMUNICATION STANDARDS:
- Provide accurate feature descriptions with practical examples
- Explain configuration implications and best practices
- Identify potential conflicts between features or settings
- Suggest optimal product configurations for specific use cases
- Include relevant limitations or prerequisites
- Recommend alternative approaches when preferred options aren't available

QUALITY STANDARDS:
- Ensure product recommendations are current and accurate
- Consider customer's specific product tier and permissions
- Provide step-by-step configuration guidance
- Include relevant warnings about potential impacts
- Suggest testing and validation approaches
- Reference official documentation when appropriate
"""
```

**Key Elements**:
- Comprehensive product knowledge scope
- Clear collaboration patterns with other agents
- Focus on customer needs and practical solutions
- Quality standards for accurate recommendations

## Enhanced Query Processing

### Intelligent Agent Consultation
```python
async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
    """
    Query another agent for specialized information.
    Enhanced for product-technical collaboration patterns.
    """
    if target_agent not in self.agent_registry:
        print(f"❌ Product Agent: {target_agent} not available")
        return None
    
    print(f"📦 Product Agent consulting with {target_agent}")
    print(f"   Product query: {query}")
    
    consultation_message = Message(
        content=query,
        source_agent=self.name,
        target_agent=target_agent,
        message_type="query",
        context={
            "consultation_type": "product_expertise",
            "requires_product_context": True,
            "implementation_focus": True,
            "urgency": "normal"
        }
    )
    
    try:
        response = await self.agent_registry[target_agent].process_message(consultation_message)
        print(f"📥 Product Agent received consultation from {target_agent}")
        return response
    except Exception as e:
        print(f"❌ Product Agent: Error consulting with {target_agent}: {e}")
        return None
```

## Specialized Message Processing

### Enhanced Response Generation
```python
async def _generate_response(self, message: Message) -> Response:
    """
    Enhanced product response generation with comprehensive feature analysis.
    Provides structured product guidance with proper depth.
    """
    
    if message.message_type == "handoff" and message.source_agent == "Support":
        print(f"📦 Product Agent analyzing product aspects of customer issue...")
        return await self._handle_customer_product_issue(message)
        
    elif message.source_agent == "Technical":
        print(f"📦 Product Agent providing product context for technical implementation...")
        return await self._handle_technical_product_query(message)
        
    else:
        print(f"📦 Product Agent performing standard product analysis...")
        return await super()._generate_response(message)
```

### Customer Issue Analysis
```python
async def _handle_customer_product_issue(self, message: Message) -> Response:
    """Handle product issues escalated from customer support"""
    
    original_query = ""
    customer_context = ""
    
    if hasattr(message, 'context') and message.context:
        original_query = message.context.get('original_query', '')
        customer_context = f"Customer inquiry: {original_query}\n" if original_query else ""
    
    enhanced_prompt = f"""
PRODUCT ANALYSIS REQUEST FROM SUPPORT TEAM:

{customer_context}
SUPPORT TEAM ANALYSIS REQUEST:
{message.content}

COMPREHENSIVE PRODUCT ANALYSIS REQUIRED:

1. FEATURE ASSESSMENT:
   - Identify relevant product features and capabilities
   - Assess if the customer's use case is supported
   - Evaluate any feature limitations or constraints

2. CONFIGURATION ANALYSIS:
   - Review optimal configuration options
   - Identify potential configuration conflicts
   - Suggest settings that would address the issue

3. COMPATIBILITY EVALUATION:
   - Check version and tier compatibility
   - Assess integration requirements
   - Identify any system or platform constraints

4. SOLUTION RECOMMENDATIONS:
   - Provide product-based solutions and workarounds
   - Suggest alternative approaches using different features
   - Recommend configuration changes or optimizations

5. IMPLEMENTATION GUIDANCE:
   - Detail step-by-step configuration procedures
   - Highlight important settings and their implications
   - Include verification steps for proper setup

6. LIMITATIONS AND CONSIDERATIONS:
   - Identify any product limitations that apply
   - Note potential impacts on other features or workflows
   - Suggest monitoring or validation approaches

7. ENHANCEMENT OPPORTUNITIES:
   - Identify if upcoming features would address the issue
   - Suggest product upgrades or tier changes if beneficial
   - Recommend additional features that complement the solution

Provide your analysis in a format suitable for customer communication.
"""
    
    analysis_message = Message(
        content=enhanced_prompt,
        source_agent="internal",
        target_agent=self.name,
        message_type="analysis"
    )
    
    product_response = await super()._generate_response(analysis_message)
    
    # Check if technical consultation is needed
    needs_technical_info = self._assess_technical_dependency(product_response.content, message.content)
    
    if needs_technical_info and "Technical" in self.agent_registry:
        print(f"🔀 Product Agent consulting Technical for implementation details...")
        
        technical_query = self._generate_technical_query(product_response.content, message.content)
        technical_response = await self.query_other_agent("Technical", technical_query)
        
        if technical_response:
            return await self._synthesize_product_technical_response(
                product_response, technical_response, message
            )
    
    return product_response
```

## Technical Collaboration

### Technical Dependency Assessment
```python
def _assess_technical_dependency(self, product_content: str, original_query: str) -> bool:
    """Assess if technical implementation details would enhance the product solution"""
    
    technical_indicators = [
        "implementation", "integration", "api", "configuration", "setup",
        "installation", "deployment", "performance", "security", "infrastructure"
    ]
    
    combined_content = f"{product_content} {original_query}".lower()
    
    return any(indicator in combined_content for indicator in technical_indicators)
```

### Technical Query Generation
```python
def _generate_technical_query(self, product_analysis: str, original_query: str) -> str:
    """Generate a focused query for technical agent based on product analysis"""
    
    return f"""
PRODUCT-TECHNICAL CONSULTATION REQUEST:

ORIGINAL CUSTOMER ISSUE:
{original_query}

PRODUCT ANALYSIS SUMMARY:
{product_analysis[:500]}...

TECHNICAL IMPLEMENTATION NEEDED:
- System requirements and infrastructure considerations for the product solution
- Implementation best practices for the recommended product configuration
- Performance implications of the suggested product settings
- Security considerations for the proposed solution
- Integration requirements and technical constraints
- Monitoring and maintenance recommendations

Please provide technical context that would ensure successful implementation of the product solution.
"""
```

### Product-Technical Response Synthesis
```python
async def _synthesize_product_technical_response(
    self, 
    product_response: Response, 
    technical_response: Response, 
    original_message: Message
) -> Response:
    """Synthesize product knowledge with technical implementation details"""
    
    synthesis_prompt = f"""
COMPREHENSIVE PRODUCT-TECHNICAL SOLUTION SYNTHESIS:

Combine your product expertise with technical implementation insights to provide a complete solution.

YOUR PRODUCT ANALYSIS:
{product_response.content}

TECHNICAL TEAM INSIGHTS:
{technical_response.content}

SYNTHESIS REQUIREMENTS:
1. Integrate technical requirements into product recommendations
2. Adjust product configurations based on technical constraints
3. Provide implementation steps that account for both product features and technical considerations
4. Highlight any conflicts between optimal product usage and technical limitations
5. Offer alternative product approaches if technical constraints limit preferred solutions
6. Ensure the solution is both product-optimal and technically feasible

COMPREHENSIVE PRODUCT-TECHNICAL SOLUTION:
"""
    
    try:
        from google.genai import types
        
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
        
        synthesized_content = response.text.strip()
        
        confidence_score = min(0.95, (
            product_response.confidence + 
            technical_response.confidence + 
            0.15 
        ) / 2)
        
        return Response(
            content=synthesized_content,
            source_agent=self.name,
            target_agent=original_message.source_agent,
            confidence=confidence_score,
            reasoning="Comprehensive product solution integrating technical expertise",
            metadata={
                "analysis_type": "product_technical_synthesis",
                "collaboration_agents": [technical_response.source_agent],
                "product_confidence": product_response.confidence,
                "technical_confidence": technical_response.confidence
            }
        )
        
    except Exception as e:
        print(f"❌ Product Agent: Error synthesizing responses: {e}")
        return product_response
```

## Technical Query Handling

### Technical-Product Integration
```python
async def _handle_technical_product_query(self, message: Message) -> Response:
    """Handle product queries from technical agent"""
    
    enhanced_prompt = f"""
TECHNICAL-PRODUCT INTEGRATION CONSULTATION:

TECHNICAL TEAM REQUEST:
{message.content}

PRODUCT KNOWLEDGE FOCUS:
- Product feature capabilities and constraints relevant to the technical implementation
- Configuration options that affect technical behavior
- Product architecture considerations that impact the technical solution
- Version-specific behaviors and compatibility requirements
- Integration patterns and API capabilities
- Known product limitations that might affect technical approaches

Provide detailed product insights that complement technical implementation.
"""
    
    analysis_message = Message(
        content=enhanced_prompt,
        source_agent="internal",
        target_agent=self.name,
        message_type="analysis"
    )
    
    return await super()._generate_response(analysis_message)
```

## Product Capabilities and Analytics

### Comprehensive Capability Overview
```python
def get_product_capabilities(self) -> dict:
    """Get detailed information about product capabilities"""
    return {
        "core_expertise": self.expertise,
        "product_areas": [
            "Feature Functionality",
            "Configuration Management",
            "Integration Capabilities",
            "Compatibility Assessment",
            "Licensing and Tiers",
            "Roadmap and Development",
            "User Experience Optimization",
            "Best Practices and Guidelines"
        ],
        "analysis_methods": [
            "Feature Gap Analysis",
            "Configuration Optimization",
            "Compatibility Assessment",
            "Use Case Evaluation",
            "Product Roadmap Alignment"
        ],
        "collaboration_patterns": {
            "with_support": "Customer-focused product solutions",
            "with_technical": "Implementation-ready product guidance"
        },
        "specializations": [
            "Feature limitations and workarounds",
            "Configuration best practices",
            "Integration requirements",
            "Upgrade and migration guidance",
            "Performance optimization through product settings"
        ]
    }
```

### Feature Matrix Information
```python
def get_feature_matrix(self) -> dict:
    """Get information about feature availability and capabilities"""
    return {
        "feature_categories": [
            "Core Features",
            "Advanced Features", 
            "Integration Features",
            "Administrative Features",
            "Security Features",
            "Performance Features"
        ],
        "configuration_domains": [
            "User Interface Settings",
            "Security and Access Control",
            "Performance and Scaling",
            "Integration and API Settings",
            "Workflow and Automation",
            "Monitoring and Analytics"
        ],
        "compatibility_factors": [
            "Version Requirements",
            "Platform Compatibility",
            "Browser Support",
            "Integration Dependencies",
            "Third-party Requirements"
        ]
    }
```

## Usage Examples

### Basic Product Consultation
```python
# Initialize Product Agent
product_agent = ProductAgent()

# Customer product inquiry
message = Message(
    content="How do I configure SSO for enterprise users?",
    source_agent="Support",
    target_agent="Product",
    message_type="handoff",
    context={
        "original_query": "Customer needs SSO setup help",
        "customer_tier": "enterprise",
        "urgency": "high"
    }
)

response = await product_agent.process_message(message)
```

### Multi-Agent Collaboration
```python
# Setup agent network
product_agent = ProductAgent()
technical_agent = TechnicalAgent()
support_agent = SupportAgent()

# Register agents
product_agent.register_agent("Technical", technical_agent)
product_agent.register_agent("Support", support_agent)

# Complex product-technical query
response = await product_agent.query_other_agent(
    "Technical",
    "What are the infrastructure requirements for implementing our advanced analytics features?"
)
```

### Product Analysis Workflow
```python
# Customer configuration issue
customer_issue = Message(
    content="Customer reports that their API integration is failing with 429 errors",
    source_agent="Support",
    target_agent="Product",
    message_type="handoff",
    context={
        "original_query": "API integration failing",
        "customer_id": "ENT_12345",
        "product_tier": "enterprise"
    }
)

# Process with automatic technical consultation
response = await product_agent.process_message(customer_issue)

# The agent will:
# 1. Analyze product aspects (rate limits, configuration)
# 2. Detect need for technical consultation
# 3. Query Technical agent for infrastructure advice
# 4. Synthesize comprehensive solution
```

## Design Patterns

### 1. Expert System Pattern
Product Agent acts as an expert system for product knowledge and configuration.

### 2. Mediator Pattern
Facilitates communication between support and technical domains.

### 3. Strategy Pattern
Different analysis strategies based on message type and context.

### 4. Template Method Pattern
Structured approach to product analysis and solution generation.

## Integration Points

### Support Agent Integration
- Handles product-specific customer issues
- Provides feature analysis and configuration guidance
- Translates customer needs into product recommendations

### Technical Agent Integration
- Collaborates on implementation requirements
- Provides product context for technical solutions
- Ensures technical feasibility of product recommendations

### Orchestrator Integration
- Participates in dynamic agent routing
- Provides product capability information
- Supports health monitoring and performance tracking

## Best Practices

### 1. Product Knowledge Management
- Keep product information current and accurate
- Understand customer use cases and workflows
- Maintain awareness of product roadmap and limitations
- Document configuration best practices

### 2. Collaboration Strategies
- Identify when technical expertise is needed
- Provide sufficient context for technical consultations
- Synthesize multi-agent responses effectively
- Maintain customer focus throughout collaboration

### 3. Solution Quality
- Consider customer's product tier and permissions
- Provide step-by-step implementation guidance
- Include relevant warnings and limitations
- Suggest validation and testing approaches

## Troubleshooting

### Common Issues

1. **Configuration Conflicts**
   ```python
   # Analyze configuration compatibility
   def check_config_compatibility(self, settings: Dict[str, Any]) -> List[str]:
       conflicts = []
       # Implementation for detecting conflicts
       return conflicts
   ```

2. **Integration Challenges**
   ```python
   # Assess integration requirements
   def assess_integration_requirements(self, integration_type: str) -> Dict[str, Any]:
       requirements = {
           "api_version": "v2.1",
           "authentication": "OAuth 2.0",
           "rate_limits": "1000 requests/hour"
       }
       return requirements
   ```

3. **Feature Limitations**
   ```python
   # Document known limitations
   def get_feature_limitations(self, feature_name: str) -> List[str]:
       limitations = self.feature_database.get(feature_name, {}).get("limitations", [])
       return limitations
   ```

## Extension Points

### Custom Product Analysis
```python
class CustomProductAgent(ProductAgent):
    def __init__(self):
        super().__init__()
        self.capabilities.extend([
            "custom_feature_analysis",
            "specialized_integration_support"
        ])
    
    async def analyze_custom_feature(self, feature_request: str) -> Dict[str, Any]:
        """Perform specialized feature analysis"""
        # Custom analysis logic
        return analysis_results
```

### Advanced Configuration Management
```python
async def optimize_configuration(self, current_config: Dict[str, Any], goals: List[str]) -> Dict[str, Any]:
    """Optimize product configuration for specific goals"""
    optimization_prompt = f"""
    Current Configuration: {current_config}
    Optimization Goals: {goals}
    
    Provide optimized configuration recommendations.
    """
    
    # Generate optimized configuration
    return optimized_config
```

## Testing Considerations

### Unit Tests
```python
@pytest.mark.asyncio
async def test_product_agent_feature_analysis():
    """Test product feature analysis capability"""
    agent = ProductAgent()
    
    message = Message(
        content="Explain the advanced reporting features",
        source_agent="Support",
        target_agent="Product"
    )
    
    response = await agent.process_message(message)
    
    assert "reporting" in response.content.lower()
    assert response.confidence > 0.7
    assert response.source_agent == "Product"
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_product_technical_collaboration():
    """Test collaboration between Product and Technical agents"""
    product_agent = ProductAgent()
    technical_agent = TechnicalAgent()
    
    product_agent.register_agent("Technical", technical_agent)
    
    message = Message(
        content="Customer needs high-performance data processing setup",
        source_agent="Support",
        target_agent="Product",
        message_type="handoff"
    )
    
    response = await product_agent.process_message(message)
    
    # Should involve technical consultation
    assert "performance" in response.content.lower()
    assert response.metadata.get("collaboration_agents")
```

## Performance Considerations

### Response Optimization
- Cache common product information
- Optimize prompt engineering for faster responses
- Use appropriate temperature settings for factual accuracy
- Implement efficient message processing patterns

### Collaboration Efficiency
- Minimize unnecessary agent handoffs
- Use structured communication patterns
- Implement proper timeout handling
- Monitor collaboration success rates

### Knowledge Management
- Maintain up-to-date product information
- Implement efficient knowledge retrieval
- Use version-specific guidance appropriately
- Regular validation of product recommendations

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