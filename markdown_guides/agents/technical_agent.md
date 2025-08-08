# Technical Agent - System Diagnostics Expert Guide

## Overview
The `TechnicalAgent` is a specialized expert agent focused on system diagnostics, technical problem-solving, and infrastructure analysis within the bidirectional agent chaining system. It provides deep technical expertise and collaborates intelligently with product and support agents to deliver comprehensive technical solutions.

## File Purpose
This technical expert provides:
- **System Diagnostics**: Advanced troubleshooting and root cause analysis
- **Technical Problem-Solving**: Complex technical issue resolution
- **Infrastructure Analysis**: Performance, security, and architecture evaluation
- **Implementation Guidance**: Detailed technical implementation recommendations
- **Product Integration**: Technical context for product-related solutions
- **Error Analysis**: Comprehensive error diagnosis and remediation

## Class Architecture

### Core Dependencies
```python
from typing import Optional, Literal
from agents.base_agent import BaseAgent
from models.message import Message, Response
```

### Class Definition
```python
class TechnicalAgent(BaseAgent):
    """
    Technical Support Agent specialized in system diagnostics and technical problem-solving.
    Provides deep technical analysis and solutions within the bidirectional agent chain.
    """
```

## Initialization and System Prompt

### Constructor
```python
def __init__(self):
    super().__init__(
        name="Technical",
        expertise="system_diagnostics",
        system_prompt="""[Comprehensive technical expertise prompt]"""
    )
    
    self.capabilities.extend([
        "error_diagnosis",
        "performance_analysis", 
        "security_assessment",
        "system_architecture",
        "troubleshooting",
        "infrastructure_analysis"
    ])
```

### Comprehensive Technical Expertise Definition
```python
system_prompt = """
You are an Expert Technical Support Agent with deep expertise in:

PRIMARY TECHNICAL CAPABILITIES:
- Advanced system diagnostics and troubleshooting
- Error analysis and root cause identification
- Performance optimization and system monitoring
- Security assessment and vulnerability analysis
- Infrastructure and architecture evaluation
- Technical implementation guidance

BIDIRECTIONAL AGENT CHAIN ROLE:
- Provide technical expertise to Support and Product agents
- Analyze complex technical problems with precision
- Offer detailed technical solutions and implementation steps
- Collaborate with Product agents for feature-specific technical issues
- Translate complex technical concepts for Support agent communication

DIAGNOSTIC METHODOLOGY:
- Systematic problem analysis using structured approaches
- Root cause analysis with evidence-based reasoning
- Performance impact assessment
- Security and stability considerations
- Scalability and maintainability evaluation

COLLABORATION PATTERNS:
- When product-specific technical details are needed → Product Agent
- When implementation requires product configuration knowledge → Product Agent
- Always provide technical depth while remaining accessible for translation

TECHNICAL COMMUNICATION STANDARDS:
- Provide precise technical terminology with explanations
- Include step-by-step troubleshooting procedures
- Offer multiple solution approaches when applicable
- Specify prerequisites and dependencies
- Include relevant warnings about risks or limitations
- Suggest preventive measures and monitoring approaches

SOLUTION QUALITY:
- Evidence-based recommendations
- Consider both immediate fixes and long-term solutions
- Address potential side effects or complications
- Provide verification steps to confirm resolution
- Include logging and monitoring recommendations
"""
```

**Key Elements**:
- Deep technical expertise across multiple domains
- Systematic diagnostic methodology
- Clear collaboration patterns with other agents
- Evidence-based solution approaches
- Emphasis on both immediate and long-term solutions

## Enhanced Technical Collaboration

### Intelligent Product Consultation
```python
async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
    """
    Query another agent for specialized information.
    Enhanced for technical collaboration patterns.
    """
    if target_agent not in self.agent_registry:
        print(f"❌ Technical Agent: {target_agent} not available")
        return None
    
    print(f"🔧 Technical Agent consulting with {target_agent}")
    print(f"   Technical query: {query}")
    
    consultation_message = Message(
        content=query,
        source_agent=self.name,
        target_agent=target_agent,
        message_type="query",
        context={
            "consultation_type": "technical_expertise",
            "requires_technical_detail": True,
            "analysis_depth": "comprehensive",
            "urgency": "normal"
        }
    )
    
    try:
        response = await self.agent_registry[target_agent].process_message(consultation_message)
        print(f"📥 Technical Agent received consultation from {target_agent}")
        return response
    except Exception as e:
        print(f"❌ Technical Agent: Error consulting with {target_agent}: {e}")
        return None
```

## Specialized Message Processing

### Enhanced Technical Response Generation
```python
async def _generate_response(self, message: Message) -> Response:
    """
    Enhanced technical response generation with systematic analysis.
    Provides structured technical solutions with proper depth.
    """
    
    if message.message_type == "handoff" and message.source_agent == "Support":
        print(f"🔧 Technical Agent conducting analysis for customer issue...")
        return await self._handle_customer_technical_issue(message)
        
    elif message.source_agent == "Product":
        print(f"🔧 Technical Agent analyzing product-technical integration...")
        return await self._handle_product_technical_query(message)
        
    else:
        print(f"🔧 Technical Agent performing standard technical analysis...")
        return await super()._generate_response(message)
```

### Customer Technical Issue Analysis
```python
async def _handle_customer_technical_issue(self, message: Message) -> Response:
    """Handle technical issues escalated from customer support"""
    
    original_query = ""
    customer_context = ""
    
    if hasattr(message, 'context') and message.context:
        original_query = message.context.get('original_query', '')
        customer_context = f"Customer reported: {original_query}\n" if original_query else ""
    
    enhanced_prompt = f"""
TECHNICAL ANALYSIS REQUEST FROM SUPPORT TEAM:

{customer_context}
SUPPORT TEAM ANALYSIS REQUEST:
{message.content}

COMPREHENSIVE TECHNICAL ANALYSIS REQUIRED:

1. PROBLEM IDENTIFICATION:
   - Analyze the technical symptoms and indicators
   - Identify potential root causes
   - Classify the issue type and severity

2. DIAGNOSTIC APPROACH:
   - Recommend specific diagnostic steps
   - Identify what information/logs would be helpful
   - Suggest tools or methods for investigation

3. SOLUTION RECOMMENDATIONS:
   - Provide immediate remediation steps if applicable
   - Offer both short-term fixes and long-term solutions
   - Consider different approaches based on environment/constraints

4. IMPLEMENTATION GUIDANCE:
   - Detail step-by-step procedures
   - Highlight prerequisites and dependencies
   - Include verification steps

5. RISK ASSESSMENT:
   - Identify potential risks or side effects
   - Suggest precautions and backup procedures
   - Recommend monitoring during implementation

6. PREVENTION MEASURES:
   - Suggest preventive measures for future occurrences
   - Recommend monitoring or alerting improvements

Provide your analysis in a structured format that can be easily communicated to customers.
"""
    
    analysis_message = Message(
        content=enhanced_prompt,
        source_agent="internal",
        target_agent=self.name,
        message_type="analysis"
    )
    
    technical_response = await super()._generate_response(analysis_message)
    
    # Check if product consultation is needed
    needs_product_info = self._assess_product_dependency(technical_response.content, message.content)
    
    if needs_product_info and "Product" in self.agent_registry:
        print(f"🔀 Technical Agent consulting Product for feature-specific details...")
        
        product_query = self._generate_product_query(technical_response.content, message.content)
        product_response = await self.query_other_agent("Product", product_query)
        
        if product_response:
            return await self._synthesize_technical_product_response(
                technical_response, product_response, message
            )
    
    return technical_response
```

## Product-Technical Integration

### Product Dependency Assessment
```python
def _assess_product_dependency(self, technical_content: str, original_query: str) -> bool:
    """Assess if product-specific information would enhance the technical solution"""
    
    product_indicators = [
        "feature", "configuration", "setting", "option", "capability",
        "version", "upgrade", "compatibility", "limitation", "support"
    ]
    
    combined_content = f"{technical_content} {original_query}".lower()
    
    return any(indicator in combined_content for indicator in product_indicators)
```

### Product Query Generation
```python
def _generate_product_query(self, technical_analysis: str, original_query: str) -> str:
    """Generate a focused query for product agent based on technical analysis"""
    
    return f"""
TECHNICAL-PRODUCT CONSULTATION REQUEST:

ORIGINAL ISSUE:
{original_query}

TECHNICAL ANALYSIS SUMMARY:
{technical_analysis[:500]}...

PRODUCT INFORMATION NEEDED:
- Feature capabilities and limitations relevant to this technical issue
- Configuration options that might affect the technical solution
- Known product behaviors or constraints that impact implementation
- Version-specific considerations
- Product roadmap items that might address this issue

Please provide product-specific context that would help refine the technical solution.
"""
```

### Technical-Product Response Synthesis
```python
async def _synthesize_technical_product_response(
    self, 
    technical_response: Response, 
    product_response: Response, 
    original_message: Message
) -> Response:
    """Synthesize technical analysis with product insights"""
    
    synthesis_prompt = f"""
COMPREHENSIVE TECHNICAL SOLUTION SYNTHESIS:

Combine your technical analysis with product-specific insights to provide a complete solution.

YOUR TECHNICAL ANALYSIS:
{technical_response.content}

PRODUCT TEAM INSIGHTS:
{product_response.content}

SYNTHESIS REQUIREMENTS:
1. Integrate product constraints and capabilities into technical recommendations
2. Adjust technical solutions based on product-specific information
3. Provide implementation steps that account for both technical and product considerations
4. Highlight any conflicts between technical best practices and product limitations
5. Offer alternative approaches if product constraints limit optimal technical solutions

COMPREHENSIVE TECHNICAL-PRODUCT SOLUTION:
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
            technical_response.confidence + 
            product_response.confidence + 
            0.15 
        ) / 2)
        
        return Response(
            content=synthesized_content,
            source_agent=self.name,
            target_agent=original_message.source_agent,
            confidence=confidence_score,
            reasoning="Comprehensive technical solution integrating product expertise",
            metadata={
                "analysis_type": "technical_product_synthesis",
                "collaboration_agents": [product_response.source_agent],
                "technical_confidence": technical_response.confidence,
                "product_confidence": product_response.confidence
            }
        )
        
    except Exception as e:
        print(f"❌ Technical Agent: Error synthesizing responses: {e}")
        return technical_response
```

## Product-Technical Query Handling

### Product-Technical Integration Analysis
```python
async def _handle_product_technical_query(self, message: Message) -> Response:
    """Handle technical queries from product agent"""
    
    enhanced_prompt = f"""
PRODUCT-TECHNICAL INTEGRATION ANALYSIS:

PRODUCT TEAM REQUEST:
{message.content}

TECHNICAL ANALYSIS FOCUS:
- System-level implementation considerations
- Performance and scalability implications
- Security and compliance aspects
- Infrastructure requirements
- Integration patterns and best practices
- Potential technical limitations or constraints

Provide detailed technical insights that complement product knowledge.
"""
    
    analysis_message = Message(
        content=enhanced_prompt,
        source_agent="internal",
        target_agent=self.name,
        message_type="analysis"
    )
    
    return await super()._generate_response(analysis_message)
```

## Technical Capabilities and Diagnostics

### Comprehensive Diagnostic Capabilities
```python
def get_diagnostic_capabilities(self) -> dict:
    """Get detailed information about diagnostic capabilities"""
    return {
        "core_expertise": self.expertise,
        "diagnostic_areas": [
            "System Performance Analysis",
            "Error Log Analysis", 
            "Security Vulnerability Assessment",
            "Infrastructure Evaluation",
            "API and Integration Debugging",
            "Database Performance Optimization",
            "Network Connectivity Issues",
            "Application Architecture Review"
        ],
        "analysis_methods": [
            "Root Cause Analysis",
            "Performance Profiling",
            "Security Scanning",
            "Load Testing Analysis",
            "Failure Mode Analysis"
        ],
        "collaboration_patterns": {
            "with_support": "Customer-focused technical solutions",
            "with_product": "Feature-technical integration analysis"
        }
    }
```

## Usage Examples

### Basic Technical Diagnosis
```python
# Initialize Technical Agent
technical_agent = TechnicalAgent()

# Customer technical issue
technical_issue = Message(
    content="Customer reporting 500 errors during file uploads, affecting 15% of users",
    source_agent="Support",
    target_agent="Technical",
    message_type="handoff",
    context={
        "original_query": "File upload fails with server error",
        "customer_tier": "enterprise",
        "severity": "high"
    }
)

response = await technical_agent.process_message(technical_issue)
```

### Multi-Agent Technical Analysis
```python
# Setup agent network
technical_agent = TechnicalAgent()
product_agent = ProductAgent()
support_agent = SupportAgent()

# Register agents
technical_agent.register_agent("Product", product_agent)
technical_agent.register_agent("Support", support_agent)

# Complex technical-product issue
complex_technical_issue = Message(
    content="Database performance degradation during peak hours, possibly related to new reporting features",
    source_agent="Support",
    target_agent="Technical",
    message_type="handoff"
)

# Process with automatic product consultation
response = await technical_agent.process_message(complex_technical_issue)

# The agent will:
# 1. Perform technical analysis (performance, database issues)
# 2. Detect need for product consultation (reporting features)
# 3. Query Product agent for feature-specific context
# 4. Synthesize comprehensive technical solution
```

### Direct Product Consultation
```python
# Technical agent querying product expert
product_response = await technical_agent.query_other_agent(
    "Product",
    "What are the database query patterns and indexing requirements for the new analytics dashboard feature?"
)

# Integration-focused consultation
integration_response = await technical_agent.query_other_agent(
    "Product", 
    "Customer needs API rate limiting configuration for high-volume data ingestion - what are the product constraints and recommended settings?"
)
```

## Design Patterns

### 1. Expert System Pattern
Technical Agent acts as an expert system for technical knowledge and diagnostics.

### 2. Strategy Pattern
Different diagnostic strategies based on issue type and system domain.

### 3. Template Method Pattern
Structured approach to technical analysis and solution generation.

### 4. Chain of Responsibility Pattern
Technical analysis can be passed to product experts when needed.

## Integration Points

### Support Agent Integration
- Handles technical aspects of customer issues
- Provides detailed technical analysis and solutions
- Translates complex technical problems for customer communication

### Product Agent Integration
- Collaborates on product-technical integration issues
- Provides technical context for product recommendations
- Ensures technical feasibility of product solutions

### Orchestrator Integration
- Participates in technical routing decisions
- Provides technical capability information
- Supports performance monitoring and health checks

## Best Practices

### 1. Technical Analysis
- Use systematic diagnostic approaches
- Provide evidence-based recommendations
- Consider both immediate and long-term solutions
- Include proper risk assessment and mitigation

### 2. Collaboration Strategies
- Identify when product context is needed
- Provide sufficient technical detail for product consultation
- Synthesize technical and product perspectives effectively
- Maintain technical accuracy throughout collaboration

### 3. Solution Quality
- Offer multiple solution approaches when applicable
- Include step-by-step implementation guidance
- Provide verification and monitoring recommendations
- Consider security and performance implications

## Troubleshooting

### Common Issues

1. **Performance Diagnostics**
   ```python
   def diagnose_performance_issue(self, symptoms: Dict[str, Any]) -> Dict[str, Any]:
       diagnostic_steps = {
           "cpu_usage": "Monitor CPU utilization patterns",
           "memory_usage": "Analyze memory consumption and leaks",
           "database_queries": "Review slow query logs and execution plans",
           "network_latency": "Measure network response times"
       }
       return diagnostic_steps
   ```

2. **Error Analysis**
   ```python
   def analyze_error_patterns(self, error_logs: List[str]) -> Dict[str, Any]:
       analysis = {
           "error_frequency": self._calculate_error_frequency(error_logs),
           "common_patterns": self._identify_error_patterns(error_logs),
           "root_causes": self._suggest_root_causes(error_logs),
           "remediation_steps": self._recommend_fixes(error_logs)
       }
       return analysis
   ```

3. **Security Assessment**
   ```python
   def assess_security_vulnerabilities(self, system_config: Dict[str, Any]) -> List[Dict[str, Any]]:
       vulnerabilities = []
       # Implementation for security assessment
       return vulnerabilities
   ```

## Extension Points

### Custom Diagnostic Tools
```python
class AdvancedTechnicalAgent(TechnicalAgent):
    def __init__(self):
        super().__init__()
        self.capabilities.extend([
            "automated_log_analysis",
            "predictive_failure_detection",
            "advanced_security_scanning"
        ])
    
    async def run_automated_diagnostics(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Run automated diagnostic analysis"""
        # Custom diagnostic implementation
        return diagnostic_results
```

### Performance Optimization
```python
async def optimize_system_performance(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Provide system performance optimization recommendations"""
    optimization_prompt = f"""
    System Performance Metrics: {performance_metrics}
    
    Provide detailed optimization recommendations including:
    - Infrastructure improvements
    - Configuration optimizations
    - Code-level optimizations
    - Monitoring enhancements
    """
    
    # Generate optimization recommendations
    return optimization_plan
```

## Testing Considerations

### Unit Tests
```python
@pytest.mark.asyncio
async def test_technical_agent_diagnosis():
    """Test technical diagnostic capabilities"""
    agent = TechnicalAgent()
    
    message = Message(
        content="Server returning 500 errors under load",
        source_agent="Support",
        target_agent="Technical",
        message_type="handoff"
    )
    
    response = await agent.process_message(message)
    
    assert "500" in response.content
    assert response.confidence > 0.7
    assert "diagnostic" in response.content.lower() or "analysis" in response.content.lower()
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_technical_product_collaboration():
    """Test collaboration between Technical and Product agents"""
    technical_agent = TechnicalAgent()
    product_agent = ProductAgent()
    
    technical_agent.register_agent("Product", product_agent)
    
    message = Message(
        content="Database performance issues with new reporting feature",
        source_agent="Support",
        target_agent="Technical",
        message_type="handoff"
    )
    
    response = await technical_agent.process_message(message)
    
    # Should involve product consultation for feature context
    assert "reporting" in response.content.lower()
    assert response.metadata.get("collaboration_agents")
    assert response.confidence > 0.6
```

## Performance Considerations

### Diagnostic Efficiency
- Optimize diagnostic algorithms for speed
- Cache common diagnostic patterns
- Use parallel analysis where applicable
- Implement proper timeout handling

### Collaboration Optimization
- Minimize unnecessary product consultations
- Use structured communication for efficiency
- Implement proper error handling and retries
- Monitor collaboration success rates

### Solution Quality
- Balance thorough analysis with response time
- Provide progressive disclosure of technical details
- Use appropriate technical depth for audience
- Ensure scalable solution recommendations

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