from typing import Optional, Literal
from agents.base_agent import BaseAgent
from models.message import Message, Response

class TechnicalAgent(BaseAgent):
    """
    Technical Support Agent specialized in system diagnostics and technical problem-solving.
    Provides deep technical analysis and solutions within the bidirectional agent chain.
    """
    
    def __init__(self):
        super().__init__(
            name="Technical",
            expertise="system_diagnostics",
            system_prompt="""
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
        )
        
        self.capabilities.extend([
            "error_diagnosis",
            "performance_analysis", 
            "security_assessment",
            "system_architecture",
            "troubleshooting",
            "infrastructure_analysis"
        ])
    
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
    
    def _assess_product_dependency(self, technical_content: str, original_query: str) -> bool:
        """Assess if product-specific information would enhance the technical solution"""
        
        product_indicators = [
            "feature", "configuration", "setting", "option", "capability",
            "version", "upgrade", "compatibility", "limitation", "support"
        ]
        
        combined_content = f"{technical_content} {original_query}".lower()
        
        return any(indicator in combined_content for indicator in product_indicators)
    
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