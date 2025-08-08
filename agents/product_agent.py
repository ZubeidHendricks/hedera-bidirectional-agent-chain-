from typing import Optional, Literal
from agents.base_agent import BaseAgent
from models.message import Message, Response

class ProductAgent(BaseAgent):
    """
    Product Expert Agent specialized in product knowledge and feature expertise.
    Provides comprehensive product insights within the bidirectional agent chain.
    """
    
    def __init__(self):
        super().__init__(
            name="Product",
            expertise="product_knowledge",
            system_prompt="""
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
        )
        
        self.capabilities.extend([
            "feature_analysis",
            "product_configuration",
            "compatibility_assessment",
            "integration_guidance",
            "roadmap_insights",
            "licensing_consultation"
        ])
    
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
    
    def _assess_technical_dependency(self, product_content: str, original_query: str) -> bool:
        """Assess if technical implementation details would enhance the product solution"""
        
        technical_indicators = [
            "implementation", "integration", "api", "configuration", "setup",
            "installation", "deployment", "performance", "security", "infrastructure"
        ]
        
        combined_content = f"{product_content} {original_query}".lower()
        
        return any(indicator in combined_content for indicator in technical_indicators)
    
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