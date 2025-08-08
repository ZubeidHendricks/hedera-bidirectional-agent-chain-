from typing import Optional, Literal
from agents.base_agent import BaseAgent
from models.message import Message, Response

class SupportAgent(BaseAgent):
    """
    Customer Support Agent specialized in customer communication and coordination.
    Acts as the primary interface for users and orchestrates other agents.
    """
    
    def __init__(self):
        super().__init__(
            name="Support",
            expertise="customer_communication",
            system_prompt="""
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
        )
    
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