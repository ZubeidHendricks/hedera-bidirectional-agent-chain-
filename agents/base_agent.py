from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Literal
import os
import asyncio
from datetime import datetime
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from models.message import (
    Message, Response
)

class BaseAgent(ABC):
    """
    Enhanced BaseAgent using latest Google GenAI SDK and LangGraph patterns.
    Implements bidirectional agent communication with modern AI orchestration.
    """
    
    def __init__(
        self, 
        name: str, 
        expertise: str, 
        system_prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        self.name = name
        self.expertise = expertise
        self.system_prompt = system_prompt
        self.conversation_history: List[Message] = []
        self.capabilities = [expertise]
        self.agent_registry: Dict[str, 'BaseAgent'] = {}
        self.model_name = model_name or os.getenv(f"{name.upper()}_AGENT_MODEL", "gemini-2.0-flash-exp")
        self.temperature = float(os.getenv(f"{name.upper()}_AGENT_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("MAX_TOKENS_PER_RESPONSE", max_tokens))
        self._initialize_genai_client()
        
        print(f"🤖 {self.name} Agent initialized with:")
        print(f"   - Expertise: {expertise}")
        print(f"   - Model: {self.model_name}")
        print(f"   - Temperature: {self.temperature}")
    
    def _initialize_genai_client(self):
        """Initialize Google GenAI client with environment configuration"""
        try:
            use_vertex_ai = os.getenv("USE_VERTEX_AI", "false").lower() == "true"
            
            if use_vertex_ai:
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                
                if not project_id:
                    raise ValueError("GOOGLE_CLOUD_PROJECT_ID is required for Vertex AI")
                
                self.client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location
                )
                print(f"   - Using Vertex AI (Project: {project_id}, Location: {location})")
                
            else:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is required for Gemini Developer API")
                
                self.client = genai.Client(api_key=api_key)
                print(f"   - Using Gemini Developer API")
                
        except Exception as e:
            print(f"❌ Error initializing GenAI client: {e}")
            raise
    
    async def process_message(self, message: Message) -> Response:
        """Process incoming message and return response"""
        print(f"\n🔄 {self.name} processing message from {message.source_agent}")
        print(f"📝 Message: {message.content}")
        
        self.conversation_history.append(message)
        
        try:
            response = await self._generate_response(message)
            print(f"💬 {self.name} response: {response.content}")
            return response
            
        except Exception as e:
            print(f"❌ Error in {self.name}: {e}")
            return Response(
                content=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                source_agent=self.name,
                target_agent=message.source_agent,
                confidence=0.1,
                needs_clarification=False
            )
    
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
- If you need specific expertise from another agent, use: "HANDOFF_REQUEST: [agent_name] - [specific question or task]"
- If you have sufficient information to provide a complete answer, provide it directly
- Be concise, professional, and accurate
- Consider the full conversation context when responding
- If unsure, acknowledge limitations and suggest next steps

AVAILABLE AGENT CONNECTIONS:
{', '.join(self.agent_registry.keys()) if self.agent_registry else 'None'}

RESPONSE:
"""
        
        try:
            generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                candidate_count=1
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )
            
            content = response.text.strip()
            
            needs_handoff, target_agent, handoff_query = self._parse_handoff_request(content)
            
            if needs_handoff and target_agent in self.agent_registry:
                print(f"🔀 {self.name} requesting handoff to {target_agent}")
                
                handoff_response = await self._handle_handoff(target_agent, handoff_query, message)
                if handoff_response:
                    return await self._synthesize_response(content, handoff_response, message)
            
            return Response(
                content=content,
                source_agent=self.name,
                target_agent=message.source_agent,
                needs_clarification=needs_handoff and target_agent not in self.agent_registry,
                confidence=self._calculate_confidence(content),
                reasoning=f"Generated response using {self.model_name} with {self.expertise} expertise"
            )
            
        except Exception as e:
            print(f"❌ Error generating response with GenAI: {e}")
            return Response(
                content=f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                source_agent=self.name,
                target_agent=message.source_agent,
                needs_clarification=False,
                confidence=0.1
            )
    
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
    
    async def _handle_handoff(self, target_agent: str, query: str, original_message: Message) -> Optional[Response]:
        """Handle handoff to another agent"""
        if target_agent not in self.agent_registry:
            print(f"❌ Target agent {target_agent} not available")
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

INSTRUCTIONS:
- Synthesize both perspectives into a unified, comprehensive response
- Maintain your {self.expertise} perspective while incorporating the expert input
- Provide actionable information to the user
- Be clear, concise, and professional

SYNTHESIZED RESPONSE:
"""
        
        try:
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
            
            return Response(
                content=synthesized_content,
                source_agent=self.name,
                target_agent=original_message.source_agent,
                confidence=min(0.9, (self._calculate_confidence(synthesized_content) + handoff_response.confidence) / 2),
                reasoning=f"Synthesized response combining {self.expertise} analysis with {handoff_response.source_agent} expertise"
            )
            
        except Exception as e:
            print(f"❌ Error synthesizing response: {e}")
            return handoff_response
    
    def _calculate_confidence(self, content: str) -> float:
        """Calculate confidence score based on response characteristics"""
        base_confidence = 0.8
        if len(content) < 50:
            base_confidence -= 0.2
        if "I don't know" in content.lower() or "uncertain" in content.lower():
            base_confidence -= 0.3
        if "error" in content.lower() or "problem" in content.lower():
            base_confidence -= 0.2
        if len(content.split('.')) > 3:
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _build_context(self) -> str:
        """Build context from conversation history"""
        if not self.conversation_history:
            return "No prior conversation in this session."
        
        context_parts = []
        recent_messages = self.conversation_history[-8:]
        
        for msg in recent_messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S") if hasattr(msg, 'timestamp') else "unknown"
            context_parts.append(f"[{timestamp}] {msg.source_agent} → {msg.target_agent}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def register_agent(self, agent_name: str, agent_instance: 'BaseAgent'):
        """Register another agent for bidirectional communication"""
        self.agent_registry[agent_name] = agent_instance
        print(f"🔗 {self.name} Agent registered connection to {agent_name}")
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return self.capabilities
    
    def can_handle(self, message: Message) -> bool:
        """Check if agent can handle the message based on expertise"""
        content_lower = message.content.lower()
        return any(
            capability.lower() in content_lower 
            for capability in self.capabilities
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
            "name": self.name,
            "expertise": self.expertise,
            "capabilities": self.capabilities,
            "model": self.model_name,
            "temperature": self.temperature,
            "connected_agents": list(self.agent_registry.keys()),
            "conversation_history_length": len(self.conversation_history),
            "system_prompt_preview": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt
        }
    
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
                "client_initialized": False,
                "connections": len(self.agent_registry)
            }
    
    @abstractmethod
    async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
        """Query another agent for information - implemented by concrete agents"""
        pass