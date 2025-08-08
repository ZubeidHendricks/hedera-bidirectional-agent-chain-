import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from agents.support_agent import SupportAgent
from agents.technical_agent import TechnicalAgent
from agents.product_agent import ProductAgent
from agents.hedera_rag_agent import HederaRAGAgent
from models.message import (
    Message, UserRequest, ChainResult, Response
)

class ChainOrchestrator:
    """
    Dynamic Bidirectional Agent Chain Orchestrator.
    Implements true bidirectional agent chaining with confidence-based iteration.
    Agents communicate and collaborate until confidence threshold (0.7) is reached.
    """
    
    def __init__(self):
        print("🏗️ Initializing Modern Bidirectional Agent Chain with LangGraph...")
        
        self._load_configuration()
        
        self._initialize_agents()
        
        self._setup_agent_network()
        
        self._initialize_dynamic_routing()
        
        self.conversation_flows = {}
        self.performance_metrics = {}
        
        self.active_conversations = {}
        
        print("✅ Modern Bidirectional Agent Chain initialized successfully!")
        print(f"   - Confidence threshold: {self.confidence_threshold}")
        print(f"   - Max chain hops: {self.max_chain_hops}")
        print(f"   - Agent timeout: {self.agent_timeout}s")
        print(f"   - Dynamic routing: Enabled")
    
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
        self.enable_content_analysis_routing = os.getenv("ENABLE_CONTENT_ANALYSIS_ROUTING", "true").lower() == "true"
    
    def _initialize_agents(self):
        """Initialize all agents with enhanced capabilities"""
        print("🤖 Initializing enhanced agent network...")
        
        try:
            self.support_agent = SupportAgent()
            self.technical_agent = TechnicalAgent()
            self.product_agent = ProductAgent()
            self.hedera_rag_agent = HederaRAGAgent()
            self.agents = {
                "Support": self.support_agent,
                "Technical": self.technical_agent,
                "Product": self.product_agent,
                "Hedera_RAG": self.hedera_rag_agent
            }
            
            print("✅ All agents initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            raise
    
    def _setup_agent_network(self):
        """Set up the bidirectional agent network with full connectivity"""
        print("🔗 Setting up bidirectional agent network...")
        
        try:
            # Register all agents with each other for bidirectional communication
            self.support_agent.register_agent("Technical", self.technical_agent)
            self.support_agent.register_agent("Product", self.product_agent)
            self.support_agent.register_agent("Hedera_RAG", self.hedera_rag_agent)
            
            self.technical_agent.register_agent("Support", self.support_agent)
            self.technical_agent.register_agent("Product", self.product_agent)
            self.technical_agent.register_agent("Hedera_RAG", self.hedera_rag_agent)
            
            self.product_agent.register_agent("Support", self.support_agent)
            self.product_agent.register_agent("Technical", self.technical_agent)
            self.product_agent.register_agent("Hedera_RAG", self.hedera_rag_agent)
            
            self.hedera_rag_agent.register_agent("Support", self.support_agent)
            self.hedera_rag_agent.register_agent("Technical", self.technical_agent)
            self.hedera_rag_agent.register_agent("Product", self.product_agent)
            
            print("✅ Bidirectional agent network established!")
            print("   Network topology:")
            print("   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐")
            print("   │   Support   │ ←→  │  Technical  │ ←→  │ Hedera_RAG  │")
            print("   │    Agent    │     │    Agent    │     │    Agent    │")
            print("   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘")
            print("          │                   │                   │")
            print("          └─────────┬─────────┴─────────┬─────────┘")
            print("                    │                   │")
            print("                    ↓                   ↓")
            print("              ┌─────────────────────────────────────┐")
            print("              │           Product Agent             │")
            print("              └─────────────────────────────────────┘")
            
        except Exception as e:
            print(f"❌ Error setting up agent network: {e}")
            raise
    
    def _initialize_dynamic_routing(self):
        """Initialize dynamic routing system for true bidirectional chaining"""
        print("🧠 Initializing dynamic bidirectional routing...")
        
        try:
            self.routing_patterns = {
                "technical": ["error", "bug", "crash", "performance", "system", "database", "api", "server", "timeout", "memory", "cpu"],
                "product": ["feature", "pricing", "plan", "upgrade", "license", "integration", "configuration", "settings", "workflow"],
                "support": ["help", "question", "how to", "tutorial", "guide", "assistance", "problem", "issue", "need"],
                "hedera": ["hedera", "blockchain", "hashgraph", "hcs", "hfs", "hts", "consensus", "smart contracts", "dlt", "cryptocurrency", "hbar", "mainnet", "testnet", "sdk", "api", "developer", "defi", "nft", "enterprise", "supply chain"]
            }
            
            self.agent_expertise = {
                "Support": ["customer_communication", "issue_triage", "response_synthesis", "coordination"],
                "Technical": ["system_diagnostics", "troubleshooting", "performance_optimization", "architecture"],
                "Product": ["product_knowledge", "feature_expertise", "configuration", "workflow_optimization"],
                "Hedera_RAG": ["hedera_knowledge", "blockchain_analysis", "service_documentation", "developer_guidance", "integration_examples", "rag_retrieval"]
            }
            
            self.routing_history = {}
            
            print("✅ Dynamic bidirectional routing initialized")
            
        except Exception as e:
            print(f"❌ Error initializing dynamic routing: {e}")
            raise
    
    def _analyze_content_for_routing(self, content: str) -> Dict[str, float]:
        """Analyze content to determine optimal agent routing"""
        content_lower = content.lower()
        routing_scores = {
            "Support": 0.1,
            "Technical": 0.1,
            "Product": 0.1,
            "Hedera_RAG": 0.1
        }
        
        for domain, keywords in self.routing_patterns.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if domain == "technical":
                routing_scores["Technical"] += score * 0.2
            elif domain == "product":
                routing_scores["Product"] += score * 0.2
            elif domain == "support":
                routing_scores["Support"] += score * 0.2
            elif domain == "hedera":
                routing_scores["Hedera_RAG"] += score * 0.3
        
        if "how" in content_lower or "why" in content_lower:
            routing_scores["Support"] += 0.3
            
        if "error" in content_lower or "crash" in content_lower or "performance" in content_lower:
            routing_scores["Technical"] += 0.4
            
        if "feature" in content_lower or "configuration" in content_lower or "plan" in content_lower:
            routing_scores["Product"] += 0.4
            
        if "hedera" in content_lower or "blockchain" in content_lower or "hashgraph" in content_lower:
            routing_scores["Hedera_RAG"] += 0.5
        
        return routing_scores
    
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
        
        if current_agent in routing_scores:
            if confidence_score < 0.5:
                routing_scores[current_agent] *= 1.2
            elif len(conversation_history) > 0:
                routing_scores[current_agent] *= 0.8
        
        if confidence_score >= self.confidence_threshold:
            max_score = max(routing_scores.values())
            if max_score < 0.8:
                return None
        
        if routing_scores:
            selected_agent = max(routing_scores, key=routing_scores.get)
            if routing_scores[selected_agent] > 0.3: 
                return selected_agent
        
        return None
    
    async def _process_agent_interaction(
        self, 
        agent_name: str, 
        message: Message, 
        conversation_state: Dict[str, Any]
    ) -> Response:
        """Process a single agent interaction with detailed logging and consultation tracking"""
        start_time = time.time()
        
        try:
            if self.enable_detailed_flow_logging:
                print(f"\n🔄 {agent_name} Agent Processing:")
                print(f"   📝 Message from {message.source_agent}: {message.content[:100]}{'...' if len(message.content) > 100 else ''}")
                print(f"   🎯 Message Type: {message.message_type}")
                if message.message_type == "handoff":
                    print(f"   🔀 HANDOFF: {message.source_agent} → {agent_name}")
                print(f"   🎯 Context: {message.context}")
            
            agent_instance = self.agents[agent_name]
            
            # Track original agent consultations by monitoring print statements
            original_print = print
            consultation_log = []
            
            def tracking_print(*args, **kwargs):
                output = ' '.join(str(arg) for arg in args)
                # Capture agent consultation patterns
                if "🔀" in output and "requesting handoff to" in output:
                    parts = output.split("requesting handoff to")
                    if len(parts) > 1:
                        consulted_agent = parts[1].strip()
                        consultation_log.append({
                            "consulted_agent": consulted_agent,
                            "type": "handoff_request"
                        })
                elif "🔄" in output and "collaborating with" in output:
                    parts = output.split("collaborating with")
                    if len(parts) > 1:
                        consulted_agent = parts[1].strip()
                        consultation_log.append({
                            "consulted_agent": consulted_agent,
                            "type": "collaboration"
                        })
                elif "📥" in output and "received expert input from" in output:
                    parts = output.split("received expert input from")
                    if len(parts) > 1:
                        consulted_agent = parts[1].strip()
                        consultation_log.append({
                            "consulted_agent": consulted_agent,
                            "type": "expert_input_received"
                        })
                return original_print(*args, **kwargs)
            
            # Temporarily replace print to capture consultations
            import builtins
            builtins.print = tracking_print
            
            try:
                response = await agent_instance.process_message(message)
            finally:
                # Restore original print
                builtins.print = original_print
            
            processing_time = time.time() - start_time
            if self.enable_performance_monitoring:
                self._track_agent_performance(agent_name, processing_time, response.confidence)
            
            # Add consultation steps to conversation flow
            for consultation in consultation_log:
                if consultation["type"] in ["handoff_request", "collaboration"]:
                    consulted_agent = consultation["consulted_agent"]
                    # Add agents involved
                    conversation_state["agents_involved"].add(consulted_agent)
                    
                    # Create flow step for consultation
                    consultation_step = {
                        "hop": conversation_state["hop_count"] + 0.1,  # Sub-hop
                        "agent": consulted_agent,
                        "action": "consulted",
                        "timestamp": datetime.now(),
                        "confidence": response.confidence * 0.8,  # Estimated confidence for consultation
                        "content_preview": f"Consulted by {agent_name} for specialized knowledge",
                        "source_agent": agent_name,
                        "target_agent": consulted_agent,
                        "message_type": "consultation",
                        "response_attribution": consulted_agent,
                        "agent_usage_count": 1,
                        "is_handoff": True,
                        "handoff_chain": f"{agent_name} → {consulted_agent}",
                        "processing_time": processing_time * 0.3  # Estimated consultation time
                    }
                    conversation_state["conversation_flow"].append(consultation_step)
            
            if self.enable_detailed_flow_logging:
                print(f"   💬 {agent_name} RESPONSE: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")
                print(f"   ⏱️  Processing time: {processing_time:.2f}s")
                print(f"   🎯 Confidence: {response.confidence:.2f}")
                print(f"   🏷️  Attribution: Response by {response.source_agent}")
                if response.suggested_next_agent:
                    print(f"   ➡️  Suggests next agent: {response.suggested_next_agent}")
                if hasattr(response, 'reasoning') and response.reasoning:
                    print(f"   🧠 Reasoning: {response.reasoning}")
            
            return response
            
        except Exception as e:
            print(f"❌ Error in {agent_name} agent: {e}")
            return Response(
                content=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                source_agent=agent_name,
                target_agent=message.source_agent,
                confidence=0.1,
                needs_clarification=True,
                reasoning=f"Error in {agent_name}: {str(e)}"
            )
    

    

    
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
        
        current_avg = metrics["average_confidence"]
        n = metrics["total_requests"]
        metrics["average_confidence"] = ((current_avg * (n - 1)) + confidence) / n
    
    async def process_request(self, user_request: UserRequest) -> Dict[str, Any]:
        """
        Process user request through the dynamic bidirectional agent chain.
        Agents iterate until confidence threshold (0.7) is reached or max hops exceeded.
        """
        start_time = time.time()
        conversation_id = user_request.request_id
        
        print(f"\n🚀 Processing request: {conversation_id}")
        print(f"📝 Query: {user_request.query}")
        print(f"🎯 Target confidence: {self.confidence_threshold}")
        
        try:
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
            
            self.active_conversations[conversation_id] = conversation_state
            
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
            
            print(f"\n🔄 Starting Dynamic Bidirectional Agent Chain")
            print(f"   Initial Agent: {current_agent}")
            print(f"   Confidence Threshold: {self.confidence_threshold}")
            print(f"   Max Hops: {self.max_chain_hops}")
            
            agent_usage_count = {agent: 0 for agent in self.agents.keys()}
            
            while (conversation_state["hop_count"] < self.max_chain_hops and 
                   conversation_state["current_confidence"] < self.confidence_threshold):
                
                conversation_state["hop_count"] += 1
                agent_usage_count[current_agent] += 1
                conversation_state["agents_involved"].add(current_agent)
                
                print(f"\n📍 Hop {conversation_state['hop_count']}: Processing with {current_agent} Agent")
                
                hop_start_time = time.time()
                response = await self._process_agent_interaction(
                    current_agent, 
                    current_message, 
                    conversation_state
                )
                hop_processing_time = time.time() - hop_start_time
                
                conversation_state["current_response"] = response
                conversation_state["current_confidence"] = response.confidence
                
                # Use the hop processing time
                
                flow_step = {
                    "hop": conversation_state["hop_count"],
                    "agent": current_agent,
                    "action": "processed",
                    "timestamp": datetime.now(),
                    "confidence": response.confidence,
                    "content_preview": response.content[:150] + "..." if len(response.content) > 150 else response.content,
                    "source_agent": current_message.source_agent,
                    "target_agent": current_agent,
                    "message_type": current_message.message_type,
                    "response_attribution": response.source_agent,
                    "agent_usage_count": agent_usage_count[current_agent],
                    "is_handoff": current_message.message_type == "handoff",
                    "handoff_chain": f"{current_message.source_agent} → {current_agent}" if current_message.message_type == "handoff" else None,
                    "processing_time": hop_processing_time
                }
                conversation_state["conversation_flow"].append(flow_step)
                
                print(f"   📊 Current confidence: {response.confidence:.2f} (threshold: {self.confidence_threshold})")
                
                if response.confidence >= self.confidence_threshold:
                    print(f"   ✅ Confidence threshold reached! Final response ready.")
                    break
                
                next_agent = self._select_next_agent(
                    current_agent,
                    response.content,
                    conversation_state["conversation_flow"],
                    response.confidence,
                    agent_usage_count
                )
                
                if not next_agent:
                    print(f"   🏁 No suitable next agent found. Ending chain.")
                    break
                
                if next_agent == current_agent:
                    print(f"   🔄 Continuing with {current_agent} for iterative improvement")
                    current_message = Message(
                        content=f"Continue improving this analysis. Previous confidence: {response.confidence:.2f}. Original query: {user_request.query}\n\nPrevious analysis to enhance: {response.content}",
                        source_agent="orchestrator",
                        target_agent=current_agent,
                        message_type="query",
                        context={
                            "request_id": conversation_id,
                            "hop_count": conversation_state["hop_count"],
                            "previous_confidence": response.confidence,
                            "original_query": user_request.query,
                            "iteration_type": "improvement"
                        }
                    )
                else:
                    print(f"   ➡️  Routing to {next_agent} for additional expertise")
                    current_message = Message(
                        content=f"Previous analysis from {current_agent}: {response.content}\n\nOriginal query: {user_request.query}",
                        source_agent=current_agent,
                        target_agent=next_agent,
                        message_type="handoff",
                        context={
                            "request_id": conversation_id,
                            "hop_count": conversation_state["hop_count"],
                            "previous_confidence": response.confidence,
                            "original_query": user_request.query,
                            "agent_usage": agent_usage_count
                        }
                    )
                
                current_agent = next_agent
            
            processing_time = time.time() - start_time
            final_response = conversation_state["current_response"]
            
            if not final_response:
                final_response = Response(
                    content="I apologize, but I was unable to process your request adequately.",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.1
                )
            
            result = ChainResult(
                request_id=conversation_id,
                response=final_response.content,
                agents_involved=list(conversation_state["agents_involved"]),
                conversation_flow=conversation_state["conversation_flow"],
                total_processing_time=processing_time,
                success=True,
                confidence_score=final_response.confidence,
                performance_metrics={
                    "total_hops": conversation_state["hop_count"],
                    "final_confidence": final_response.confidence,
                    "threshold_met": final_response.confidence >= self.confidence_threshold,
                    "agent_performance": self.performance_metrics.copy() if self.enable_performance_monitoring else {}
                }
            )
            
            self.conversation_flows[conversation_id] = conversation_state["conversation_flow"]
            
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
            
            print(f"\n🎉 Dynamic Bidirectional Chain Complete!")
            print(f"   ⏱️  Total processing time: {processing_time:.2f}s")
            print(f"   🔄 Total hops: {conversation_state['hop_count']}")
            print(f"   👥 Agents involved: {', '.join(result.agents_involved)}")
            print(f"   🎯 Overall confidence: {result.confidence_score:.2f}")
            print(f"   ✅ Threshold met: {'Yes' if result.confidence_score >= self.confidence_threshold else 'No'}")
            
            return self._format_result_response(result)
            
        except Exception as e:
            print(f"❌ Error processing request: {e}")
            
            processing_time = time.time() - start_time
            
            error_result = ChainResult(
                request_id=conversation_id,
                response=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                agents_involved=["Support"],
                conversation_flow=conversation_state.get("conversation_flow", []),
                total_processing_time=processing_time,
                success=False,
                confidence_score=0.1,
                error_details=str(e)
            )
            
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
            
            return self._format_result_response(error_result)
    

    
    def _format_result_response(self, result: ChainResult) -> Dict[str, Any]:
        """Format the result for API response"""
        return {
            "result_id": result.result_id,
            "request_id": result.request_id,
            "response": result.response,
            "agents_involved": result.agents_involved,
            "conversation_flow": result.conversation_flow,
            "processing_time": result.total_processing_time,
            "success": result.success,
            "confidence_score": result.confidence_score,
            "performance_metrics": result.performance_metrics,
            "timestamp": result.timestamp.isoformat(),
            **({"error_details": result.error_details} if result.error_details else {})
        }
    
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
            "active_conversations": len(self.active_conversations),
            "performance_monitoring": self.enable_performance_monitoring,
            "detailed_flow_logging": self.enable_detailed_flow_logging
        }
        
        return {
            "orchestrator": orchestrator_status,
            "agents": agent_statuses,
            "network_health": all(
                agent_status.get("health", {}).get("status") == "healthy"
                for agent_status in agent_statuses.values()
            )
        }
    
    def get_conversation_history(self, request_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific request"""
        return self.conversation_flows.get(request_id, [])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "agent_metrics": self.performance_metrics.copy(),
            "orchestrator_metrics": {
                "total_requests": len(self.conversation_flows),
                "active_conversations": len(self.active_conversations),
                "average_chain_length": self._calculate_average_chain_length(),
                "configuration": {
                    "confidence_threshold": self.confidence_threshold,
                    "max_hops": self.max_chain_hops,
                    "timeout": self.agent_timeout,
                    "dynamic_routing": self.enable_dynamic_routing,
                    "performance_monitoring": self.enable_performance_monitoring
                }
            },
            "bidirectional_chaining_metrics": {
                "threshold_success_rate": self._calculate_threshold_success_rate(),
                "average_confidence_achieved": self._calculate_average_confidence(),
                "routing_efficiency": self._calculate_routing_efficiency()
            }
        }
    
    def _calculate_average_chain_length(self) -> float:
        """Calculate average conversation chain length"""
        if not self.conversation_flows:
            return 0.0
        
        total_hops = sum(len(flow) for flow in self.conversation_flows.values())
        return total_hops / len(self.conversation_flows)
    
    def _calculate_threshold_success_rate(self) -> float:
        """Calculate the rate at which conversations reach the confidence threshold"""
        if not self.conversation_flows:
            return 0.0
        
        successful = 0
        for flow in self.conversation_flows.values():
            if flow and flow[-1].get("confidence", 0) >= self.confidence_threshold:
                successful += 1
        
        return successful / len(self.conversation_flows)
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average final confidence across all conversations"""
        if not self.conversation_flows:
            return 0.0
        
        total_confidence = 0
        count = 0
        
        for flow in self.conversation_flows.values():
            if flow:
                final_confidence = flow[-1].get("confidence", 0)
                total_confidence += final_confidence
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def _calculate_routing_efficiency(self) -> float:
        """Calculate routing efficiency (lower hop count for same confidence is better)"""
        if not self.conversation_flows:
            return 1.0
        
        efficiency_scores = []
        for flow in self.conversation_flows.values():
            if flow:
                hops = len(flow)
                final_confidence = flow[-1].get("confidence", 0)
                efficiency = final_confidence / max(hops, 1)
                efficiency_scores.append(efficiency)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 1.0