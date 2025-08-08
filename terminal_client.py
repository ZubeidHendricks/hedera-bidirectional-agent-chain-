#!/usr/bin/env python3
"""
Terminal Client for Bidirectional Agent Chain
Interactive chat interface for testing agent-to-agent communication with real-time processing visualization
"""

import requests
import json
import time
import sys
import threading
from typing import Dict, Any, List
from datetime import datetime

class TerminalClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"terminal_session_{int(time.time())}"
        self.processing = False
        self.processing_dots = 0
        print("🤖 Bidirectional Agent Chain - Terminal Client")
        print("=" * 50)
        
    def check_server_health(self, retries: int = 3, retry_delay: float = 1.0) -> bool:
        """Check if the server is running and healthy with retry logic"""
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ Server Status: {health_data.get('status', 'unknown')}")
                    
                    agents = health_data.get('agents', {})
                    print("\n🔗 Agent Network Status:")
                    for agent_name, agent_info in agents.items():
                        status = agent_info.get('health', {}).get('status', 'unknown')
                        connections = agent_info.get('health', {}).get('connections', 0)
                        emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                        print(f"  {emoji} {agent_info.get('name', agent_name)}: {status} ({connections} connections)")
                    
                    return health_data.get('status') in ['healthy', 'degraded']
                else:
                    if attempt < retries - 1:
                        print(f"⚠️ Server returned status {response.status_code}, retrying... ({attempt + 1}/{retries})")
                        time.sleep(retry_delay)
                        continue
                    return False
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(f"⚠️ Connection attempt {attempt + 1} failed, retrying... ({e})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("❌ Server is not running or not accessible after multiple attempts")
                    return False
        return False
    
    def _show_processing_animation(self):
        """Show animated processing indicator"""
        animations = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        while self.processing:
            for char in animations:
                if not self.processing:
                    break
                print(f"\r{char} Processing bidirectional agent chain...", end="", flush=True)
                time.sleep(0.1)
    
    def _simulate_real_time_processing(self, estimated_time: float = 20.0):
        """Simulate real-time processing feedback"""
        stages = [
            ("🚀 Initializing agent chain...", 1.0),
            ("🔍 Analyzing query complexity...", 1.5),
            ("🎯 Selecting initial agent...", 1.0),
            ("🔄 Support Agent processing...", 3.0),
            ("🤔 Evaluating confidence level...", 1.5),
            ("🔀 Considering handoffs...", 2.0),
            ("💡 MotoGP_RAG Agent consulted...", 4.0),
            ("🧠 Synthesizing knowledge...", 2.5),
            ("📊 Analyzing performance data...", 3.0),
            ("✅ Confidence threshold reached...", 1.0),
            ("🎉 Generating final response...", 2.5)
        ]
        
        for stage, duration in stages:
            if not self.processing:
                break
            print(f"\r{stage}", end="", flush=True)
            time.sleep(min(duration, estimated_time / len(stages)))
        
        if self.processing:
            print(f"\r🏁 Processing complete!                    ", flush=True)
    
    def send_query(self, query: str, priority: str = "medium") -> Dict[str, Any]:
        """Send a query to the agent chain with real-time processing visualization"""
        try:
            payload = {
                "query": query,
                "user_id": self.session_id,
                "priority": priority,
                "expected_response_format": "text"
            }
            
            print(f"\n🔄 Initiating Bidirectional Agent Chain...")
            print(f"📝 Query: {query[:80]}{'...' if len(query) > 80 else ''}")
            print(f"🎯 Priority: {priority}")
            print("-" * 60)
            
            # Start processing animation
            self.processing = True
            
            # Start real-time processing simulation in a separate thread
            processing_thread = threading.Thread(target=self._simulate_real_time_processing, daemon=True)
            processing_thread.start()
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=180 
            )
            
            # Stop processing animation
            self.processing = False
            processing_thread.join(timeout=0.1)
            
            processing_time = time.time() - start_time
            print(f"\n⏱️  Total processing time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Server error: {response.status_code}",
                    "details": response.text
                }
                
        except requests.exceptions.RequestException as e:
            self.processing = False
            return {
                "error": "Connection error",
                "details": str(e)
            }
    
    def display_response(self, response_data: Dict[str, Any]):
        """Display the agent chain response with enhanced visualization"""
        if "error" in response_data:
            print(f"❌ Error: {response_data['error']}")
            if "details" in response_data:
                print(f"Details: {response_data['details']}")
            return
        
        print("\n" + "="*70)
        print("🤖 BIDIRECTIONAL AGENT CHAIN EXECUTION SUMMARY")
        print("="*70)
        
        # Show execution flow first
        conversation_flow = response_data.get('conversation_flow', [])
        if conversation_flow:
            self._display_execution_flow(conversation_flow)
        
        # Then show the actual response
        response_text = response_data.get('response', 'No response received')
        print(f"\n💬 FINAL RESPONSE:")
        print("─" * 50)
        print(response_text)
        print("─" * 50)
        
        # Performance metrics
        self._display_performance_metrics(response_data)
        
        print("="*70)
    
    def _display_execution_flow(self, conversation_flow: List[Dict[str, Any]]):
        """Display detailed execution flow with timing and confidence"""
        print(f"\n🔄 EXECUTION TRACE ({len(conversation_flow)} hops):")
        print("─" * 70)
        
        total_agents = set()
        handoff_count = 0
        max_confidence = 0
        
        for i, step in enumerate(conversation_flow, 1):
            agent = step.get('agent', 'Unknown')
            confidence = step.get('confidence', 0)
            processing_time = step.get('processing_time', 0)
            action = step.get('action', 'processed')
            is_handoff = step.get('is_handoff', False)
            source_agent = step.get('source_agent', 'User')
            
            total_agents.add(agent)
            max_confidence = max(max_confidence, confidence)
            
            # Dynamic action emojis
            action_emojis = {
                'initiated': '🚀',
                'processed': '🔄', 
                'consulted': '🤝',
                'analyzed': '🔍',
                'synthesized': '🧠',
                'responded': '💬',
                'escalated': '⬆️',
                'delegated': '➡️',
                'handoff_requested': '🔀',
                'knowledge_retrieved': '📚'
            }
            emoji = action_emojis.get(action, '🔄')
            
            # Confidence bar
            conf_bars = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            confidence_color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
            
            print(f"Hop {i:2d}: {source_agent} ➜ {agent}")
            print(f"       {emoji} {action.replace('_', ' ').title()}")
            print(f"       🎯 Confidence: {confidence:.2f} {confidence_color} [{conf_bars}]")
            print(f"       ⏱️  Time: {processing_time:.2f}s")
            
            if is_handoff:
                handoff_count += 1
                print(f"       🔀 HANDOFF TRIGGERED")
            
            if i < len(conversation_flow):
                print(f"       ↓")
            else:
                threshold_met = confidence >= 0.7
                print(f"       {'✅' if threshold_met else '⚠️'} {'THRESHOLD MET' if threshold_met else 'THRESHOLD NOT MET'}")
            
            print()
        
        # Summary statistics
        print("📊 EXECUTION STATISTICS:")
        print(f"   🤖 Agents involved: {len(total_agents)} ({', '.join(sorted(total_agents))})")
        print(f"   🔀 Handoffs: {handoff_count}")
        print(f"   🎯 Peak confidence: {max_confidence:.2f}")
        print(f"   🔄 Total hops: {len(conversation_flow)}")
        
        # Flow type analysis
        flow_types = []
        if len(total_agents) > 1:
            flow_types.append("Multi-agent")
        if handoff_count > 0:
            flow_types.append("Bidirectional")
        if len(conversation_flow) > 2:
            flow_types.append("Chain")
        if any(step.get('action') == 'synthesized' for step in conversation_flow):
            flow_types.append("Knowledge synthesis")
            
        if flow_types:
            print(f"   🧠 Flow type: {' + '.join(flow_types)}")
    
    def _display_performance_metrics(self, response_data: Dict[str, Any]):
        """Display performance metrics"""
        processing_time = response_data.get('processing_time', 0)
        confidence_score = response_data.get('confidence_score', 0)
        performance_metrics = response_data.get('performance_metrics', {})
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   ⏱️  Total time: {processing_time:.2f}s")
        print(f"   🎯 Final confidence: {confidence_score:.2f}")
        
        total_hops = performance_metrics.get('total_hops', 0)
        if total_hops:
            efficiency = confidence_score / max(total_hops, 1)
            print(f"   📊 Efficiency: {efficiency:.3f} confidence/hop")
            
            # Performance rating
            if efficiency >= 0.15:
                rating = "🌟 Excellent"
            elif efficiency >= 0.10:
                rating = "✅ Good"
            elif efficiency >= 0.05:
                rating = "⚠️ Fair"
            else:
                rating = "🔴 Needs optimization"
            print(f"   📋 Rating: {rating}")
        
        # Agent performance breakdown
        agent_performance = performance_metrics.get('agent_performance', {})
        if agent_performance:
            print(f"\n🤖 AGENT PERFORMANCE:")
            for agent, metrics in agent_performance.items():
                avg_time = metrics.get('average_time', 0)
                avg_conf = metrics.get('average_confidence', 0)
                requests_count = metrics.get('total_requests', 0)
                
                # Performance indicators
                time_indicator = "⚡" if avg_time < 5 else "⏱️" if avg_time < 15 else "🐌"
                conf_indicator = "🎯" if avg_conf >= 0.7 else "📈" if avg_conf >= 0.5 else "⚠️"
                
                print(f"   {agent}: {time_indicator} {avg_time:.1f}s | {conf_indicator} {avg_conf:.2f} | {requests_count} calls")
    
    def interactive_chat(self):
        """Start interactive chat session"""
        if not self.check_server_health():
            print("\n❌ Cannot start chat - server is not healthy")
            print("Make sure to:")
            print("1. Start the server: python main.py")
            print("2. Set a valid GOOGLE_API_KEY in your .env file")
            return
        
        print(f"\n💬 Interactive Chat Started (Session: {self.session_id})")
        print("Type your questions and see the bidirectional agent chain in action!")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'health' - Check agent health")
        print("  'help' - Show example queries")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for testing the bidirectional agent chain!")
                    break
                
                elif user_input.lower() == 'health':
                    self.check_server_health()
                    continue
                
                elif user_input.lower() == 'help':
                    self.show_examples()
                    continue
                
                elif not user_input:
                    print("Please enter a question or command.")
                    continue
                
                # Smart priority detection
                priority = "medium"
                if any(word in user_input.lower() for word in ['urgent', 'critical', 'emergency', 'crash', 'down']):
                    priority = "high"
                elif any(word in user_input.lower() for word in ['quick', 'simple', 'basic']):
                    priority = "low"
                
                response = self.send_query(user_input, priority)
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
    
    def show_examples(self):
        """Show example queries that demonstrate bidirectional chaining"""
        print("\n📝 Example Queries to Test Bidirectional Agent Chaining:")
        print("-" * 60)
        examples = [
            "My mobile app crashes when uploading large files - need technical diagnosis and product solutions",
            "Customer wants to integrate our API but getting authentication errors",
            "System performance is slow during peak hours - need troubleshooting and optimization advice",
            "User reporting data sync issues between mobile and web - need technical analysis",
            "How do I configure the premium features for enterprise customers?",
            "Security scan found vulnerabilities - need assessment and remediation plan",
            "Tell me about Brad Binder's performance and create a support strategy for improvement",
            "Compare MotoGP rider performances in wet vs dry conditions"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("-" * 60)

def main():
    """Main function to run the terminal client"""
    client = TerminalClient()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n🔍 Single Query Mode: {query}")
        response = client.send_query(query)
        client.display_response(response)
    else:
        client.interactive_chat()

if __name__ == "__main__":
    main() 