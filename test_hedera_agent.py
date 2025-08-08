#!/usr/bin/env python3
"""
Test script for Hedera RAG Agent
Tests the Hedera RAG agent directly without needing the full server
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.hedera_rag_agent import HederaRAGAgent
from models.message import Message

async def test_hedera_agent():
    """Test the Hedera RAG agent with sample queries"""
    print("🌐 Testing Hedera RAG Agent")
    print("=" * 50)
    
    try:
        # Initialize the agent
        print("🔧 Initializing Hedera RAG Agent...")
        agent = HederaRAGAgent()
        
        # Test queries related to Hedera blockchain
        test_queries = [
            "What is Hedera Consensus Service?",
            "How do I create tokens with HTS?", 
            "Tell me about smart contracts on Hedera",
            "What SDKs are available for Hedera development?",
            "How can I build a DeFi application on Hedera?"
        ]
        
        print(f"\n🧪 Running {len(test_queries)} test queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Test {i}: {query}")
            print("-" * 40)
            
            # Create message object
            message = Message(
                content=query,
                source_agent="test_client",
                target_agent="Hedera_RAG",
                message_type="query"
            )
            
            # Process the message
            try:
                response = await agent.process_message(message)
                
                print(f"🎯 Confidence: {response.confidence:.2f}")
                print(f"📄 Response: {response.content[:200]}...")
                if len(response.content) > 200:
                    print("    [Response truncated]")
                
                if hasattr(response, 'metadata') and response.metadata:
                    print(f"📊 Metadata: {response.metadata}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print()
        
        # Test agent capabilities
        print("🔍 Testing agent capabilities...")
        capabilities = agent.get_capabilities()
        print(f"📋 Agent Capabilities ({len(capabilities)}):")
        for i, capability in enumerate(capabilities, 1):
            print(f"  {i}. {capability}")
        
        print(f"\n✅ Hedera RAG Agent test completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize or test agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hedera_agent())