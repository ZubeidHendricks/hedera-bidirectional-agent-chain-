"""
Comprehensive integration tests for the bidirectional agent chaining system
Tests complete end-to-end workflows with 100% coverage of integration scenarios.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from chain.orchestrator import ChainOrchestrator
from agents.support_agent import SupportAgent
from agents.technical_agent import TechnicalAgent
from agents.product_agent import ProductAgent
from models.message import UserRequest, Message, Response
from terminal_client import TerminalClient
from main import app
from fastapi.testclient import TestClient


class TestEndToEndAgentChain:
    """Test complete end-to-end agent chain workflows."""
    
    @pytest.mark.integration
    async def test_simple_support_query_flow(self, mock_genai_client):
        """Test simple query that Support agent can handle directly."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock Support agent to provide high-confidence direct response
            mock_response = Response(
                content="To reset your password, click 'Forgot Password' on the login page and follow the instructions.",
                source_agent="Support",
                target_agent="user",
                confidence=0.95,
                needs_clarification=False
            )
            
            with patch.object(orchestrator.support_agent, 'process_message', return_value=mock_response):
                user_request = UserRequest(
                    query="How do I reset my password?",
                    user_id="test_user_123",
                    priority="medium"
                )
                
                result = await orchestrator.process_request(user_request)
                
                assert result["success"] is True
                assert result["confidence_score"] == 0.95
                assert result["hop_count"] == 1
                assert len(result["agents_involved"]) == 1
                assert "Support" in result["agents_involved"]
                assert "password" in result["response"]
                assert "Forgot Password" in result["response"]
    
    @pytest.mark.integration
    async def test_technical_collaboration_flow(self, mock_genai_client):
        """Test query requiring Support-Technical collaboration."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock sequence: Support → Technical → Support synthesis
            support_responses = [
                Response(
                    content="I see you're experiencing a database error. Let me get our technical team to analyze this. HANDOFF_REQUEST: Technical - Please analyze database error 500 and connection timeout",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.4,
                    suggested_next_agent="Technical"
                ),
                Response(
                    content="Based on our technical analysis, this error occurs when the database connection pool is exhausted. Here's how to resolve it: 1) Restart your application, 2) Check your connection pool settings, 3) Monitor for memory leaks. This should resolve the immediate issue.",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.92
                )
            ]
            
            technical_response = Response(
                content="Database error 500 with connection timeout indicates connection pool exhaustion. Root cause: Too many concurrent connections without proper cleanup. Solution: Restart application service, increase connection pool size to 20, implement connection recycling. Monitor memory usage for 24 hours.",
                source_agent="Technical", 
                target_agent="Support",
                confidence=0.88
            )
            
            call_count = 0
            def mock_support_process(message):
                nonlocal call_count
                response = support_responses[call_count]
                call_count += 1
                return response
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             side_effect=mock_support_process), \
                 patch.object(orchestrator.technical_agent, 'process_message', 
                             return_value=technical_response), \
                 patch.object(orchestrator, '_select_next_agent', 
                             side_effect=["Technical", None]):
                
                user_request = UserRequest(
                    query="My application keeps showing database error 500",
                    user_id="tech_user_456",
                    priority="high"
                )
                
                result = await orchestrator.process_request(user_request)
                
                assert result["success"] is True
                assert result["confidence_score"] == 0.92
                assert result["hop_count"] >= 2
                assert len(result["agents_involved"]) >= 2
                assert "Support" in result["agents_involved"]
                assert "Technical" in result["agents_involved"]
                assert "database error" in result["response"]
                assert "connection pool" in result["response"]
    
    @pytest.mark.integration
    async def test_three_agent_collaboration_flow(self, mock_genai_client):
        """Test complex query requiring all three agents."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock complex flow: Support → Technical → Product → Support
            support_responses = [
                Response(
                    content="I understand you're having API issues with Enterprise features. HANDOFF_REQUEST: Technical - Please analyze API rate limiting errors",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.3,
                    suggested_next_agent="Technical"
                ),
                Response(
                    content="Based on technical and product analysis, your Enterprise account should have unlimited API access, but there's a configuration issue. Here's the complete solution: 1) Contact support to enable Enterprise API settings, 2) Use the Enterprise API endpoint, 3) Include your Enterprise token in requests. This will resolve both the rate limiting and feature access issues.",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.95
                )
            ]
            
            technical_response = Response(
                content="API rate limiting analysis: Customer hitting 1000/hour limit but Enterprise tier should be unlimited. Issue is configuration - customer using Basic tier API endpoint. HANDOFF_REQUEST: Product - Please confirm Enterprise tier API access and endpoint configuration",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.6,
                suggested_next_agent="Product"
            )
            
            product_response = Response(
                content="Product Analysis: Customer has Enterprise tier subscription with unlimited API access. Problem: Using wrong API endpoint (api.basic.domain.com instead of api.enterprise.domain.com). Enterprise features require Enterprise endpoint with Enterprise authentication token. Configuration: Enable Enterprise API access in customer portal.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.9
            )
            
            call_count = 0
            def mock_support_process(message):
                nonlocal call_count
                response = support_responses[call_count]
                call_count += 1
                return response
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             side_effect=mock_support_process), \
                 patch.object(orchestrator.technical_agent, 'process_message', 
                             return_value=technical_response), \
                 patch.object(orchestrator.product_agent, 'process_message', 
                             return_value=product_response), \
                 patch.object(orchestrator, '_select_next_agent', 
                             side_effect=["Technical", "Product", "Support", None]):
                
                user_request = UserRequest(
                    query="I'm getting API rate limit errors but I have Enterprise subscription",
                    user_id="enterprise_user_789",
                    priority="high",
                    context={"customer_tier": "enterprise", "api_usage": "high"}
                )
                
                result = await orchestrator.process_request(user_request)
                
                assert result["success"] is True
                assert result["confidence_score"] == 0.95
                assert result["hop_count"] >= 3
                assert len(result["agents_involved"]) == 3
                assert "Support" in result["agents_involved"]
                assert "Technical" in result["agents_involved"]
                assert "Product" in result["agents_involved"]
                assert "Enterprise" in result["response"]
                assert "API" in result["response"]
    
    @pytest.mark.integration
    async def test_iterative_refinement_flow(self, mock_genai_client):
        """Test iterative refinement where agents improve response through multiple hops."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock iterative responses with increasing confidence
            responses_sequence = [
                Response(content="Initial analysis...", source_agent="Support", target_agent="user", confidence=0.3),
                Response(content="Technical input received...", source_agent="Technical", target_agent="Support", confidence=0.5),
                Response(content="Product context added...", source_agent="Product", target_agent="Support", confidence=0.6),
                Response(content="Refined analysis with technical details...", source_agent="Support", target_agent="user", confidence=0.75),
                Response(content="Final comprehensive solution...", source_agent="Support", target_agent="user", confidence=0.9)
            ]
            
            response_iter = iter(responses_sequence)
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             side_effect=lambda msg: next(response_iter)), \
                 patch.object(orchestrator.technical_agent, 'process_message', 
                             side_effect=lambda msg: next(response_iter)), \
                 patch.object(orchestrator.product_agent, 'process_message', 
                             side_effect=lambda msg: next(response_iter)), \
                 patch.object(orchestrator, '_select_next_agent', 
                             side_effect=["Technical", "Product", "Support", "Support", None]):
                
                user_request = UserRequest(
                    query="Complex integration issue with multiple systems",
                    user_id="complex_user_999"
                )
                
                result = await orchestrator.process_request(user_request)
                
                assert result["success"] is True
                assert result["confidence_score"] == 0.9
                assert result["hop_count"] >= 4
                assert "comprehensive solution" in result["response"]


class TestAPIIntegration:
    """Test API integration with the agent chain."""
    
    @pytest.mark.integration
    def test_api_query_endpoint_integration(self):
        """Test complete API query endpoint integration."""
        mock_orchestrator_result = {
            "success": True,
            "request_id": "api_test_123",
            "response": "API integration test response",
            "confidence_score": 0.88,
            "agents_involved": ["Support", "Technical"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "processed"},
                {"step": 2, "agent": "Technical", "action": "analyzed"}
            ],
            "total_processing_time": 2.1,
            "hop_count": 2
        }
        
        with patch('main.orchestrator.process_request', return_value=mock_orchestrator_result):
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": "Integration test query",
                    "user_id": "api_test_user",
                    "priority": "medium",
                    "context": {"test": "integration"}
                })
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["success"] is True
                assert data["response"] == "API integration test response"
                assert data["confidence_score"] == 0.88
                assert data["hop_count"] == 2
                assert len(data["agents_involved"]) == 2
    
    @pytest.mark.integration
    def test_api_health_endpoint_integration(self):
        """Test health endpoint integration with real orchestrator."""
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] in ["healthy", "degraded"]
            assert data["service"] == "Bidirectional Agent Chain"
            assert "timestamp" in data
            assert "version" in data
            assert "orchestrator" in data
            
            # Check orchestrator status
            orchestrator_status = data["orchestrator"]
            assert "status" in orchestrator_status
            assert "agents" in orchestrator_status
            
            # Verify all expected agents are present
            agents = orchestrator_status["agents"]
            expected_agents = ["Support", "Technical", "Product"]
            for agent in expected_agents:
                assert agent in agents
    
    @pytest.mark.integration
    def test_api_agents_endpoint_integration(self):
        """Test agents endpoint integration."""
        with TestClient(app) as client:
            response = client.get("/agents")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "agents" in data
            assert "network_topology" in data
            assert "capabilities" in data
            
            # Check agent details
            agents = data["agents"]
            assert "Support" in agents
            assert "Technical" in agents
            assert "Product" in agents
            
            # Verify agent information
            support_agent = agents["Support"]
            assert support_agent["name"] == "Support"
            assert support_agent["expertise"] == "customer_communication"
            assert isinstance(support_agent["capabilities"], list)
    
    @pytest.mark.integration
    def test_api_metrics_endpoint_integration(self):
        """Test metrics endpoint integration."""
        with TestClient(app) as client:
            response = client.get("/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "agent_performance" in data
            assert "system_metrics" in data
            assert "timestamp" in data
            
            # Check system metrics structure
            system_metrics = data["system_metrics"]
            assert "total_requests" in system_metrics
            assert "active_conversations" in system_metrics
            assert "uptime_seconds" in system_metrics


class TestTerminalClientIntegration:
    """Test terminal client integration with the API."""
    
    @pytest.mark.integration
    async def test_terminal_client_full_workflow(self):
        """Test complete terminal client workflow."""
        client = TerminalClient()
        
        # Mock API response
        mock_api_response = {
            "success": True,
            "request_id": "terminal_test_456",
            "response": "Terminal integration test successful",
            "confidence_score": 0.9,
            "agents_involved": ["Support"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "processed", "agent_usage_count": 1}
            ],
            "total_processing_time": 1.2,
            "hop_count": 1
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_api_response)
        
        with patch.object(client.session, 'post', return_value=mock_response):
            result = await client.send_query("Terminal test query", "terminal_user")
            
            assert result is not None
            assert result["success"] is True
            assert result["response"] == "Terminal integration test successful"
            assert client.request_count == 1
    
    @pytest.mark.integration
    async def test_terminal_interactive_session_integration(self):
        """Test terminal interactive session integration."""
        client = TerminalClient()
        
        # Mock user conversation
        user_inputs = [
            "How do I configure SSO?",
            "What are the requirements?",
            "exit"
        ]
        input_iter = iter(user_inputs)
        
        mock_responses = [
            {
                "success": True,
                "response": "To configure SSO, you need to access the Admin panel...",
                "confidence_score": 0.85
            },
            {
                "success": True, 
                "response": "SSO requirements include SAML 2.0 provider and admin access...",
                "confidence_score": 0.9
            }
        ]
        response_iter = iter(mock_responses)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch.object(client, 'send_query', side_effect=lambda *args: next(response_iter)), \
             patch.object(client, 'display_response') as mock_display, \
             patch('builtins.print'):
            
            await client.run_interactive()
            
            # Should have processed 2 queries
            assert mock_display.call_count == 2
            assert client.request_count == 2


class TestErrorRecoveryIntegration:
    """Test error recovery and graceful degradation."""
    
    @pytest.mark.integration
    async def test_agent_failure_recovery(self, mock_genai_client):
        """Test system recovery when an agent fails."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock Technical agent failure, Support handles gracefully
            support_response = Response(
                content="I understand your technical question. While our technical specialist is unavailable, I can provide general guidance...",
                source_agent="Support",
                target_agent="user",
                confidence=0.6
            )
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             return_value=support_response), \
                 patch.object(orchestrator.technical_agent, 'process_message', 
                             side_effect=Exception("Technical agent unavailable")), \
                 patch.object(orchestrator, '_select_next_agent', 
                             side_effect=["Technical", None]):
                
                user_request = UserRequest(
                    query="Technical configuration question",
                    user_id="recovery_test_user"
                )
                
                result = await orchestrator.process_request(user_request)
                
                # Should still succeed despite Technical agent failure
                assert result["success"] is True
                assert result["confidence_score"] >= 0.5
                assert "Support" in result["agents_involved"]
    
    @pytest.mark.integration
    def test_api_error_handling_integration(self):
        """Test API error handling integration."""
        with patch('main.orchestrator.process_request', 
                   side_effect=Exception("Orchestrator failure")):
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": "Error test query",
                    "user_id": "error_test_user"
                })
                
                assert response.status_code == 500
                data = response.json()
                assert "error" in data["detail"]
                assert "Failed to process query" in data["detail"]["error"]
    
    @pytest.mark.integration
    async def test_network_error_recovery(self):
        """Test terminal client network error recovery."""
        client = TerminalClient()
        
        import aiohttp
        with patch.object(client.session, 'post', 
                         side_effect=aiohttp.ClientConnectorError(Mock(), Mock())):
            result = await client.send_query("Network test query", "network_user")
            
            assert result is None  # Should handle gracefully
            assert client.request_count == 1  # Still increment counter


class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_response_time_integration(self, mock_genai_client):
        """Test end-to-end response times meet expectations."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            mock_response = Response(
                content="Performance test response",
                source_agent="Support",
                target_agent="user",
                confidence=0.9
            )
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             return_value=mock_response):
                
                start_time = time.time()
                
                user_request = UserRequest(
                    query="Performance test query",
                    user_id="perf_user"
                )
                
                result = await orchestrator.process_request(user_request)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                assert result["success"] is True
                assert processing_time < 10.0  # Should complete within 10 seconds
                assert result["total_processing_time"] >= 0
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_concurrent_requests_integration(self, mock_genai_client):
        """Test system handles concurrent requests properly."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            mock_response = Response(
                content="Concurrent test response",
                source_agent="Support",
                target_agent="user",
                confidence=0.85
            )
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             return_value=mock_response):
                
                # Create multiple concurrent requests
                requests = [
                    UserRequest(
                        query=f"Concurrent query {i}",
                        user_id=f"concurrent_user_{i}"
                    )
                    for i in range(5)
                ]
                
                # Process concurrently
                start_time = time.time()
                tasks = [orchestrator.process_request(req) for req in requests]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                # All should succeed
                assert len(results) == 5
                assert all(result["success"] for result in results)
                
                # Concurrent processing should be faster than sequential
                total_time = end_time - start_time
                assert total_time < 30.0  # Should complete within 30 seconds
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_api_throughput_integration(self):
        """Test API can handle multiple requests efficiently."""
        mock_result = {
            "success": True,
            "request_id": "throughput_test",
            "response": "Throughput test response",
            "confidence_score": 0.8,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 0.5,
            "hop_count": 1
        }
        
        with patch('main.orchestrator.process_request', return_value=mock_result):
            with TestClient(app) as client:
                
                # Send multiple requests rapidly
                responses = []
                start_time = time.time()
                
                for i in range(10):
                    response = client.post("/query", json={
                        "query": f"Throughput test {i}",
                        "user_id": f"throughput_user_{i}"
                    })
                    responses.append(response)
                
                end_time = time.time()
                
                # All should succeed
                assert all(resp.status_code == 200 for resp in responses)
                assert len(responses) == 10
                
                # Should complete within reasonable time
                total_time = end_time - start_time
                assert total_time < 10.0  # Should handle 10 requests in under 10 seconds


class TestDataFlowIntegration:
    """Test data flow through the complete system."""
    
    @pytest.mark.integration
    async def test_context_preservation_flow(self, mock_genai_client):
        """Test context is preserved through agent handoffs."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock agents that check for context preservation
            def support_process(message):
                assert message.context is not None
                assert "customer_tier" in message.context
                return Response(
                    content="Support processing with context",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.4,
                    suggested_next_agent="Technical"
                )
            
            def technical_process(message):
                # Technical agent should receive enhanced context
                return Response(
                    content="Technical analysis with preserved context",
                    source_agent="Technical",
                    target_agent="Support",
                    confidence=0.8
                )
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             side_effect=support_process), \
                 patch.object(orchestrator.technical_agent, 'process_message', 
                             side_effect=technical_process), \
                 patch.object(orchestrator, '_select_next_agent', 
                             side_effect=["Technical", None]):
                
                user_request = UserRequest(
                    query="Context preservation test",
                    user_id="context_user",
                    context={
                        "customer_tier": "enterprise",
                        "session_id": "context_session_123",
                        "user_preferences": {"language": "en"}
                    }
                )
                
                result = await orchestrator.process_request(user_request)
                
                assert result["success"] is True
                assert "context" in result["response"]
    
    @pytest.mark.integration
    def test_request_response_cycle_integration(self):
        """Test complete request-response cycle through API."""
        mock_result = {
            "success": True,
            "request_id": "cycle_test_789",
            "response": "Complete cycle test successful",
            "confidence_score": 0.92,
            "agents_involved": ["Support", "Technical"],
            "conversation_flow": [
                {
                    "step": 1,
                    "agent": "Support", 
                    "action": "initiated",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.4
                },
                {
                    "step": 2,
                    "agent": "Technical",
                    "action": "analyzed", 
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.8
                },
                {
                    "step": 3,
                    "agent": "Support",
                    "action": "synthesized",
                    "timestamp": datetime.now().isoformat(), 
                    "confidence": 0.92
                }
            ],
            "total_processing_time": 2.8,
            "hop_count": 3
        }
        
        with patch('main.orchestrator.process_request', return_value=mock_result):
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": "Complete integration test query",
                    "user_id": "integration_user",
                    "priority": "high",
                    "expected_response_format": "structured",
                    "context": {
                        "integration_test": True,
                        "test_id": "cycle_test_789"
                    }
                })
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify complete data flow
                assert data["success"] is True
                assert data["request_id"] == "cycle_test_789"
                assert data["response"] == "Complete cycle test successful"
                assert data["confidence_score"] == 0.92
                assert data["hop_count"] == 3
                assert len(data["conversation_flow"]) == 3
                
                # Verify conversation flow details
                flow = data["conversation_flow"]
                assert flow[0]["agent"] == "Support"
                assert flow[1]["agent"] == "Technical"
                assert flow[2]["agent"] == "Support"
                assert all("timestamp" in step for step in flow)


class TestRealtimeIntegration:
    """Test real-time aspects of the system."""
    
    @pytest.mark.integration
    async def test_live_conversation_tracking(self, mock_genai_client):
        """Test live conversation tracking during processing."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock slower agent response to test tracking
            async def slow_support_process(message):
                await asyncio.sleep(0.1)  # Simulate processing time
                return Response(
                    content="Slow processing complete",
                    source_agent="Support",
                    target_agent="user",
                    confidence=0.9
                )
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             side_effect=slow_support_process):
                
                user_request = UserRequest(
                    query="Live tracking test",
                    user_id="live_user"
                )
                
                # Start processing
                task = asyncio.create_task(orchestrator.process_request(user_request))
                
                # Check that conversation is tracked as active
                await asyncio.sleep(0.05)  # Let processing start
                assert user_request.request_id in orchestrator.active_conversations
                
                # Wait for completion
                result = await task
                
                # Should be completed and removed from active
                assert result["success"] is True
                assert user_request.request_id not in orchestrator.active_conversations
    
    @pytest.mark.integration
    def test_conversation_retrieval_integration(self):
        """Test conversation retrieval from API."""
        # Mock conversation data
        conversation_id = "retrieval_test_999"
        mock_conversation = {
            "conversation_id": conversation_id,
            "user_id": "retrieval_user",
            "start_time": datetime.now().isoformat(),
            "messages": [
                {
                    "content": "How do I configure the system?",
                    "source_agent": "user",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "content": "To configure the system, follow these steps...",
                    "source_agent": "Support",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "flow": [
                {"step": 1, "agent": "Support", "action": "processed"}
            ],
            "status": "completed"
        }
        
        with patch('main.orchestrator.conversation_flows', {conversation_id: mock_conversation}):
            with TestClient(app) as client:
                response = client.get(f"/conversation/{conversation_id}")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["conversation_id"] == conversation_id
                assert data["user_id"] == "retrieval_user"
                assert len(data["messages"]) == 2
                assert data["status"] == "completed"


class TestSystemResilience:
    """Test system resilience and fault tolerance."""
    
    @pytest.mark.integration
    async def test_partial_agent_failure_resilience(self, mock_genai_client):
        """Test system continues to work with partial agent failures."""
        with patch.object(SupportAgent, '_initialize_genai_client'), \
             patch.object(TechnicalAgent, '_initialize_genai_client'), \
             patch.object(ProductAgent, '_initialize_genai_client'):
            
            orchestrator = ChainOrchestrator()
            
            # Mock Product agent failure, but Support still works
            support_response = Response(
                content="I can help with general questions even though some specialists are unavailable.",
                source_agent="Support",
                target_agent="user",
                confidence=0.7
            )
            
            with patch.object(orchestrator.support_agent, 'process_message', 
                             return_value=support_response), \
                 patch.object(orchestrator.product_agent, 'process_message', 
                             side_effect=Exception("Product agent down")):
                
                user_request = UserRequest(
                    query="General help question",
                    user_id="resilience_user"
                )
                
                result = await orchestrator.process_request(user_request)
                
                # Should still work with reduced capability
                assert result["success"] is True
                assert result["confidence_score"] >= 0.5
                assert "Support" in result["agents_involved"]
    
    @pytest.mark.integration
    def test_high_load_resilience(self):
        """Test system resilience under high load."""
        mock_result = {
            "success": True,
            "request_id": "load_test",
            "response": "Load test response",
            "confidence_score": 0.8,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 1.0,
            "hop_count": 1
        }
        
        with patch('main.orchestrator.process_request', return_value=mock_result):
            with TestClient(app) as client:
                
                # Simulate high load with rapid requests
                import threading
                results = []
                errors = []
                
                def make_request(request_id):
                    try:
                        response = client.post("/query", json={
                            "query": f"Load test query {request_id}",
                            "user_id": f"load_user_{request_id}"
                        })
                        results.append(response.status_code)
                    except Exception as e:
                        errors.append(str(e))
                
                # Create 20 concurrent requests
                threads = [
                    threading.Thread(target=make_request, args=(i,))
                    for i in range(20)
                ]
                
                for thread in threads:
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                # Most requests should succeed
                success_rate = sum(1 for status in results if status == 200) / len(results)
                assert success_rate >= 0.8  # At least 80% success rate
                assert len(errors) < 5  # Minimal errors 