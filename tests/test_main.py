"""
Comprehensive tests for main.py FastAPI application
Tests all endpoints, middleware, and application logic with 100% coverage.
"""

import pytest
import asyncio
import os
import threading
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import FastAPI app and dependencies
from main import app


class TestFastAPIApplication:
    """Test FastAPI application setup and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_initialization(self):
        """Test FastAPI app is properly initialized."""
        assert app is not None
        assert app.title == "Bidirectional Agent Chain API"
        assert app.version == "2.0.0"
        assert "Modern Bidirectional Agent Chaining System" in app.description
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_middleware_configuration(self):
        """Test CORS middleware is properly configured."""
        # Check that CORS middleware is added
        middleware_types = [type(middleware.cls) for middleware in app.user_middleware]
        from starlette.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_types
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_exists(self):
        """Test FastAPI app exists and has basic properties."""
        assert app is not None
        assert app.title == "Bidirectional Agent Chain API"
        assert app.version == "2.0.0"


class TestHealthEndpoint:
    """Test health check endpoint functionality."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_basic(self):
        """Test basic health endpoint functionality."""
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert data["version"] == "2.0.0"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_with_orchestrator(self):
        """Test health endpoint with real orchestrator."""
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have basic health information
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert data["version"] == "2.0.0"
            
            # Should have orchestrator information
            if "orchestrator" in data:
                orchestrator_status = data["orchestrator"]
                assert "status" in orchestrator_status
                assert "agents" in orchestrator_status
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_with_orchestrator_error(self):
        """Test health endpoint when orchestrator has issues."""
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.get_agent_status = AsyncMock(side_effect=Exception("Orchestrator error"))
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.get("/health")
                
                # Should still return 200 but with error status
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "degraded"


class TestQueryEndpoint:
    """Test the main query processing endpoint."""
    
    @pytest.mark.unit
    @pytest.mark.api  
    def test_query_endpoint_validation_error(self):
        """Test query endpoint with validation errors."""
        with TestClient(app) as client:
            # Test with empty query which should fail min_length validation
            response = client.post("/query", json={
                "query": "",  # This should fail min_length=1
                "user_id": "test_user"
            })
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_query_endpoint_basic_functionality(self):
        """Test query endpoint basic functionality with real orchestrator."""
        with TestClient(app) as client:
            response = client.post("/query", json={
                "query": "Test query for validation",
                "user_id": "test_user"
            })
            
            # Should return 200 with real orchestrator
            assert response.status_code == 200
            data = response.json()
            
            # Basic response structure validation
            assert "success" in data
            assert "response" in data
            assert "confidence_score" in data
            assert "agents_involved" in data
            assert "processing_time" in data
            
            # Should have some response content
            assert len(data["response"]) > 0
            assert isinstance(data["agents_involved"], list)
            assert len(data["agents_involved"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_query_endpoint_with_all_fields(self):
        """Test query endpoint with all optional fields."""
        mock_result = {
            "success": True,
            "request_id": "test_456",
            "response": "Complete response",
            "confidence_score": 0.9,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 1.0,
            "hop_count": 1
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.process_request = AsyncMock(return_value=mock_result)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": "Complex query",
                    "user_id": "premium_user",
                    "priority": "high",
                    "expected_response_format": "structured",
                    "max_processing_time": 120,
                    "context": {
                        "session_id": "abc123",
                        "user_type": "premium"
                    }
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_query_endpoint_processing_error(self):
        """Test query endpoint when orchestrator raises an error."""
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.process_request = AsyncMock(side_effect=Exception("Processing failed"))
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": "Test query",
                    "user_id": "test_user"
                })
                
                assert response.status_code == 500
                data = response.json()
                assert "detail" in data
                assert "error" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_query_endpoint_empty_query(self):
        """Test query endpoint with empty query string."""
        mock_result = {
            "success": True,
            "request_id": "test_789",
            "response": "I notice your message was empty. How can I help you?",
            "confidence_score": 0.8,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 0.5,
            "hop_count": 1
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.process_request = AsyncMock(return_value=mock_result)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": "",
                    "user_id": "test_user"
                })
                
                # This should fail validation due to min_length=1
                assert response.status_code == 422


class TestConversationEndpoint:
    """Test conversation history retrieval endpoint."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_get_conversation_success(self):
        """Test successful conversation retrieval."""
        conversation_id = "test_conv_123"
        mock_conversation = {
            "conversation_id": conversation_id,
            "user_id": "test_user",
            "start_time": "2024-01-01T12:00:00Z",
            "messages": [
                {
                    "content": "User question",
                    "source_agent": "user",
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                {
                    "content": "Agent response",
                    "source_agent": "Support",
                    "timestamp": "2024-01-01T12:00:05Z"
                }
            ],
            "flow": [
                {"step": 1, "agent": "Support", "action": "processed"}
            ],
            "status": "completed"
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.get_conversation_history = Mock(return_value=mock_conversation)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.get(f"/conversation/{conversation_id}")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["conversation_id"] == conversation_id
                assert data["user_id"] == "test_user"
                assert len(data["messages"]) == 2
                assert data["status"] == "completed"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_get_conversation_not_found(self):
        """Test conversation retrieval when conversation doesn't exist."""
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.get_conversation_history = Mock(return_value=None)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.get("/conversation/nonexistent_id")
                
                assert response.status_code == 404
                data = response.json()
                assert "not found" in data["detail"].lower()


class TestMetricsEndpoint:
    """Test performance metrics endpoint."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_get_metrics_success(self):
        """Test successful metrics retrieval."""
        mock_metrics = {
            "Support": {
                "total_calls": 150,
                "total_time": 75.5,
                "avg_time": 0.503,
                "avg_confidence": 0.85,
                "success_rate": 0.96
            },
            "Technical": {
                "total_calls": 89,
                "total_time": 112.3,
                "avg_time": 1.262,
                "avg_confidence": 0.91,
                "success_rate": 0.94
            },
            "Product": {
                "total_calls": 64,
                "total_time": 87.2,
                "avg_time": 1.362,
                "avg_confidence": 0.88,
                "success_rate": 0.98
            }
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.get_performance_metrics = Mock(return_value=mock_metrics)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.get("/metrics")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "agent_performance" in data
                assert "system_metrics" in data
                assert "timestamp" in data
                
                agent_perf = data["agent_performance"]
                assert "Support" in agent_perf
                assert "Technical" in agent_perf
                assert "Product" in agent_perf
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_get_metrics_basic(self):
        """Test metrics endpoint basic structure."""
        with TestClient(app) as client:
            response = client.get("/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "agent_performance" in data
            assert "system_metrics" in data
            assert "timestamp" in data


class TestAgentsEndpoint:
    """Test agents information endpoint."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_get_agents_success(self):
        """Test successful agents information retrieval."""
        mock_agents = {
            "Support": {
                "name": "Support",
                "expertise": "customer_communication",
                "status": "healthy",
                "capabilities": ["communication", "coordination"],
                "connections": ["Technical", "Product"]
            },
            "Technical": {
                "name": "Technical", 
                "expertise": "technical_analysis",
                "status": "healthy",
                "capabilities": ["diagnostics", "troubleshooting"],
                "connections": ["Support", "Product"]
            },
            "Product": {
                "name": "Product",
                "expertise": "product_knowledge", 
                "status": "healthy",
                "capabilities": ["features", "configuration"],
                "connections": ["Support", "Technical"]
            }
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.agents = mock_agents
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.get("/agents")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "agents" in data
                assert "network_topology" in data
                assert "capabilities" in data
                
                agents = data["agents"]
                assert "Support" in agents
                assert "Technical" in agents
                assert "Product" in agents
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_agents_endpoint_basic(self):
        """Test agents endpoint basic structure."""
        with TestClient(app) as client:
            response = client.get("/agents")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "agents" in data
            assert "network_topology" in data
            assert "capabilities" in data


class TestDocumentationEndpoints:
    """Test API documentation endpoints."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_openapi_schema_endpoint(self):
        """Test OpenAPI schema endpoint."""
        with TestClient(app) as client:
            response = client.get("/openapi.json")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "openapi" in data
            assert "info" in data
            assert "paths" in data
            assert data["info"]["title"] == "Bidirectional Agent Chain API"


class TestErrorHandling:
    """Test application-level error handling."""
    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.error_handling
    def test_404_not_found(self):
        """Test 404 error for non-existent endpoints."""
        with TestClient(app) as client:
            response = client.get("/nonexistent")
            
            assert response.status_code == 404
    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.error_handling
    def test_method_not_allowed(self):
        """Test 405 error for wrong HTTP method."""
        with TestClient(app) as client:
            response = client.put("/health")  # Health endpoint only supports GET
            
            assert response.status_code == 405
    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.error_handling
    def test_request_validation_error(self):
        """Test validation error handling."""
        with TestClient(app) as client:
            response = client.post("/query", json={
                "invalid_field": "invalid_value"
            })
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data


class TestCORSConfiguration:
    """Test CORS configuration and middleware."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_headers(self):
        """Test CORS headers are present."""
        with TestClient(app) as client:
            response = client.options("/health")
            
            # Should handle OPTIONS request for CORS
            assert response.status_code in [200, 204, 405]  # Various valid responses
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_preflight_request(self):
        """Test CORS preflight request handling."""
        with TestClient(app) as client:
            response = client.options("/query", headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            })
            
            # Should allow the request
            assert response.status_code in [200, 204]


class TestDataModelsAndValidation:
    """Test Pydantic models and validation."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_query_request_model_validation(self):
        """Test QueryRequest model validation."""
        from main import QueryRequest
        
        # Valid request
        valid_request = QueryRequest(
            query="Test query",
            user_id="test_user",
            priority="medium"
        )
        assert valid_request.query == "Test query"
        assert valid_request.priority == "medium"
        
        # Test default values
        assert valid_request.expected_response_format == "text"
        assert valid_request.max_processing_time == 60
        assert valid_request.context is None
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_query_response_model_validation(self):
        """Test QueryResponse model validation."""
        from main import QueryResponse
        
        # Valid response
        valid_response = QueryResponse(
            result_id="result_123",
            request_id="test_123",
            response="Test response",
            agents_involved=["Support"],
            conversation_flow=[],
            processing_time=2.5,
            success=True,
            confidence_score=0.85
        )
        
        assert valid_response.success is True
        assert valid_response.confidence_score == 0.85
        assert valid_response.processing_time == 2.5
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_priority_validation(self):
        """Test priority field validation."""
        from main import QueryRequest
        
        valid_priorities = ["low", "medium", "high", "critical"]
        
        for priority in valid_priorities:
            request = QueryRequest(
                query="Test",
                user_id="user",
                priority=priority
            )
            assert request.priority == priority
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_response_format_validation(self):
        """Test response format validation."""
        from main import QueryRequest
        
        valid_formats = ["text", "structured", "json"]
        
        for format_type in valid_formats:
            request = QueryRequest(
                query="Test",
                user_id="user",
                expected_response_format=format_type
            )
            assert request.expected_response_format == format_type


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_debug_mode_configuration(self):
        """Test debug mode affects app configuration."""
        # Test that debug mode affects documentation availability
        import main
        
        # DEBUG_MODE should be loaded from environment
        assert hasattr(main, 'DEBUG_MODE')
        assert isinstance(main.DEBUG_MODE, bool)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_origins_configuration(self):
        """Test CORS origins configuration."""
        import main
        
        # CORS_ORIGINS should be loaded from environment
        assert hasattr(main, 'CORS_ORIGINS')
        assert isinstance(main.CORS_ORIGINS, list)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_server_configuration(self):
        """Test server configuration variables."""
        import main
        
        # Should have server configuration
        assert hasattr(main, 'SERVER_HOST')
        assert hasattr(main, 'SERVER_PORT')
        assert isinstance(main.SERVER_HOST, str)
        assert isinstance(main.SERVER_PORT, int)


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.performance
    def test_concurrent_requests_handling(self):
        """Test handling multiple concurrent requests."""
        mock_result = {
            "success": True,
            "request_id": "concurrent_test",
            "response": "Concurrent response",
            "confidence_score": 0.8,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 1.0,
            "hop_count": 1
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.process_request = AsyncMock(return_value=mock_result)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                # Send multiple requests concurrently
                results = []
                
                def make_request():
                    response = client.post("/query", json={
                        "query": "Concurrent test",
                        "user_id": "concurrent_user"
                    })
                    results.append(response.status_code)
                
                threads = [threading.Thread(target=make_request) for _ in range(5)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                
                # All requests should succeed
                assert all(status == 200 for status in results)
                assert len(results) == 5
    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.performance
    def test_large_request_handling(self):
        """Test handling of large request payloads."""
        large_query = "x" * 1000  # 1KB query (reduced from 10KB for test speed)
        large_context = {f"key_{i}": f"value_{i}" * 10 for i in range(10)}
        
        mock_result = {
            "success": True,
            "request_id": "large_request",
            "response": "Handled large request",
            "confidence_score": 0.8,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 3.0,
            "hop_count": 1
        }
        
        with patch('main.get_orchestrator') as mock_get_orch:
            mock_orchestrator = Mock()
            mock_orchestrator.process_request = AsyncMock(return_value=mock_result)
            mock_get_orch.return_value = mock_orchestrator
            
            with TestClient(app) as client:
                response = client.post("/query", json={
                    "query": large_query,
                    "user_id": "test_user",
                    "context": large_context
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__]) 