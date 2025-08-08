"""
Test utilities and helper functions for comprehensive testing.
Provides common utilities, assertions, and test data generation.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid

from models.message import Message, Response, UserRequest, ChainResult
from agents.base_agent import BaseAgent


class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def create_user_request(
        query: str = "Test query",
        user_id: str = "test_user",
        priority: str = "medium",
        **kwargs
    ) -> UserRequest:
        """Create a test user request with default values."""
        defaults = {
            "expected_response_format": "text",
            "max_processing_time": 60,
            "context": None
        }
        defaults.update(kwargs)
        
        return UserRequest(
            query=query,
            user_id=user_id,
            priority=priority,
            **defaults
        )
    
    @staticmethod
    def create_message(
        content: str = "Test message",
        source_agent: str = "user",
        target_agent: str = "Support",
        **kwargs
    ) -> Message:
        """Create a test message with default values."""
        defaults = {
            "message_type": "query",
            "priority": "medium",
            "context": {"test": True},
            "requires_response": True,
            "conversation_id": "test_conversation"
        }
        defaults.update(kwargs)
        
        return Message(
            content=content,
            source_agent=source_agent,
            target_agent=target_agent,
            **defaults
        )
    
    @staticmethod
    def create_response(
        content: str = "Test response",
        source_agent: str = "Support",
        confidence: float = 0.8,
        **kwargs
    ) -> Response:
        """Create a test response with default values."""
        defaults = {
            "target_agent": "user",
            "needs_clarification": False,
            "suggested_next_agent": None,
            "reasoning": "Test reasoning",
            "metadata": {}
        }
        defaults.update(kwargs)
        
        return Response(
            content=content,
            source_agent=source_agent,
            confidence=confidence,
            **defaults
        )
    
    @staticmethod
    def create_chain_result(
        request_id: str = "test_request",
        response: str = "Test chain response",
        success: bool = True,
        **kwargs
    ) -> ChainResult:
        """Create a test chain result with default values."""
        defaults = {
            "agents_involved": ["Support"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "processed"}
            ],
            "total_processing_time": 1.5,
            "confidence_score": 0.8,
            "error_details": None,
            "performance_metrics": {}
        }
        defaults.update(kwargs)
        
        return ChainResult(
            request_id=request_id,
            response=response,
            success=success,
            **defaults
        )
    
    @staticmethod
    def create_conversation_flow(steps: int = 3) -> List[Dict[str, Any]]:
        """Create a test conversation flow with specified number of steps."""
        agents = ["Support", "Technical", "Product"]
        actions = ["initiated", "processed", "analyzed", "synthesized", "completed"]
        
        flow = []
        for i in range(steps):
            step = {
                "step": i + 1,
                "agent": agents[i % len(agents)],
                "action": actions[i % len(actions)],
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.7 + (i * 0.1),
                "processing_time": 1.0 + (i * 0.5),
                "agent_usage_count": 1
            }
            flow.append(step)
        
        return flow
    
    @staticmethod
    def create_mock_api_response(
        success: bool = True,
        include_flow: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a mock API response for testing."""
        base_response = {
            "success": success,
            "request_id": str(uuid.uuid4()),
            "response": "Mock API response",
            "confidence_score": 0.85,
            "agents_involved": ["Support"],
            "total_processing_time": 2.1,
            "hop_count": 1
        }
        
        if include_flow:
            base_response["conversation_flow"] = TestDataGenerator.create_conversation_flow(2)
        else:
            base_response["conversation_flow"] = []
        
        if not success:
            base_response.update({
                "response": "An error occurred during processing",
                "confidence_score": 0.1,
                "error_details": "Mock error for testing"
            })
        
        base_response.update(kwargs)
        return base_response


class TestAssertions:
    """Custom assertions for testing agent chain functionality."""
    
    @staticmethod
    def assert_valid_message(message: Message):
        """Assert that a message has valid structure and content."""
        assert isinstance(message, Message)
        assert message.message_id is not None
        assert isinstance(message.timestamp, datetime)
        assert message.content is not None
        assert message.source_agent is not None
        assert message.target_agent is not None
        assert message.message_type in [
            "query", "response", "notification", "command", "handoff",
            "analysis", "consultation", "synthesis", "collaboration",
            "technical_analysis", "product_analysis", "support_analysis",
            "technical_product_analysis", "product_technical_analysis"
        ]
        assert message.priority in ["low", "medium", "high", "critical"]
    
    @staticmethod
    def assert_valid_response(response: Response):
        """Assert that a response has valid structure and content."""
        assert isinstance(response, Response)
        assert response.response_id is not None
        assert isinstance(response.timestamp, datetime)
        assert response.content is not None
        assert response.source_agent is not None
        assert response.target_agent is not None
        assert 0 <= response.confidence <= 1
        assert isinstance(response.needs_clarification, bool)
    
    @staticmethod
    def assert_valid_user_request(request: UserRequest):
        """Assert that a user request has valid structure."""
        assert isinstance(request, UserRequest)
        assert request.request_id is not None
        assert isinstance(request.timestamp, datetime)
        assert request.query is not None
        assert request.user_id is not None
        assert request.priority in ["low", "medium", "high", "critical"]
        assert request.expected_response_format in ["text", "structured", "json"]
        assert request.max_processing_time >= 0
    
    @staticmethod
    def assert_valid_chain_result(result: ChainResult):
        """Assert that a chain result has valid structure."""
        assert isinstance(result, ChainResult)
        assert result.result_id is not None
        assert isinstance(result.timestamp, datetime)
        assert result.request_id is not None
        assert result.response is not None
        assert isinstance(result.agents_involved, list)
        assert isinstance(result.conversation_flow, list)
        assert result.total_processing_time >= 0
        assert isinstance(result.success, bool)
        assert 0 <= result.confidence_score <= 1
    
    @staticmethod
    def assert_conversation_flow_valid(flow: List[Dict[str, Any]]):
        """Assert that conversation flow has valid structure."""
        assert isinstance(flow, list)
        
        for i, step in enumerate(flow):
            assert "step" in step
            assert "agent" in step
            assert "action" in step
            assert step["step"] == i + 1
            assert step["agent"] in ["Support", "Technical", "Product"]
            assert step["action"] in [
                "initiated", "received", "processed", "analyzed", 
                "synthesized", "responded", "handed_off", "completed"
            ]
    
    @staticmethod
    def assert_agents_involved_valid(agents: List[str]):
        """Assert that agents involved list is valid."""
        assert isinstance(agents, list)
        assert len(agents) > 0
        
        valid_agents = ["Support", "Technical", "Product"]
        for agent in agents:
            assert agent in valid_agents
    
    @staticmethod
    def assert_bidirectional_flow(flow: List[Dict[str, Any]]):
        """Assert that flow demonstrates bidirectional communication."""
        if len(flow) < 2:
            return  # Need at least 2 steps for bidirectional
        
        agents_used = [step["agent"] for step in flow]
        unique_agents = set(agents_used)
        
        # Bidirectional implies multiple agents or repeated agent usage
        assert len(unique_agents) > 1 or len(agents_used) > len(unique_agents)
    
    @staticmethod
    def assert_performance_within_bounds(
        processing_time: float,
        max_time: float = 30.0,
        min_time: float = 0.0
    ):
        """Assert that processing time is within acceptable bounds."""
        assert min_time <= processing_time <= max_time, \
            f"Processing time {processing_time} not within bounds [{min_time}, {max_time}]"
    
    @staticmethod
    def assert_confidence_reasonable(confidence: float):
        """Assert that confidence score is reasonable."""
        assert 0.0 <= confidence <= 1.0
        # Very low confidence might indicate an error
        if confidence < 0.1:
            import warnings
            warnings.warn(f"Very low confidence score: {confidence}")


class MockAgentFactory:
    """Factory for creating mock agents for testing."""
    
    @staticmethod
    def create_mock_support_agent():
        """Create a mock support agent."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "Support"
        mock_agent.expertise = "customer_communication"
        mock_agent.capabilities = ["customer_communication"]
        mock_agent.agent_registry = {}
        mock_agent.conversation_history = []
        
        mock_agent.process_message = AsyncMock(return_value=TestDataGenerator.create_response(
            content="Mock support response",
            source_agent="Support"
        ))
        
        mock_agent.register_agent = Mock()
        mock_agent.get_capabilities = Mock(return_value=["customer_communication"])
        
        return mock_agent
    
    @staticmethod
    def create_mock_technical_agent():
        """Create a mock technical agent."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "Technical"
        mock_agent.expertise = "system_diagnostics"
        mock_agent.capabilities = ["system_diagnostics"]
        mock_agent.agent_registry = {}
        mock_agent.conversation_history = []
        
        mock_agent.process_message = AsyncMock(return_value=TestDataGenerator.create_response(
            content="Mock technical analysis",
            source_agent="Technical",
            confidence=0.9
        ))
        
        mock_agent.register_agent = Mock()
        mock_agent.get_capabilities = Mock(return_value=["system_diagnostics"])
        
        return mock_agent
    
    @staticmethod
    def create_mock_product_agent():
        """Create a mock product agent."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "Product"
        mock_agent.expertise = "product_knowledge"
        mock_agent.capabilities = ["product_knowledge"]
        mock_agent.agent_registry = {}
        mock_agent.conversation_history = []
        
        mock_agent.process_message = AsyncMock(return_value=TestDataGenerator.create_response(
            content="Mock product information",
            source_agent="Product",
            confidence=0.85
        ))
        
        mock_agent.register_agent = Mock()
        mock_agent.get_capabilities = Mock(return_value=["product_knowledge"])
        
        return mock_agent


class TestEnvironmentManager:
    """Manage test environment and configuration."""
    
    def __init__(self):
        self.original_env = {}
        self.temp_files = []
    
    def set_env_var(self, key: str, value: str):
        """Set environment variable for testing."""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    def create_temp_env_file(self, env_vars: Dict[str, str]) -> str:
        """Create temporary .env file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
            temp_path = f.name
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup(self):
        """Cleanup test environment."""
        # Restore original environment variables
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
        
        self.temp_files.clear()
        self.original_env.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class PerformanceTimer:
    """Utility for measuring performance in tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop timing."""
        self.end_time = time.time()
        return self
    
    def checkpoint(self, name: str):
        """Add a checkpoint with given name."""
        self.checkpoints[name] = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time between start and stop."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        if self.end_time is None:
            raise ValueError("Timer not stopped")
        return self.end_time - self.start_time
    
    def checkpoint_elapsed(self, name: str) -> float:
        """Get elapsed time since checkpoint."""
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint '{name}' not found")
        if self.start_time is None:
            raise ValueError("Timer not started")
        return self.checkpoints[name] - self.start_time
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AsyncTestUtils:
    """Utilities for async testing."""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Operation timed out after {timeout} seconds")
    
    @staticmethod
    async def run_concurrent(coros: List, max_concurrent: int = 5):
        """Run coroutines concurrently with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_coro(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(*[limited_coro(coro) for coro in coros])
    
    @staticmethod
    def create_mock_async_context_manager():
        """Create a mock async context manager."""
        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_cm)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        return mock_cm


class TestDataValidation:
    """Validation utilities for test data."""
    
    @staticmethod
    def validate_json_serializable(data: Any) -> bool:
        """Check if data is JSON serializable."""
        try:
            json.dumps(data, default=str)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def validate_api_response_structure(response: Dict[str, Any]) -> bool:
        """Validate API response has expected structure."""
        required_fields = ["success", "request_id", "response", "confidence_score"]
        return all(field in response for field in required_fields)
    
    @staticmethod
    def validate_conversation_flow_structure(flow: List[Dict[str, Any]]) -> bool:
        """Validate conversation flow structure."""
        if not isinstance(flow, list):
            return False
        
        required_step_fields = ["step", "agent", "action"]
        for step in flow:
            if not all(field in step for field in required_step_fields):
                return False
        
        return True
    
    @staticmethod
    def validate_agent_capabilities(capabilities: List[str]) -> bool:
        """Validate agent capabilities list."""
        if not isinstance(capabilities, list):
            return False
        
        valid_capabilities = [
            "customer_communication", "system_diagnostics", "product_knowledge",
            "technical_analysis", "troubleshooting", "configuration"
        ]
        
        return all(cap in valid_capabilities for cap in capabilities)


class MockResponseBuilder:
    """Builder pattern for creating mock responses."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder to default state."""
        self._data = {
            "success": True,
            "request_id": str(uuid.uuid4()),
            "response": "Default mock response",
            "confidence_score": 0.8,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 1.5,
            "hop_count": 1
        }
        return self
    
    def with_success(self, success: bool):
        """Set success status."""
        self._data["success"] = success
        return self
    
    def with_confidence(self, confidence: float):
        """Set confidence score."""
        self._data["confidence_score"] = confidence
        return self
    
    def with_agents(self, agents: List[str]):
        """Set involved agents."""
        self._data["agents_involved"] = agents
        return self
    
    def with_flow(self, flow: List[Dict[str, Any]]):
        """Set conversation flow."""
        self._data["conversation_flow"] = flow
        self._data["hop_count"] = len(flow)
        return self
    
    def with_error(self, error_message: str):
        """Set error state."""
        self._data.update({
            "success": False,
            "response": error_message,
            "confidence_score": 0.1,
            "error_details": error_message
        })
        return self
    
    def with_processing_time(self, time_seconds: float):
        """Set processing time."""
        self._data["total_processing_time"] = time_seconds
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the mock response."""
        return self._data.copy()


# Pytest fixtures for common test utilities
@pytest.fixture
def test_data_generator():
    """Fixture providing test data generator."""
    return TestDataGenerator()


@pytest.fixture
def test_assertions():
    """Fixture providing test assertions."""
    return TestAssertions()


@pytest.fixture
def mock_agent_factory():
    """Fixture providing mock agent factory."""
    return MockAgentFactory()


@pytest.fixture
def test_env_manager():
    """Fixture providing test environment manager."""
    with TestEnvironmentManager() as manager:
        yield manager


@pytest.fixture
def performance_timer():
    """Fixture providing performance timer."""
    return PerformanceTimer()


@pytest.fixture
def async_test_utils():
    """Fixture providing async test utilities."""
    return AsyncTestUtils()


@pytest.fixture
def mock_response_builder():
    """Fixture providing mock response builder."""
    return MockResponseBuilder()


# Common test scenarios
class CommonTestScenarios:
    """Common test scenarios for reuse across test files."""
    
    @staticmethod
    async def simple_support_query_scenario(orchestrator, query: str = "How do I reset my password?"):
        """Simple support query scenario."""
        mock_response = TestDataGenerator.create_response(
            content="To reset your password, click 'Forgot Password' on the login page.",
            source_agent="Support",
            confidence=0.95
        )
        
        with patch.object(orchestrator.support_agent, 'process_message', return_value=mock_response):
            user_request = TestDataGenerator.create_user_request(query=query)
            result = await orchestrator.process_request(user_request)
            return result
    
    @staticmethod
    async def technical_collaboration_scenario(orchestrator, query: str = "Database error 500"):
        """Technical collaboration scenario."""
        support_response = TestDataGenerator.create_response(
            content="Let me get technical assistance for this error.",
            source_agent="Support",
            confidence=0.4,
            suggested_next_agent="Technical"
        )
        
        technical_response = TestDataGenerator.create_response(
            content="Database error analysis: Connection pool exhausted.",
            source_agent="Technical",
            confidence=0.9
        )
        
        with patch.object(orchestrator.support_agent, 'process_message', return_value=support_response), \
             patch.object(orchestrator.technical_agent, 'process_message', return_value=technical_response), \
             patch.object(orchestrator, '_select_next_agent', side_effect=["Technical", None]):
            
            user_request = TestDataGenerator.create_user_request(query=query, priority="high")
            result = await orchestrator.process_request(user_request)
            return result
    
    @staticmethod
    async def error_scenario(orchestrator, error_message: str = "Simulated error"):
        """Error handling scenario."""
        with patch.object(orchestrator.support_agent, 'process_message', 
                         side_effect=Exception(error_message)):
            user_request = TestDataGenerator.create_user_request(query="Error test")
            result = await orchestrator.process_request(user_request)
            return result 