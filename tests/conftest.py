"""
Pytest configuration and fixtures for bidirectional agent chaining tests.
Provides comprehensive testing utilities and mocks for all components.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Set test environment variables before importing modules
os.environ.setdefault("GOOGLE_API_KEY", "test_api_key_12345")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.7")
os.environ.setdefault("MAX_CHAIN_HOPS", "5")
os.environ.setdefault("AGENT_RESPONSE_TIMEOUT", "30")

from models.message import (
    Message, Response, UserRequest, ChainResult, AgentState,
    MessagesState, AgentHandoff, ConversationFlow
)
from agents.base_agent import BaseAgent
from agents.support_agent import SupportAgent
from agents.technical_agent import TechnicalAgent
from agents.product_agent import ProductAgent
from chain.orchestrator import ChainOrchestrator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_env_vars():
    """Fixture providing test environment variables."""
    return {
        "GOOGLE_API_KEY": "test_api_key_12345",
        "ENVIRONMENT": "test",
        "DEBUG": "false",
        "CONFIDENCE_THRESHOLD": "0.7",
        "MAX_CHAIN_HOPS": "5",
        "AGENT_RESPONSE_TIMEOUT": "30",
        "MAX_CONVERSATION_HISTORY": "10",
        "ENABLE_PERFORMANCE_MONITORING": "true",
        "ENABLE_AGENT_LOGS": "true",
        "ENABLE_DETAILED_FLOW_LOGGING": "true",
        "ENABLE_DYNAMIC_ROUTING": "true",
        "ENABLE_CONTENT_ANALYSIS_ROUTING": "true"
    }


@pytest.fixture
def mock_genai_client():
    """Mock Google GenAI client for testing."""
    with patch('google.genai.Client') as mock_client_class:
        
        # Mock the client instance
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the model instance and methods
        mock_model_instance = Mock()
        mock_client_instance.models = Mock()
        mock_client_instance.aio = Mock()
        
        # Mock generate_content response
        mock_response = Mock()
        mock_response.text = "Test response from mock AI"
        mock_response.finish_reason = "stop"
        
        # Mock both sync and async generate methods
        mock_client_instance.models.generate_content = Mock(return_value=mock_response)
        mock_client_instance.aio.models = Mock()
        mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        yield {
            'client_class': mock_client_class,
            'client_instance': mock_client_instance,
            'model_instance': mock_model_instance,
            'response': mock_response
        }


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        content="Test message content",
        source_agent="user",
        target_agent="Support",
        message_type="query",
        priority="medium",
        context={"test_key": "test_value"},
        conversation_id="test_conversation_123"
    )


@pytest.fixture
def sample_response():
    """Create a sample response for testing."""
    return Response(
        content="Test response content",
        source_agent="Support",
        target_agent="user",
        confidence=0.8,
        needs_clarification=False,
        suggested_next_agent="Technical",
        reasoning="Test reasoning",
        metadata={"test_meta": "test_value"}
    )


@pytest.fixture
def sample_user_request():
    """Create a sample user request for testing."""
    return UserRequest(
        query="Test user query",
        user_id="test_user_123",
        priority="medium",
        expected_response_format="text",
        max_processing_time=60,
        context={"user_context": "test"}
    )


@pytest.fixture
def sample_chain_result():
    """Create a sample chain result for testing."""
    return ChainResult(
        request_id="test_request_123",
        response="Test chain response",
        agents_involved=["Support", "Technical"],
        conversation_flow=[
            {"step": 1, "agent": "Support", "action": "processed"},
            {"step": 2, "agent": "Technical", "action": "processed"}
        ],
        total_processing_time=2.5,
        success=True,
        confidence_score=0.85
    )


@pytest.fixture
def sample_agent_state():
    """Create a sample agent state for testing."""
    from langchain_core.messages import HumanMessage
    
    return AgentState(
        messages=[HumanMessage(content="Test message")],
        agent_context={"test": "context"},
        request_id="test_request_123",
        user_id="test_user_123",
        current_agent="Support",
        next_agent="Technical",
        conversation_flow=[],
        processing_metadata={}
    )


@pytest.fixture
async def mock_support_agent(mock_genai_client):
    """Create a mocked support agent for testing."""
    with patch.object(SupportAgent, '_initialize_genai_client'):
        agent = SupportAgent()
        agent.client = mock_genai_client['client_instance']
        return agent


@pytest.fixture
async def mock_technical_agent(mock_genai_client):
    """Create a mocked technical agent for testing."""
    with patch.object(TechnicalAgent, '_initialize_genai_client'):
        agent = TechnicalAgent()
        agent.client = mock_genai_client['client_instance']
        return agent


@pytest.fixture
async def mock_product_agent(mock_genai_client):
    """Create a mocked product agent for testing."""
    with patch.object(ProductAgent, '_initialize_genai_client'):
        agent = ProductAgent()
        agent.client = mock_genai_client['client_instance']
        return agent


@pytest.fixture
async def mock_orchestrator(mock_genai_client):
    """Create a mocked orchestrator for testing."""
    with patch.object(SupportAgent, '_initialize_genai_client'), \
         patch.object(TechnicalAgent, '_initialize_genai_client'), \
         patch.object(ProductAgent, '_initialize_genai_client'):
        orchestrator = ChainOrchestrator()
        
        # Mock the agent instances
        for agent in orchestrator.agents.values():
            agent.client = mock_genai_client['client_instance']
        
        return orchestrator


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    test_datetime = datetime(2024, 1, 1, 12, 0, 0)
    with patch('models.message.datetime') as mock_dt:
        mock_dt.now.return_value = test_datetime
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield mock_dt


@pytest.fixture
def mock_uuid():
    """Mock UUID for consistent testing."""
    test_uuid = "test-uuid-12345"
    with patch('uuid.uuid4') as mock_uuid_func:
        mock_uuid_func.return_value = Mock(hex=test_uuid)
        with patch('models.message.uuid.uuid4') as mock_msg_uuid:
            mock_msg_uuid.return_value = test_uuid
            yield mock_uuid_func


@pytest.fixture
def temporary_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("GOOGLE_API_KEY=test_key\n")
        f.write("ENVIRONMENT=test\n")
        f.write("DEBUG=false\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


class MockResponse:
    """Mock response for testing HTTP requests."""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_http_response():
    """Create mock HTTP response for testing."""
    return MockResponse({"message": "success"}, 200)


# Utility functions for tests
def create_test_message(
    content: str = "Test content",
    source_agent: str = "user",
    target_agent: str = "Support",
    **kwargs
) -> Message:
    """Create a test message with default values."""
    defaults = {
        "message_type": "query",
        "priority": "medium",
        "context": {"test": True},
        "conversation_id": "test_conversation"
    }
    defaults.update(kwargs)
    
    return Message(
        content=content,
        source_agent=source_agent,
        target_agent=target_agent,
        **defaults
    )


def create_test_response(
    content: str = "Test response",
    source_agent: str = "Support",
    confidence: float = 0.8,
    **kwargs
) -> Response:
    """Create a test response with default values."""
    defaults = {
        "target_agent": "user",
        "needs_clarification": False,
        "reasoning": "Test reasoning"
    }
    defaults.update(kwargs)
    
    return Response(
        content=content,
        source_agent=source_agent,
        confidence=confidence,
        **defaults
    )


def assert_message_fields(message: Message, expected_fields: Dict[str, Any]):
    """Assert that message has expected field values."""
    for field, expected_value in expected_fields.items():
        actual_value = getattr(message, field)
        assert actual_value == expected_value, f"Field {field}: expected {expected_value}, got {actual_value}"


def assert_response_fields(response: Response, expected_fields: Dict[str, Any]):
    """Assert that response has expected field values."""
    for field, expected_value in expected_fields.items():
        actual_value = getattr(response, field)
        assert actual_value == expected_value, f"Field {field}: expected {expected_value}, got {actual_value}"


# Mock classes for specific testing scenarios
class MockBaseAgent(BaseAgent):
    """Mock base agent for testing abstract methods."""
    
    def __init__(self):
        # Skip parent __init__ to avoid GenAI initialization
        self.name = "MockAgent"
        self.expertise = "testing"
        self.system_prompt = "Mock system prompt"
        self.conversation_history = []
        self.capabilities = ["testing"]
        self.agent_registry = {}
        self.model_name = "test-model"
        self.temperature = 0.7
        self.max_tokens = 1024
        self.client = Mock()
    
    async def query_other_agent(self, target_agent: str, query: str) -> Optional[Response]:
        """Mock implementation of abstract method."""
        return Response(
            content=f"Mock response from {target_agent} for query: {query}",
            source_agent=target_agent,
            target_agent=self.name,
            confidence=0.8
        )


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceMonitor()


# Async testing utilities
async def run_async_test(coro):
    """Run async coroutine in test environment."""
    return await coro


# Validation utilities
def validate_message_structure(message: Message):
    """Validate that message has proper structure."""
    assert hasattr(message, 'message_id')
    assert hasattr(message, 'timestamp')
    assert hasattr(message, 'content')
    assert hasattr(message, 'source_agent')
    assert hasattr(message, 'target_agent')
    assert message.message_id is not None
    assert message.content is not None
    assert message.source_agent is not None
    assert message.target_agent is not None


def validate_response_structure(response: Response):
    """Validate that response has proper structure."""
    assert hasattr(response, 'response_id')
    assert hasattr(response, 'timestamp')
    assert hasattr(response, 'content')
    assert hasattr(response, 'source_agent')
    assert hasattr(response, 'confidence')
    assert response.response_id is not None
    assert response.content is not None
    assert response.source_agent is not None
    assert 0 <= response.confidence <= 1


# Error simulation utilities
class SimulatedError(Exception):
    """Custom exception for simulating errors in tests."""
    pass


@pytest.fixture
def error_simulator():
    """Fixture for simulating various error conditions."""
    
    class ErrorSimulator:
        @staticmethod
        def network_error():
            raise ConnectionError("Simulated network error")
        
        @staticmethod
        def timeout_error():
            raise TimeoutError("Simulated timeout error")
        
        @staticmethod
        def api_error():
            raise SimulatedError("Simulated API error")
        
        @staticmethod
        def validation_error():
            raise ValueError("Simulated validation error")
    
    return ErrorSimulator() 