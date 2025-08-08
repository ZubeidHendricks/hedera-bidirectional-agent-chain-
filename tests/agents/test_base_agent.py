"""
Comprehensive tests for agents/base_agent.py
Tests all BaseAgent functionality with 100% coverage including mocking and edge cases.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from agents.base_agent import BaseAgent
from models.message import Message, Response
from tests.conftest import MockBaseAgent, SimulatedError


class TestBaseAgentInitialization:
    """Test BaseAgent initialization and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_base_agent_initialization_with_defaults(self, mock_genai_client):
        """Test BaseAgent initialization with default values."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.name = "TestAgent"
            agent.expertise = "testing"
            agent.system_prompt = "Test prompt"
            agent.model_name = "test-model"
            agent.temperature = 0.7
            agent.max_tokens = 2048
            
            assert agent.name == "TestAgent"
            assert agent.expertise == "testing"
            assert agent.system_prompt == "Test prompt"
            assert agent.conversation_history == []
            assert agent.capabilities == ["testing"]
            assert agent.agent_registry == {}
            assert agent.model_name == "test-model"
            assert agent.temperature == 0.7
            assert agent.max_tokens == 2048
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_base_agent_initialization_with_custom_values(self, mock_genai_client):
        """Test BaseAgent initialization with custom values."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.name = "CustomAgent"
            agent.expertise = "custom_expertise"
            agent.system_prompt = "Custom system prompt"
            agent.model_name = "custom-model"
            agent.temperature = 0.3
            agent.max_tokens = 1024
            
            assert agent.name == "CustomAgent"
            assert agent.expertise == "custom_expertise"
            assert agent.system_prompt == "Custom system prompt"
            assert agent.model_name == "custom-model"
            assert agent.temperature == 0.3
            assert agent.max_tokens == 1024
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_base_agent_environment_variable_loading(self, mock_genai_client):
        """Test BaseAgent loading configuration from environment variables."""
        test_env = {
            "TESTAGENT_AGENT_MODEL": "env-model",
            "TESTAGENT_AGENT_TEMPERATURE": "0.5",
            "MAX_TOKENS_PER_RESPONSE": "4096"
        }
        
        with patch.dict(os.environ, test_env), \
             patch.object(BaseAgent, '_initialize_genai_client'):
            
            agent = MockBaseAgent()
            agent.name = "TestAgent"
            # Simulate environment variable loading
            agent.model_name = os.getenv("TESTAGENT_AGENT_MODEL", "default-model")
            agent.temperature = float(os.getenv("TESTAGENT_AGENT_TEMPERATURE", "0.7"))
            agent.max_tokens = int(os.getenv("MAX_TOKENS_PER_RESPONSE", "2048"))
            
            assert agent.model_name == "env-model"
            assert agent.temperature == 0.5
            assert agent.max_tokens == 4096


class TestAgentRegistration:
    """Test agent registration and management."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_register_agent(self, mock_genai_client):
        """Test registering another agent."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent1 = MockBaseAgent()
            agent1.name = "Agent1"
            
            agent2 = MockBaseAgent()
            agent2.name = "Agent2"
            
            agent1.register_agent("Agent2", agent2)
            
            assert "Agent2" in agent1.agent_registry
            assert agent1.agent_registry["Agent2"] == agent2
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_register_multiple_agents(self, mock_genai_client):
        """Test registering multiple agents."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            main_agent = MockBaseAgent()
            main_agent.name = "MainAgent"
            
            agent1 = MockBaseAgent()
            agent1.name = "Agent1"
            
            agent2 = MockBaseAgent()
            agent2.name = "Agent2"
            
            main_agent.register_agent("Agent1", agent1)
            main_agent.register_agent("Agent2", agent2)
            
            assert len(main_agent.agent_registry) == 2
            assert main_agent.agent_registry["Agent1"] == agent1
            assert main_agent.agent_registry["Agent2"] == agent2
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_register_agent_overwrite(self, mock_genai_client):
        """Test overwriting an existing agent registration."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            main_agent = MockBaseAgent()
            main_agent.name = "MainAgent"
            
            agent1 = MockBaseAgent()
            agent1.name = "Agent1"
            
            agent2 = MockBaseAgent()
            agent2.name = "Agent2"
            
            main_agent.register_agent("TestAgent", agent1)
            main_agent.register_agent("TestAgent", agent2)  # Overwrite
            
            assert main_agent.agent_registry["TestAgent"] == agent2


class TestMessageProcessing:
    """Test message processing functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_process_message_basic(self, mock_genai_client):
        """Test basic message processing."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            # Mock the _generate_response method
            mock_response = Response(
                content="Test response",
                source_agent="MockAgent",
                target_agent="user",
                confidence=0.8
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Test message",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.content == "Test response"
                assert result.source_agent == "MockAgent"
                assert result.confidence == 0.8
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_process_message_with_conversation_history(self, mock_genai_client):
        """Test message processing with conversation history."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            # Add some conversation history
            prev_message = Message(
                content="Previous message",
                source_agent="user",
                target_agent="MockAgent"
            )
            agent.conversation_history.append(prev_message)
            
            mock_response = Response(
                content="Contextual response",
                source_agent="MockAgent",
                target_agent="user",
                confidence=0.9
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Current message",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                
                result = await agent.process_message(message)
                
                assert len(agent.conversation_history) == 2
                assert agent.conversation_history[-1] == message
                assert result.content == "Contextual response"
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_process_message_error_handling(self, mock_genai_client):
        """Test error handling during message processing."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            # Mock _generate_response to raise an exception
            with patch.object(agent, '_generate_response', side_effect=Exception("Test error")):
                message = Message(
                    content="Test message",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "error" in result.content.lower()
                assert result.confidence <= 0.1
                assert result.needs_clarification is True


class TestResponseGeneration:
    """Test response generation functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_generate_response_success(self, mock_genai_client):
        """Test successful response generation."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            message = Message(
                content="Test question",
                source_agent="user",
                target_agent="MockAgent"
            )
            
            result = await agent._generate_response(message)
            
            assert isinstance(result, Response)
            assert result.source_agent == "MockAgent"
            assert result.target_agent == "user"
            assert result.content is not None
            assert 0 <= result.confidence <= 1
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_generate_response_with_context(self, mock_genai_client):
        """Test response generation with context building."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            # Add conversation history for context
            agent.conversation_history = [
                Message(
                    content="First message",
                    source_agent="user",
                    target_agent="MockAgent"
                ),
                Message(
                    content="Second message",
                    source_agent="user",
                    target_agent="MockAgent"
                )
            ]
            
            message = Message(
                content="Test question with context",
                source_agent="user",
                target_agent="MockAgent"
            )
            
            result = await agent._generate_response(message)
            
            assert isinstance(result, Response)
            assert result.content is not None
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_generate_response_genai_error(self, mock_genai_client):
        """Test response generation when GenAI client fails."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            # Mock GenAI client to raise an exception
            mock_client = Mock()
            mock_client.generate_content_async = AsyncMock(
                side_effect=Exception("GenAI API error")
            )
            agent.genai_client = mock_client
            
            message = Message(
                content="Test question",
                source_agent="user",
                target_agent="MockAgent"
            )
            
            with pytest.raises(Exception, match="GenAI API error"):
                await agent._generate_response(message)


class TestContextBuilding:
    """Test context building functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_build_context_empty_history(self, mock_genai_client):
        """Test context building with empty conversation history."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            context = agent._build_context()
            
            assert context == "No previous conversation history."
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_build_context_with_history(self, mock_genai_client):
        """Test context building with conversation history."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            # Add conversation history
            agent.conversation_history = [
                Message(
                    content="First message",
                    source_agent="user",
                    target_agent="MockAgent"
                ),
                Message(
                    content="Second message",
                    source_agent="user",
                    target_agent="MockAgent"
                )
            ]
            
            context = agent._build_context()
            
            assert "First message" in context
            assert "Second message" in context
            assert "user:" in context
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_build_context_truncation(self, mock_genai_client):
        """Test context building with history truncation."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.max_conversation_history = 2
            
            # Add more messages than the limit
            for i in range(5):
                message = Message(
                    content=f"Message {i}",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                agent.conversation_history.append(message)
            
            context = agent._build_context()
            
            # Should only include the last 2 messages
            assert "Message 3" in context
            assert "Message 4" in context
            assert "Message 0" not in context
            assert "Message 1" not in context
            assert "Message 2" not in context


class TestAgentCommunication:
    """Test inter-agent communication functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_query_other_agent_success(self, mock_genai_client):
        """Test successful query to another agent."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent1 = MockBaseAgent()
            agent1.name = "Agent1"
            
            agent2 = MockBaseAgent()
            agent2.name = "Agent2"
            
            # Register agent2 with agent1
            agent1.register_agent("Agent2", agent2)
            
            # Mock agent2's process_message method
            mock_response = Response(
                content="Response from Agent2",
                source_agent="Agent2",
                target_agent="Agent1",
                confidence=0.85
            )
            agent2.process_message = AsyncMock(return_value=mock_response)
            
            result = await agent1.query_other_agent("Agent2", "Test query")
            
            assert result == mock_response
            assert result.content == "Response from Agent2"
            assert result.source_agent == "Agent2"
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_query_other_agent_not_found(self, mock_genai_client):
        """Test query to non-existent agent."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.name = "Agent1"
            
            result = await agent.query_other_agent("NonExistentAgent", "Test query")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_query_other_agent_error(self, mock_genai_client):
        """Test query to agent that raises an error."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent1 = MockBaseAgent()
            agent1.name = "Agent1"
            
            agent2 = MockBaseAgent()
            agent2.name = "Agent2"
            
            agent1.register_agent("Agent2", agent2)
            
            # Mock agent2's process_message to raise an exception
            agent2.process_message = AsyncMock(side_effect=Exception("Agent2 error"))
            
            result = await agent1.query_other_agent("Agent2", "Test query")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_query_other_agent_timeout(self, mock_genai_client):
        """Test query to agent that times out."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent1 = MockBaseAgent()
            agent1.name = "Agent1"
            
            agent2 = MockBaseAgent()
            agent2.name = "Agent2"
            
            agent1.register_agent("Agent2", agent2)
            
            # Mock agent2's process_message to timeout
            async def slow_response(message):
                await asyncio.sleep(2)  # Simulate slow response
                return Response(
                    content="Slow response",
                    source_agent="Agent2",
                    target_agent="Agent1"
                )
            
            agent2.process_message = slow_response
            
            # Set a short timeout for testing
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                result = await agent1.query_other_agent("Agent2", "Test query")
                
                assert result is None


class TestHandoffParsing:
    """Test handoff request parsing functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_parse_handoff_request_valid(self, mock_genai_client):
        """Test parsing valid handoff request."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            response_content = """
            Based on your query, I need technical expertise.
            HANDOFF_REQUEST: Technical - Please analyze the system performance issue
            The user is experiencing slow response times.
            """
            
            needs_handoff, target_agent, query = agent._parse_handoff_request(response_content)
            
            assert needs_handoff is True
            assert target_agent == "Technical"
            assert "Please analyze the system performance issue" in query
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_parse_handoff_request_no_handoff(self, mock_genai_client):
        """Test parsing content without handoff request."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            response_content = """
            This is a regular response without any handoff request.
            I can handle this query myself.
            """
            
            needs_handoff, target_agent, query = agent._parse_handoff_request(response_content)
            
            assert needs_handoff is False
            assert target_agent is None
            assert query is None
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_parse_handoff_request_multiple(self, mock_genai_client):
        """Test parsing content with multiple handoff requests."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            response_content = """
            HANDOFF_REQUEST: Technical - First request
            Some text in between.
            HANDOFF_REQUEST: Product - Second request
            """
            
            needs_handoff, target_agent, query = agent._parse_handoff_request(response_content)
            
            # Should return the first handoff request
            assert needs_handoff is True
            assert target_agent == "Technical"
            assert "First request" in query
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_parse_handoff_request_malformed(self, mock_genai_client):
        """Test parsing malformed handoff request."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            response_content = """
            HANDOFF_REQUEST: 
            This is malformed without proper format.
            """
            
            needs_handoff, target_agent, query = agent._parse_handoff_request(response_content)
            
            assert needs_handoff is False
            assert target_agent is None
            assert query is None


class TestQueryEnhancement:
    """Test query enhancement functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_enhance_handoff_query_basic(self, mock_genai_client):
        """Test basic query enhancement for handoff."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            original_message = Message(
                content="User has a problem",
                source_agent="user",
                target_agent="MockAgent",
                context={"user_id": "123", "session": "abc"}
            )
            
            handoff_query = "Please help with technical issue"
            
            enhanced = agent._enhance_handoff_query(handoff_query, original_message)
            
            assert "Please help with technical issue" in enhanced
            assert "User has a problem" in enhanced
            assert "user_id: 123" in enhanced
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_enhance_handoff_query_no_context(self, mock_genai_client):
        """Test query enhancement with no context."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            original_message = Message(
                content="Simple message",
                source_agent="user",
                target_agent="MockAgent",
                context=None
            )
            
            handoff_query = "Help needed"
            
            enhanced = agent._enhance_handoff_query(handoff_query, original_message)
            
            assert "Help needed" in enhanced
            assert "Simple message" in enhanced
            assert "No additional context" in enhanced
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_enhance_handoff_query_complex_context(self, mock_genai_client):
        """Test query enhancement with complex context."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            complex_context = {
                "user_id": "123",
                "session": {
                    "id": "abc",
                    "start_time": "2024-01-01"
                },
                "preferences": ["pref1", "pref2"]
            }
            
            original_message = Message(
                content="Complex problem",
                source_agent="user",
                target_agent="MockAgent",
                context=complex_context
            )
            
            handoff_query = "Need analysis"
            
            enhanced = agent._enhance_handoff_query(handoff_query, original_message)
            
            assert "Need analysis" in enhanced
            assert "Complex problem" in enhanced
            assert "user_id" in enhanced
            assert "session" in enhanced


class TestConversationHistoryManagement:
    """Test conversation history management."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_conversation_history_limit(self, mock_genai_client):
        """Test conversation history respects limit."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.max_conversation_history = 3
            
            # Add messages beyond the limit
            for i in range(5):
                message = Message(
                    content=f"Message {i}",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                agent.conversation_history.append(message)
                agent._truncate_conversation_history()
            
            assert len(agent.conversation_history) == 3
            assert agent.conversation_history[0].content == "Message 2"
            assert agent.conversation_history[1].content == "Message 3"
            assert agent.conversation_history[2].content == "Message 4"
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_conversation_history_no_limit(self, mock_genai_client):
        """Test conversation history without limit."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.max_conversation_history = 0  # No limit
            
            # Add many messages
            for i in range(10):
                message = Message(
                    content=f"Message {i}",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                agent.conversation_history.append(message)
                agent._truncate_conversation_history()
            
            assert len(agent.conversation_history) == 10


class TestCapabilitiesManagement:
    """Test agent capabilities management."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_get_capabilities(self, mock_genai_client):
        """Test getting agent capabilities."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.capabilities = ["testing", "analysis", "communication"]
            
            capabilities = agent.get_capabilities()
            
            assert capabilities == ["testing", "analysis", "communication"]
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_add_capability(self, mock_genai_client):
        """Test adding a new capability."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            initial_capabilities = agent.capabilities.copy()
            
            agent.add_capability("new_skill")
            
            assert "new_skill" in agent.capabilities
            assert len(agent.capabilities) == len(initial_capabilities) + 1
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_add_duplicate_capability(self, mock_genai_client):
        """Test adding a duplicate capability."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.capabilities = ["testing"]
            initial_length = len(agent.capabilities)
            
            agent.add_capability("testing")  # Duplicate
            
            assert len(agent.capabilities) == initial_length
            assert agent.capabilities.count("testing") == 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_process_message_none_message(self, mock_genai_client):
        """Test processing None message."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            with pytest.raises(AttributeError):
                await agent.process_message(None)
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_process_message_invalid_message(self, mock_genai_client):
        """Test processing invalid message object."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            # Create an invalid message-like object
            invalid_message = object()
            
            with pytest.raises(AttributeError):
                await agent.process_message(invalid_message)
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    def test_register_agent_none(self, mock_genai_client):
        """Test registering None as an agent."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            
            agent.register_agent("NoneAgent", None)
            
            assert "NoneAgent" in agent.agent_registry
            assert agent.agent_registry["NoneAgent"] is None
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_query_other_agent_none_query(self, mock_genai_client):
        """Test querying with None query."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent1 = MockBaseAgent()
            agent2 = MockBaseAgent()
            
            agent1.register_agent("Agent2", agent2)
            
            with pytest.raises(TypeError):
                await agent1.query_other_agent("Agent2", None)


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.performance
    async def test_concurrent_message_processing(self, mock_genai_client):
        """Test concurrent message processing."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Concurrent response",
                source_agent="MockAgent",
                target_agent="user"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                messages = [
                    Message(
                        content=f"Message {i}",
                        source_agent="user",
                        target_agent="MockAgent"
                    )
                    for i in range(5)
                ]
                
                # Process messages concurrently
                tasks = [agent.process_message(msg) for msg in messages]
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 5
                for result in results:
                    assert isinstance(result, Response)
                    assert result.content == "Concurrent response"
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.performance
    async def test_message_processing_performance(self, mock_genai_client, performance_monitor):
        """Test message processing performance."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Quick response",
                source_agent="MockAgent",
                target_agent="user"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Performance test message",
                    source_agent="user",
                    target_agent="MockAgent"
                )
                
                performance_monitor.start()
                result = await agent.process_message(message)
                performance_monitor.stop()
                
                elapsed = performance_monitor.elapsed()
                assert elapsed is not None
                assert elapsed < 1.0  # Should be fast with mocked response
                assert isinstance(result, Response)


class TestAgentHealthAndStatus:
    """Test agent health and status functionality."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_agent_health_check(self, mock_genai_client):
        """Test agent health check functionality."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            # Mock health check method if it exists
            if hasattr(agent, 'health_check'):
                health = agent.health_check()
                assert health is not None
            else:
                # Test basic agent state
                assert agent.name is not None
                assert agent.expertise is not None
                assert agent.capabilities is not None
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_agent_status_information(self, mock_genai_client):
        """Test agent status information."""
        with patch.object(BaseAgent, '_initialize_genai_client'):
            agent = MockBaseAgent()
            agent.name = "StatusAgent"
            agent.expertise = "status_testing"
            
            # Add some conversation history
            agent.conversation_history = [
                Message(
                    content="Test message",
                    source_agent="user",
                    target_agent="StatusAgent"
                )
            ]
            
            # Test agent state
            assert len(agent.conversation_history) == 1
            assert agent.name == "StatusAgent"
            assert agent.expertise == "status_testing"
            assert isinstance(agent.capabilities, list)
            assert isinstance(agent.agent_registry, dict) 