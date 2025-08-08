"""
Comprehensive tests for models/message.py
Tests all message classes, data structures, and utility functions with 100% coverage.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any, List
import uuid

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from models.message import (
    Message, Response, UserRequest, ChainResult, AgentState,
    MessagesState, AgentHandoff, ConversationFlow,
    messages_to_langchain, langchain_to_messages, create_agent_state
)
from tests.conftest import (
    create_test_message, create_test_response, validate_message_structure,
    validate_response_structure
)


class TestMessage:
    """Test Message class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_message_creation_with_defaults(self):
        """Test message creation with default values."""
        message = Message(
            content="Test content",
            source_agent="user",
            target_agent="Support"
        )
        
        validate_message_structure(message)
        assert message.content == "Test content"
        assert message.source_agent == "user"
        assert message.target_agent == "Support"
        assert message.message_type == "query"  # default
        assert message.priority == "medium"  # default
        assert message.requires_response is True  # default
        assert message.context is None  # default
        assert message.conversation_id is None  # default
        assert isinstance(message.message_id, str)
        assert isinstance(message.timestamp, datetime)
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_message_creation_with_all_fields(self):
        """Test message creation with all fields specified."""
        test_context = {"key": "value", "number": 42}
        test_time = datetime.now()
        test_id = str(uuid.uuid4())
        
        message = Message(
            message_id=test_id,
            timestamp=test_time,
            content="Full test content",
            source_agent="Technical",
            target_agent="Product",
            message_type="handoff",
            priority="high",
            context=test_context,
            requires_response=False,
            conversation_id="conv_123"
        )
        
        assert message.message_id == test_id
        assert message.timestamp == test_time
        assert message.content == "Full test content"
        assert message.source_agent == "Technical"
        assert message.target_agent == "Product"
        assert message.message_type == "handoff"
        assert message.priority == "high"
        assert message.context == test_context
        assert message.requires_response is False
        assert message.conversation_id == "conv_123"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_message_type_validation(self):
        """Test message type validation with all valid types."""
        valid_types = [
            "query", "response", "notification", "command", "handoff",
            "analysis", "consultation", "synthesis", "collaboration",
            "technical_analysis", "product_analysis", "support_analysis",
            "technical_product_analysis", "product_technical_analysis"
        ]
        
        for msg_type in valid_types:
            message = Message(
                content="Test",
                source_agent="user",
                target_agent="Support",
                message_type=msg_type
            )
            assert message.message_type == msg_type
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_priority_validation(self):
        """Test priority validation with all valid priorities."""
        valid_priorities = ["low", "medium", "high", "critical"]
        
        for priority in valid_priorities:
            message = Message(
                content="Test",
                source_agent="user",
                target_agent="Support",
                priority=priority
            )
            assert message.priority == priority
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_to_langchain_message_user(self):
        """Test conversion to LangChain message for user source."""
        context = {"test": "context"}
        message = Message(
            content="User message",
            source_agent="user",
            target_agent="Support",
            context=context
        )
        
        lc_message = message.to_langchain_message()
        
        assert isinstance(lc_message, HumanMessage)
        assert lc_message.content == "User message"
        assert lc_message.additional_kwargs["message_id"] == message.message_id
        assert lc_message.additional_kwargs["source_agent"] == "user"
        assert lc_message.additional_kwargs["target_agent"] == "Support"
        assert lc_message.additional_kwargs["context"] == context
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_to_langchain_message_agent(self):
        """Test conversion to LangChain message for agent source."""
        message = Message(
            content="Agent message",
            source_agent="Support",
            target_agent="user",
            context=None
        )
        
        lc_message = message.to_langchain_message()
        
        assert isinstance(lc_message, AIMessage)
        assert lc_message.content == "Agent message"
        assert lc_message.additional_kwargs["source_agent"] == "Support"
        assert lc_message.additional_kwargs["target_agent"] == "user"
        assert lc_message.additional_kwargs["context"] == {}
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_message_with_empty_context(self):
        """Test message creation with empty context."""
        message = Message(
            content="Test",
            source_agent="user",
            target_agent="Support",
            context={}
        )
        
        lc_message = message.to_langchain_message()
        assert lc_message.additional_kwargs["context"] == {}


class TestResponse:
    """Test Response class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_response_creation_with_defaults(self):
        """Test response creation with default values."""
        response = Response(
            content="Test response",
            source_agent="Support",
            target_agent="user"
        )
        
        validate_response_structure(response)
        assert response.content == "Test response"
        assert response.source_agent == "Support"
        assert response.target_agent == "user"
        assert response.confidence == 0.8  # default
        assert response.needs_clarification is False  # default
        assert response.suggested_next_agent is None  # default
        assert response.reasoning is None  # default
        assert response.metadata is None  # default
        assert isinstance(response.response_id, str)
        assert isinstance(response.timestamp, datetime)
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_response_creation_with_all_fields(self):
        """Test response creation with all fields specified."""
        test_metadata = {"processing_time": 1.5, "model": "test-model"}
        test_time = datetime.now()
        test_id = str(uuid.uuid4())
        
        response = Response(
            response_id=test_id,
            timestamp=test_time,
            content="Complete response",
            source_agent="Technical",
            target_agent="Support",
            confidence=0.95,
            needs_clarification=True,
            suggested_next_agent="Product",
            reasoning="Detailed reasoning",
            metadata=test_metadata
        )
        
        assert response.response_id == test_id
        assert response.timestamp == test_time
        assert response.content == "Complete response"
        assert response.source_agent == "Technical"
        assert response.target_agent == "Support"
        assert response.confidence == 0.95
        assert response.needs_clarification is True
        assert response.suggested_next_agent == "Product"
        assert response.reasoning == "Detailed reasoning"
        assert response.metadata == test_metadata
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_confidence_validation_bounds(self):
        """Test confidence score validation bounds."""
        # Test valid confidence values
        valid_confidences = [0.0, 0.5, 1.0, 0.75, 0.123]
        
        for confidence in valid_confidences:
            response = Response(
                content="Test",
                source_agent="Support",
                target_agent="user",
                confidence=confidence
            )
            assert response.confidence == confidence
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_to_langchain_message_response(self):
        """Test conversion to LangChain AI message."""
        metadata = {"test": "metadata"}
        response = Response(
            content="Response content",
            source_agent="Technical",
            target_agent="user",
            confidence=0.9,
            needs_clarification=True,
            suggested_next_agent="Product",
            reasoning="Test reasoning",
            metadata=metadata
        )
        
        lc_message = response.to_langchain_message()
        
        assert isinstance(lc_message, AIMessage)
        assert lc_message.content == "Response content"
        kwargs = lc_message.additional_kwargs
        assert kwargs["response_id"] == response.response_id
        assert kwargs["source_agent"] == "Technical"
        assert kwargs["target_agent"] == "user"
        assert kwargs["confidence"] == 0.9
        assert kwargs["needs_clarification"] is True
        assert kwargs["suggested_next_agent"] == "Product"
        assert kwargs["reasoning"] == "Test reasoning"
        assert kwargs["metadata"] == metadata
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_response_with_none_metadata(self):
        """Test response with None metadata."""
        response = Response(
            content="Test",
            source_agent="Support",
            target_agent="user",
            metadata=None
        )
        
        lc_message = response.to_langchain_message()
        assert lc_message.additional_kwargs["metadata"] == {}


class TestUserRequest:
    """Test UserRequest class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_user_request_creation_with_defaults(self):
        """Test user request creation with default values."""
        request = UserRequest(
            query="Test query",
            user_id="user_123"
        )
        
        assert request.query == "Test query"
        assert request.user_id == "user_123"
        assert request.priority == "medium"  # default
        assert request.expected_response_format == "text"  # default
        assert request.max_processing_time == 60  # default
        assert request.context is None  # default
        assert isinstance(request.request_id, str)
        assert isinstance(request.timestamp, datetime)
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_user_request_creation_with_all_fields(self):
        """Test user request creation with all fields specified."""
        test_context = {"user_type": "premium", "session": "abc123"}
        test_time = datetime.now()
        test_id = str(uuid.uuid4())
        
        request = UserRequest(
            request_id=test_id,
            timestamp=test_time,
            query="Complex query",
            user_id="premium_user_456",
            context=test_context,
            priority="critical",
            expected_response_format="json",
            max_processing_time=120
        )
        
        assert request.request_id == test_id
        assert request.timestamp == test_time
        assert request.query == "Complex query"
        assert request.user_id == "premium_user_456"
        assert request.context == test_context
        assert request.priority == "critical"
        assert request.expected_response_format == "json"
        assert request.max_processing_time == 120
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_priority_validation_user_request(self):
        """Test priority validation for user request."""
        valid_priorities = ["low", "medium", "high", "critical"]
        
        for priority in valid_priorities:
            request = UserRequest(
                query="Test",
                user_id="user_123",
                priority=priority
            )
            assert request.priority == priority
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_response_format_validation(self):
        """Test response format validation."""
        valid_formats = ["text", "structured", "json"]
        
        for format_type in valid_formats:
            request = UserRequest(
                query="Test",
                user_id="user_123",
                expected_response_format=format_type
            )
            assert request.expected_response_format == format_type
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_to_initial_message_default_target(self):
        """Test conversion to initial message with default target."""
        context = {"test": "context"}
        request = UserRequest(
            query="User question",
            user_id="user_123",
            priority="high",
            expected_response_format="structured",
            context=context
        )
        
        message = request.to_initial_message()
        
        assert isinstance(message, Message)
        assert message.content == "User question"
        assert message.source_agent == "user"
        assert message.target_agent == "Support"  # default
        assert message.message_type == "query"
        assert message.priority == "high"
        assert message.conversation_id == request.request_id
        
        expected_context = {
            "user_id": "user_123",
            "request_id": request.request_id,
            "expected_format": "structured",
            "test": "context"
        }
        assert message.context == expected_context
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_to_initial_message_custom_target(self):
        """Test conversion to initial message with custom target."""
        request = UserRequest(
            query="Technical question",
            user_id="user_123"
        )
        
        message = request.to_initial_message(target_agent="Technical")
        
        assert message.target_agent == "Technical"
        assert message.source_agent == "user"
        assert message.content == "Technical question"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_to_initial_message_none_context(self):
        """Test conversion to initial message with None context."""
        request = UserRequest(
            query="Test",
            user_id="user_123",
            context=None
        )
        
        message = request.to_initial_message()
        
        expected_context = {
            "user_id": "user_123",
            "request_id": request.request_id,
            "expected_format": "text"
        }
        assert message.context == expected_context


class TestChainResult:
    """Test ChainResult class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_chain_result_creation_with_defaults(self):
        """Test chain result creation with default values."""
        result = ChainResult(
            request_id="req_123",
            response="Test response",
            total_processing_time=2.5
        )
        
        assert result.request_id == "req_123"
        assert result.response == "Test response"
        assert result.total_processing_time == 2.5
        assert result.agents_involved == []  # default
        assert result.conversation_flow == []  # default
        assert result.success is True  # default
        assert result.confidence_score == 0.8  # default
        assert result.error_details is None  # default
        assert result.performance_metrics == {}  # default
        assert isinstance(result.result_id, str)
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_chain_result_creation_with_all_fields(self):
        """Test chain result creation with all fields specified."""
        agents = ["Support", "Technical", "Product"]
        flow = [
            {"step": 1, "agent": "Support", "action": "received"},
            {"step": 2, "agent": "Technical", "action": "processed"}
        ]
        metrics = {"total_hops": 3, "avg_confidence": 0.85}
        test_time = datetime.now()
        test_id = str(uuid.uuid4())
        
        result = ChainResult(
            result_id=test_id,
            timestamp=test_time,
            request_id="req_456",
            response="Complete response",
            agents_involved=agents,
            conversation_flow=flow,
            total_processing_time=4.2,
            success=False,
            confidence_score=0.95,
            error_details="Test error",
            performance_metrics=metrics
        )
        
        assert result.result_id == test_id
        assert result.timestamp == test_time
        assert result.request_id == "req_456"
        assert result.response == "Complete response"
        assert result.agents_involved == agents
        assert result.conversation_flow == flow
        assert result.total_processing_time == 4.2
        assert result.success is False
        assert result.confidence_score == 0.95
        assert result.error_details == "Test error"
        assert result.performance_metrics == metrics
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_confidence_score_validation_chain_result(self):
        """Test confidence score validation bounds for chain result."""
        valid_confidences = [0.0, 0.5, 1.0, 0.234]
        
        for confidence in valid_confidences:
            result = ChainResult(
                request_id="req_123",
                response="Test",
                total_processing_time=1.0,
                confidence_score=confidence
            )
            assert result.confidence_score == confidence


class TestAgentState:
    """Test AgentState TypedDict functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_agent_state_structure(self, sample_agent_state):
        """Test agent state structure and fields."""
        state = sample_agent_state
        
        # Check required fields
        assert "messages" in state
        assert "agent_context" in state
        assert "request_id" in state
        assert "user_id" in state
        assert "current_agent" in state
        assert "next_agent" in state
        assert "conversation_flow" in state
        assert "processing_metadata" in state
        
        # Check field types
        assert isinstance(state["messages"], list)
        assert isinstance(state["agent_context"], dict)
        assert isinstance(state["request_id"], str)
        assert isinstance(state["user_id"], str)
        assert isinstance(state["current_agent"], str)
        assert isinstance(state["conversation_flow"], list)
        assert isinstance(state["processing_metadata"], dict)
        
        # Check specific values
        assert state["request_id"] == "test_request_123"
        assert state["user_id"] == "test_user_123"
        assert state["current_agent"] == "Support"
        assert state["next_agent"] == "Technical"


class TestMessagesState:
    """Test MessagesState TypedDict functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_messages_state_structure(self):
        """Test messages state structure."""
        from langchain_core.messages import HumanMessage, AIMessage
        
        state: MessagesState = {
            "messages": [
                HumanMessage(content="User message"),
                AIMessage(content="Agent response")
            ]
        }
        
        assert "messages" in state
        assert isinstance(state["messages"], list)
        assert len(state["messages"]) == 2
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)


class TestAgentHandoff:
    """Test AgentHandoff class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_agent_handoff_creation_with_defaults(self):
        """Test agent handoff creation with default values."""
        handoff = AgentHandoff(
            from_agent="Support",
            to_agent="Technical",
            reason="Need technical expertise"
        )
        
        assert handoff.from_agent == "Support"
        assert handoff.to_agent == "Technical"
        assert handoff.reason == "Need technical expertise"
        assert handoff.context_data == {}  # default
        assert handoff.handoff_type == "query"  # default
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_agent_handoff_creation_with_all_fields(self):
        """Test agent handoff creation with all fields specified."""
        context = {"issue_id": "12345", "priority": "high"}
        
        handoff = AgentHandoff(
            from_agent="Technical",
            to_agent="Product",
            reason="Product configuration needed",
            context_data=context,
            handoff_type="escalation"
        )
        
        assert handoff.from_agent == "Technical"
        assert handoff.to_agent == "Product"
        assert handoff.reason == "Product configuration needed"
        assert handoff.context_data == context
        assert handoff.handoff_type == "escalation"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_handoff_type_validation(self):
        """Test handoff type validation."""
        valid_types = ["query", "clarification", "escalation", "completion"]
        
        for handoff_type in valid_types:
            handoff = AgentHandoff(
                from_agent="Support",
                to_agent="Technical",
                reason="Test",
                handoff_type=handoff_type
            )
            assert handoff.handoff_type == handoff_type


class TestConversationFlow:
    """Test ConversationFlow class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_conversation_flow_creation_with_defaults(self):
        """Test conversation flow creation with default values."""
        flow = ConversationFlow(
            step_number=1,
            agent_name="Support",
            action="received",
            message_content="Test message"
        )
        
        assert flow.step_number == 1
        assert flow.agent_name == "Support"
        assert flow.action == "received"
        assert flow.message_content == "Test message"
        assert flow.processing_time is None  # default
        assert flow.confidence is None  # default
        assert isinstance(flow.timestamp, datetime)
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_conversation_flow_creation_with_all_fields(self):
        """Test conversation flow creation with all fields specified."""
        test_time = datetime.now()
        
        flow = ConversationFlow(
            step_number=2,
            agent_name="Technical",
            action="processed",
            timestamp=test_time,
            message_content="Technical analysis",
            processing_time=1.5,
            confidence=0.9
        )
        
        assert flow.step_number == 2
        assert flow.agent_name == "Technical"
        assert flow.action == "processed"
        assert flow.timestamp == test_time
        assert flow.message_content == "Technical analysis"
        assert flow.processing_time == 1.5
        assert flow.confidence == 0.9
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_action_validation(self):
        """Test action validation."""
        valid_actions = ["received", "processed", "responded", "handed_off", "completed"]
        
        for action in valid_actions:
            flow = ConversationFlow(
                step_number=1,
                agent_name="Support",
                action=action,
                message_content="Test"
            )
            assert flow.action == action


class TestUtilityFunctions:
    """Test utility functions for message conversion."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_messages_to_langchain(self):
        """Test conversion of Message list to LangChain messages."""
        messages = [
            Message(
                content="User message",
                source_agent="user",
                target_agent="Support"
            ),
            Message(
                content="Agent response",
                source_agent="Support",
                target_agent="user"
            )
        ]
        
        lc_messages = messages_to_langchain(messages)
        
        assert len(lc_messages) == 2
        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)
        assert lc_messages[0].content == "User message"
        assert lc_messages[1].content == "Agent response"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_messages_to_langchain_empty_list(self):
        """Test conversion of empty message list."""
        messages = []
        lc_messages = messages_to_langchain(messages)
        assert lc_messages == []
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_langchain_to_messages(self):
        """Test conversion of LangChain messages to Message list."""
        lc_messages = [
            HumanMessage(
                content="User message",
                additional_kwargs={
                    "source_agent": "user",
                    "target_agent": "Support",
                    "context": {"test": "context"}
                }
            ),
            AIMessage(
                content="Agent response",
                additional_kwargs={
                    "source_agent": "Support",
                    "target_agent": "user",
                    "context": {}
                }
            )
        ]
        
        messages = langchain_to_messages(lc_messages)
        
        assert len(messages) == 2
        assert isinstance(messages[0], Message)
        assert isinstance(messages[1], Message)
        assert messages[0].content == "User message"
        assert messages[0].source_agent == "user"
        assert messages[0].target_agent == "Support"
        assert messages[0].context == {"test": "context"}
        assert messages[1].content == "Agent response"
        assert messages[1].source_agent == "Support"
        assert messages[1].target_agent == "user"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_langchain_to_messages_missing_kwargs(self):
        """Test conversion with missing additional_kwargs."""
        lc_messages = [
            HumanMessage(content="User message"),
            AIMessage(content="Agent response")
        ]
        
        messages = langchain_to_messages(lc_messages)
        
        assert len(messages) == 2
        assert messages[0].source_agent == "user"
        assert messages[0].target_agent == "unknown"
        assert messages[0].context == {}
        assert messages[1].source_agent == "ai"
        assert messages[1].target_agent == "unknown"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_langchain_to_messages_other_message_type(self):
        """Test conversion with other message types."""
        from langchain_core.messages import SystemMessage
        
        lc_messages = [
            SystemMessage(content="System message")
        ]
        
        messages = langchain_to_messages(lc_messages)
        
        assert len(messages) == 1
        assert messages[0].source_agent == "system"
        assert messages[0].target_agent == "unknown"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_langchain_to_messages_empty_list(self):
        """Test conversion of empty LangChain message list."""
        lc_messages = []
        messages = langchain_to_messages(lc_messages)
        assert messages == []
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_create_agent_state(self):
        """Test creation of agent state from user request."""
        request = UserRequest(
            query="Test query",
            user_id="user_123",
            priority="high"
        )
        
        additional_context = {"session": "abc123"}
        
        state = create_agent_state(
            request=request,
            current_agent="Technical",
            additional_context=additional_context
        )
        
        assert state["request_id"] == request.request_id
        assert state["user_id"] == "user_123"
        assert state["current_agent"] == "Technical"
        assert state["next_agent"] is None
        assert state["agent_context"] == additional_context
        assert state["conversation_flow"] == []
        assert "start_time" in state["processing_metadata"]
        assert state["processing_metadata"]["max_processing_time"] == request.max_processing_time
        assert state["processing_metadata"]["priority"] == "high"
        assert len(state["messages"]) == 1
        
        # Check the initial message
        initial_message = state["messages"][0]
        assert isinstance(initial_message, HumanMessage)
        assert initial_message.content == "Test query"
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_create_agent_state_defaults(self):
        """Test creation of agent state with default values."""
        request = UserRequest(
            query="Test query",
            user_id="user_123"
        )
        
        state = create_agent_state(request=request)
        
        assert state["current_agent"] == "Support"  # default
        assert state["agent_context"] == {}  # default when None
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_create_agent_state_none_context(self):
        """Test creation of agent state with None additional context."""
        request = UserRequest(
            query="Test query",
            user_id="user_123"
        )
        
        state = create_agent_state(
            request=request,
            additional_context=None
        )
        
        assert state["agent_context"] == {}


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_message_with_very_long_content(self):
        """Test message with very long content."""
        long_content = "x" * 10000
        message = Message(
            content=long_content,
            source_agent="user",
            target_agent="Support"
        )
        
        assert message.content == long_content
        assert len(message.content) == 10000
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_message_with_empty_content(self):
        """Test message with empty content."""
        message = Message(
            content="",
            source_agent="user",
            target_agent="Support"
        )
        
        assert message.content == ""
        lc_message = message.to_langchain_message()
        assert lc_message.content == ""
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_response_with_edge_confidence_values(self):
        """Test response with edge confidence values."""
        # Test minimum confidence
        response_min = Response(
            content="Test",
            source_agent="Support",
            target_agent="user",
            confidence=0.0
        )
        assert response_min.confidence == 0.0
        
        # Test maximum confidence
        response_max = Response(
            content="Test",
            source_agent="Support",
            target_agent="user",
            confidence=1.0
        )
        assert response_max.confidence == 1.0
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_user_request_with_zero_processing_time(self):
        """Test user request with zero max processing time."""
        request = UserRequest(
            query="Test",
            user_id="user_123",
            max_processing_time=0
        )
        
        assert request.max_processing_time == 0
    
    @pytest.mark.unit
    @pytest.mark.models
    def test_chain_result_with_zero_processing_time(self):
        """Test chain result with zero processing time."""
        result = ChainResult(
            request_id="req_123",
            response="Test",
            total_processing_time=0.0
        )
        
        assert result.total_processing_time == 0.0
    
    @pytest.mark.unit
    @pytest.mark.models  
    def test_complex_nested_context(self):
        """Test message with complex nested context data."""
        complex_context = {
            "user": {
                "id": "user_123",
                "preferences": {
                    "language": "en",
                    "timezone": "UTC"
                }
            },
            "session": {
                "id": "session_456",
                "start_time": "2024-01-01T00:00:00Z"
            },
            "metadata": [
                {"key": "value1"},
                {"key": "value2"}
            ]
        }
        
        message = Message(
            content="Test",
            source_agent="user",
            target_agent="Support",
            context=complex_context
        )
        
        assert message.context == complex_context
        lc_message = message.to_langchain_message()
        assert lc_message.additional_kwargs["context"] == complex_context 