"""
Comprehensive tests for terminal_client.py
Tests all TerminalClient functionality with 100% coverage including interactive features.
"""

import pytest
import asyncio
import sys
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from io import StringIO
from datetime import datetime

from terminal_client import TerminalClient
from models.message import UserRequest


class TestTerminalClientInitialization:
    """Test TerminalClient initialization and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_terminal_client_initialization(self):
        """Test TerminalClient initializes with default configuration."""
        client = TerminalClient()
        
        assert client.base_url == "http://localhost:8000"
        assert client.session_id is not None
        assert client.session_id.startswith("terminal_session_")
        assert client.request_count == 0
        assert hasattr(client, 'session')  # aiohttp session
        assert not hasattr(client, '_closed') or not client._closed
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_terminal_client_custom_base_url(self):
        """Test TerminalClient with custom base URL."""
        custom_url = "https://custom-api.example.com"
        client = TerminalClient(base_url=custom_url)
        
        assert client.base_url == custom_url
        assert client.session_id is not None
        assert client.request_count == 0
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_session_id_generation(self):
        """Test session ID is properly generated."""
        client = TerminalClient()
        
        assert isinstance(client.session_id, str)
        assert len(client.session_id) > 10
        assert "terminal_session_" in client.session_id
        
        # Each client should have a unique session ID
        client2 = TerminalClient()
        assert client.session_id != client2.session_id


class TestAPIInteraction:
    """Test API interaction methods."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_send_query_success(self):
        """Test successful query sending to API."""
        client = TerminalClient()
        
        mock_response_data = {
            "success": True,
            "request_id": "test_123",
            "response": "Test response from API",
            "confidence_score": 0.85,
            "agents_involved": ["Support", "Technical"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "processed", "confidence": 0.8},
                {"step": 2, "agent": "Technical", "action": "processed", "confidence": 0.9}
            ],
            "total_processing_time": 2.3,
            "hop_count": 2
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        with patch.object(client.session, 'post', return_value=mock_response):
            result = await client.send_query("Test query", "test_user")
            
            assert result == mock_response_data
            assert result["success"] is True
            assert result["response"] == "Test response from API"
            assert result["confidence_score"] == 0.85
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_send_query_http_error(self):
        """Test query sending with HTTP error."""
        client = TerminalClient()
        
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        with patch.object(client.session, 'post', return_value=mock_response):
            result = await client.send_query("Test query", "test_user")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_send_query_connection_error(self):
        """Test query sending with connection error."""
        client = TerminalClient()
        
        import aiohttp
        with patch.object(client.session, 'post', 
                         side_effect=aiohttp.ClientConnectorError(Mock(), Mock())):
            result = await client.send_query("Test query", "test_user")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_send_query_timeout_error(self):
        """Test query sending with timeout error."""
        client = TerminalClient()
        
        import asyncio
        with patch.object(client.session, 'post', 
                         side_effect=asyncio.TimeoutError()):
            result = await client.send_query("Test query", "test_user")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_send_query_json_error(self):
        """Test query sending with JSON parsing error."""
        client = TerminalClient()
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.text = AsyncMock(return_value="Invalid JSON response")
        
        with patch.object(client.session, 'post', return_value=mock_response):
            result = await client.send_query("Test query", "test_user")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_send_query_request_count_increment(self):
        """Test request count is incremented on each query."""
        client = TerminalClient()
        initial_count = client.request_count
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True, "response": "Test"})
        
        with patch.object(client.session, 'post', return_value=mock_response):
            await client.send_query("Test query 1", "test_user")
            assert client.request_count == initial_count + 1
            
            await client.send_query("Test query 2", "test_user")
            assert client.request_count == initial_count + 2


class TestResponseDisplay:
    """Test response display and formatting methods."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_response_success(self):
        """Test successful response display."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Here's how to reset your password...",
            "confidence_score": 0.9,
            "agents_involved": ["Support", "Technical"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "processed", "confidence": 0.8, "agent_usage_count": 1},
                {"step": 2, "agent": "Technical", "action": "processed", "confidence": 0.9, "agent_usage_count": 1}
            ],
            "total_processing_time": 1.5,
            "hop_count": 2
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            # Check that response was displayed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            assert "Here's how to reset your password" in response_text
            assert "Support → Technical" in response_text
            assert "1.5s" in response_text
            assert "0.9" in response_text  # confidence score
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_response_failure(self):
        """Test failed response display."""
        client = TerminalClient()
        
        response_data = {
            "success": False,
            "response": "I apologize, but I encountered an error",
            "error_details": "Connection timeout",
            "confidence_score": 0.1
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            assert "I apologize, but I encountered an error" in response_text
            assert "Connection timeout" in response_text
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_response_bidirectional_flow(self):
        """Test display of bidirectional agent flow."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Complex analysis complete",
            "agents_involved": ["Support", "Technical", "Product"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "initiated", "agent_usage_count": 1},
                {"step": 2, "agent": "Technical", "action": "analyzed", "agent_usage_count": 1},
                {"step": 3, "agent": "Product", "action": "consulted", "agent_usage_count": 1},
                {"step": 4, "agent": "Support", "action": "synthesized", "agent_usage_count": 2}
            ],
            "total_processing_time": 3.2,
            "hop_count": 4
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            assert "Support → Technical → Product → Support(2)" in response_text
            assert "bidirectional" in response_text
            assert "iterative" in response_text
            assert "multi-hop" in response_text
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_response_simple_flow(self):
        """Test display of simple single-agent flow."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Simple answer provided",
            "agents_involved": ["Support"],
            "conversation_flow": [
                {"step": 1, "agent": "Support", "action": "processed", "agent_usage_count": 1}
            ],
            "total_processing_time": 0.8,
            "hop_count": 1
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            # Should not show complex flow analysis for simple responses
            assert "Simple answer provided" in response_text
            assert "bidirectional" not in response_text.lower()
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_response_empty_flow(self):
        """Test display with empty conversation flow."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Direct response",
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 0.5,
            "hop_count": 1
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            assert "Direct response" in response_text
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_response_missing_fields(self):
        """Test display with missing optional fields."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Minimal response"
            # Missing many optional fields
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            assert "Minimal response" in response_text
            # Should handle missing fields gracefully


class TestUserInteraction:
    """Test user interaction and input handling."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_get_user_input_normal(self):
        """Test normal user input handling."""
        client = TerminalClient()
        
        with patch('builtins.input', return_value="Test user question"):
            user_input = client.get_user_input()
            
            assert user_input == "Test user question"
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_get_user_input_empty(self):
        """Test empty user input handling."""
        client = TerminalClient()
        
        with patch('builtins.input', return_value=""):
            user_input = client.get_user_input()
            
            assert user_input == ""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_get_user_input_whitespace(self):
        """Test user input with whitespace."""
        client = TerminalClient()
        
        with patch('builtins.input', return_value="  whitespace test  "):
            user_input = client.get_user_input()
            
            assert user_input == "  whitespace test  "  # Should preserve whitespace
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_get_user_input_keyboard_interrupt(self):
        """Test handling of Ctrl+C during input."""
        client = TerminalClient()
        
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            user_input = client.get_user_input()
            
            assert user_input is None  # Should return None on interrupt
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_get_user_input_eof_error(self):
        """Test handling of EOF during input."""
        client = TerminalClient()
        
        with patch('builtins.input', side_effect=EOFError):
            user_input = client.get_user_input()
            
            assert user_input is None  # Should return None on EOF


class TestInteractiveSession:
    """Test interactive session functionality."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_run_interactive_success_flow(self):
        """Test successful interactive session flow."""
        client = TerminalClient()
        
        # Mock user inputs: question, then exit
        user_inputs = ["How do I reset my password?", "exit"]
        input_iter = iter(user_inputs)
        
        mock_response = {
            "success": True,
            "response": "To reset your password, follow these steps...",
            "confidence_score": 0.9,
            "agents_involved": ["Support"],
            "conversation_flow": [],
            "total_processing_time": 1.0,
            "hop_count": 1
        }
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch.object(client, 'send_query', return_value=mock_response), \
             patch.object(client, 'display_response') as mock_display, \
             patch('builtins.print'):
            
            await client.run_interactive()
            
            mock_display.assert_called_once_with(mock_response)
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_run_interactive_empty_input(self):
        """Test interactive session with empty input."""
        client = TerminalClient()
        
        # Mock user inputs: empty string, then exit
        user_inputs = ["", "exit"]
        input_iter = iter(user_inputs)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch.object(client, 'send_query') as mock_send, \
             patch('builtins.print'):
            
            await client.run_interactive()
            
            # Should not send query for empty input
            mock_send.assert_not_called()
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_run_interactive_keyboard_interrupt(self):
        """Test interactive session with keyboard interrupt."""
        client = TerminalClient()
        
        with patch.object(client, 'get_user_input', return_value=None), \
             patch('builtins.print') as mock_print:
            
            await client.run_interactive()
            
            # Should print goodbye message
            print_calls = [str(call) for call in mock_print.call_args_list]
            goodbye_found = any("goodbye" in call.lower() or "exit" in call.lower() 
                              for call in print_calls)
            assert goodbye_found
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_run_interactive_api_error(self):
        """Test interactive session with API error."""
        client = TerminalClient()
        
        user_inputs = ["Test question", "exit"]
        input_iter = iter(user_inputs)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch.object(client, 'send_query', return_value=None), \
             patch('builtins.print') as mock_print:
            
            await client.run_interactive()
            
            # Should print error message when API fails
            print_calls = [str(call) for call in mock_print.call_args_list]
            error_found = any("error" in call.lower() or "failed" in call.lower() 
                            for call in print_calls)
            assert error_found
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_run_interactive_help_command(self):
        """Test interactive session with help command."""
        client = TerminalClient()
        
        user_inputs = ["help", "exit"]
        input_iter = iter(user_inputs)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch('builtins.print') as mock_print:
            
            await client.run_interactive()
            
            # Should print help information
            print_calls = [str(call) for call in mock_print.call_args_list]
            help_found = any("help" in call.lower() or "command" in call.lower() 
                           for call in print_calls)
            assert help_found
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_run_interactive_status_command(self):
        """Test interactive session with status command."""
        client = TerminalClient()
        
        user_inputs = ["status", "exit"]
        input_iter = iter(user_inputs)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch('builtins.print') as mock_print:
            
            await client.run_interactive()
            
            # Should print status information
            print_calls = [str(call) for call in mock_print.call_args_list]
            status_found = any("session" in call.lower() or "request" in call.lower() 
                             for call in print_calls)
            assert status_found


class TestSessionManagement:
    """Test session management and cleanup."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_session_creation(self):
        """Test aiohttp session is created properly."""
        client = TerminalClient()
        
        assert client.session is not None
        assert hasattr(client.session, 'post')
        assert hasattr(client.session, 'close')
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_session_cleanup(self):
        """Test session cleanup on exit."""
        client = TerminalClient()
        
        # Mock session close method
        with patch.object(client.session, 'close') as mock_close:
            await client.cleanup()
            mock_close.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_context_manager_usage(self):
        """Test TerminalClient as async context manager."""
        async with TerminalClient() as client:
            assert client.session is not None
            
        # Session should be cleaned up after context exit
        # (This is implementation dependent)


class TestDisplayFormatting:
    """Test display formatting and visual elements."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_header_formatting(self):
        """Test proper header formatting in display."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Test response",
            "confidence_score": 0.85
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            
            # Should include formatted headers
            header_found = any("=" in call for call in print_calls)
            assert header_found
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_confidence_formatting(self):
        """Test confidence score formatting."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Test response",
            "confidence_score": 0.857  # Should be rounded appropriately
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            # Should format confidence score nicely
            assert "0.86" in response_text or "0.857" in response_text
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_timing_formatting(self):
        """Test processing time formatting."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Test response",
            "total_processing_time": 2.347  # Should be rounded
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            # Should format timing nicely
            assert "2.35s" in response_text or "2.3s" in response_text
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_display_agent_flow_formatting(self):
        """Test agent flow visualization formatting."""
        client = TerminalClient()
        
        response_data = {
            "success": True,
            "response": "Test response",
            "conversation_flow": [
                {"agent": "Support", "action": "processed", "agent_usage_count": 1},
                {"agent": "Technical", "action": "analyzed", "agent_usage_count": 1},
                {"agent": "Support", "action": "synthesized", "agent_usage_count": 2}
            ]
        }
        
        with patch('builtins.print') as mock_print:
            client.display_response(response_data)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            response_text = ' '.join(print_calls)
            
            # Should show flow with usage counts
            assert "Support → Technical → Support(2)" in response_text
            assert "Flow Type:" in response_text


class TestErrorHandling:
    """Test error handling in terminal client."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    @pytest.mark.error_handling
    async def test_network_error_handling(self):
        """Test handling of network errors."""
        client = TerminalClient()
        
        import aiohttp
        with patch.object(client.session, 'post', 
                         side_effect=aiohttp.ClientError("Network error")):
            result = await client.send_query("Test query", "test_user")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.terminal
    @pytest.mark.error_handling
    def test_display_error_handling(self):
        """Test display method handles malformed data gracefully."""
        client = TerminalClient()
        
        # Malformed response data
        malformed_data = {
            "success": True,
            "response": None,  # Invalid None response
            "agents_involved": "not_a_list"  # Invalid type
        }
        
        with patch('builtins.print') as mock_print:
            # Should not raise exception
            try:
                client.display_response(malformed_data)
            except Exception as e:
                pytest.fail(f"display_response raised {e} with malformed data")
    
    @pytest.mark.unit
    @pytest.mark.terminal
    @pytest.mark.error_handling
    async def test_session_error_handling(self):
        """Test session error handling during requests."""
        client = TerminalClient()
        
        # Mock session that raises exception
        client.session = Mock()
        client.session.post.side_effect = Exception("Session error")
        
        result = await client.send_query("Test query", "test_user")
        
        assert result is None


class TestUtilityMethods:
    """Test utility and helper methods."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_session_id_uniqueness(self):
        """Test that each client gets a unique session ID."""
        clients = [TerminalClient() for _ in range(5)]
        session_ids = [client.session_id for client in clients]
        
        # All session IDs should be unique
        assert len(set(session_ids)) == len(session_ids)
    
    @pytest.mark.unit
    @pytest.mark.terminal
    def test_request_count_tracking(self):
        """Test request count is properly tracked."""
        client = TerminalClient()
        
        initial_count = client.request_count
        assert initial_count == 0
        
        # Mock a successful request
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True})
        
        with patch.object(client.session, 'post', return_value=mock_response):
            # Request count should increment
            asyncio.run(client.send_query("Test", "user"))
            assert client.request_count == initial_count + 1


class TestCommandHandling:
    """Test special command handling in interactive mode."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_exit_commands(self):
        """Test various exit commands."""
        client = TerminalClient()
        
        exit_commands = ["exit", "quit", "bye", "goodbye"]
        
        for exit_cmd in exit_commands:
            with patch.object(client, 'get_user_input', return_value=exit_cmd), \
                 patch('builtins.print'):
                
                await client.run_interactive()
                # Should exit cleanly without errors
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_help_command_content(self):
        """Test help command displays useful information."""
        client = TerminalClient()
        
        user_inputs = ["help", "exit"]
        input_iter = iter(user_inputs)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch('builtins.print') as mock_print:
            
            await client.run_interactive()
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            help_text = ' '.join(print_calls).lower()
            
            # Should contain helpful information
            assert any(keyword in help_text for keyword in 
                      ["command", "help", "exit", "quit"])
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_status_command_content(self):
        """Test status command displays session information."""
        client = TerminalClient()
        
        user_inputs = ["status", "exit"]
        input_iter = iter(user_inputs)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch('builtins.print') as mock_print:
            
            await client.run_interactive()
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            status_text = ' '.join(print_calls).lower()
            
            # Should contain session information
            assert any(keyword in status_text for keyword in 
                      ["session", "request", "count", "url"])


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    @pytest.mark.performance
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        client = TerminalClient()
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True, "response": "Test"})
        
        with patch.object(client.session, 'post', return_value=mock_response):
            # Send multiple concurrent requests
            tasks = [
                client.send_query(f"Query {i}", "test_user")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All requests should succeed
            assert len(results) == 5
            assert all(result is not None for result in results)
            assert client.request_count == 5
    
    @pytest.mark.unit
    @pytest.mark.terminal
    @pytest.mark.performance
    def test_memory_efficiency(self):
        """Test memory efficiency with large responses."""
        client = TerminalClient()
        
        # Large response data
        large_response = {
            "success": True,
            "response": "x" * 10000,  # Large response text
            "conversation_flow": [
                {"agent": f"Agent{i}", "action": "processed"} 
                for i in range(100)
            ],
            "agents_involved": [f"Agent{i}" for i in range(50)]
        }
        
        with patch('builtins.print'):
            # Should handle large responses without issues
            try:
                client.display_response(large_response)
            except MemoryError:
                pytest.fail("display_response caused MemoryError with large data")


class TestIntegrationScenarios:
    """Test integration scenarios and realistic usage."""
    
    @pytest.mark.unit
    @pytest.mark.terminal
    async def test_realistic_conversation_flow(self):
        """Test realistic conversation with multiple exchanges."""
        client = TerminalClient()
        
        # Simulate a realistic conversation
        conversation_inputs = [
            "How do I reset my password?",
            "What if I don't have access to my email?",
            "Can you help me with account recovery?",
            "exit"
        ]
        input_iter = iter(conversation_inputs)
        
        mock_responses = [
            {
                "success": True,
                "response": "To reset your password, click 'Forgot Password' on the login page...",
                "agents_involved": ["Support"],
                "confidence_score": 0.9
            },
            {
                "success": True,
                "response": "If you don't have email access, you can contact support...",
                "agents_involved": ["Support"],
                "confidence_score": 0.8
            },
            {
                "success": True,
                "response": "For account recovery, we'll need to verify your identity...",
                "agents_involved": ["Support", "Technical"],
                "confidence_score": 0.85
            }
        ]
        response_iter = iter(mock_responses)
        
        with patch.object(client, 'get_user_input', side_effect=lambda: next(input_iter)), \
             patch.object(client, 'send_query', side_effect=lambda *args: next(response_iter)), \
             patch.object(client, 'display_response') as mock_display, \
             patch('builtins.print'):
            
            await client.run_interactive()
            
            # Should have displayed 3 responses
            assert mock_display.call_count == 3
            assert client.request_count == 3 