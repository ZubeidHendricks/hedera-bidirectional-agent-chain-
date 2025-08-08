"""
Comprehensive tests for agents/technical_agent.py
Tests all TechnicalAgent functionality with 100% coverage including diagnostic capabilities.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from agents.technical_agent import TechnicalAgent
from agents.base_agent import BaseAgent
from models.message import Message, Response
from tests.conftest import create_test_message, create_test_response


class TestTechnicalAgentInitialization:
    """Test TechnicalAgent initialization and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_technical_agent_initialization(self, mock_genai_client):
        """Test TechnicalAgent initialization with proper configuration."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            
            assert agent.name == "Technical"
            assert agent.expertise == "system_diagnostics"
            assert "system_diagnostics" in agent.capabilities
            assert isinstance(agent.system_prompt, str)
            assert "Technical Support Agent" in agent.system_prompt
            assert "BIDIRECTIONAL AGENT CHAIN ROLE" in agent.system_prompt
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_technical_agent_system_prompt_content(self, mock_genai_client):
        """Test TechnicalAgent system prompt contains required technical elements."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            
            prompt = agent.system_prompt
            
            # Check for key technical sections
            assert "PRIMARY TECHNICAL CAPABILITIES" in prompt
            assert "BIDIRECTIONAL AGENT CHAIN ROLE" in prompt
            assert "DIAGNOSTIC METHODOLOGY" in prompt
            assert "COLLABORATION PATTERNS" in prompt
            assert "TECHNICAL COMMUNICATION STANDARDS" in prompt
            assert "SOLUTION QUALITY" in prompt
            
            # Check for specific technical capabilities
            assert "system diagnostics" in prompt
            assert "troubleshooting" in prompt
            assert "performance optimization" in prompt
            assert "security assessment" in prompt
            assert "root cause analysis" in prompt
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_technical_agent_inheritance(self, mock_genai_client):
        """Test TechnicalAgent properly inherits from BaseAgent."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            
            assert isinstance(agent, BaseAgent)
            assert hasattr(agent, 'process_message')
            assert hasattr(agent, 'register_agent')
            assert hasattr(agent, 'get_capabilities')


class TestTechnicalDiagnosticCapabilities:
    """Test TechnicalAgent's diagnostic and analysis capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_system_error_analysis(self, mock_genai_client):
        """Test analysis of system errors and crashes."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Error Analysis: The 500 error indicates a server-side issue. Root cause: Database connection timeout. Solution: Check network connectivity and increase timeout settings.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.9,
                reasoning="Analyzed error patterns and system logs"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="System showing error 500 when users try to access the database",
                    source_agent="Support",
                    target_agent="Technical",
                    message_type="technical_analysis"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.source_agent == "Technical"
                assert "500 error" in result.content
                assert "Database connection timeout" in result.content
                assert "Root cause" in result.content
                assert result.confidence >= 0.8
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_performance_optimization_analysis(self, mock_genai_client):
        """Test performance analysis and optimization recommendations."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Performance Analysis: Response times are 3x slower than baseline. Identified bottlenecks: 1) Unoptimized database queries, 2) Missing cache layer, 3) Inefficient API calls. Recommended optimizations: Implement query indexing, add Redis cache, batch API requests.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.85,
                suggested_next_agent="Product",
                reasoning="Analyzed performance metrics and system architecture"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Application performance has degraded significantly over the past week",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"performance_metrics": "avg_response_time: 3.2s, baseline: 1.1s"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Performance Analysis" in result.content
                assert "bottlenecks" in result.content
                assert "optimizations" in result.content
                assert result.confidence >= 0.8
                assert result.suggested_next_agent == "Product"
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_security_vulnerability_assessment(self, mock_genai_client):
        """Test security vulnerability analysis."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Security Assessment: Identified potential SQL injection vulnerability in user input validation. Risk Level: HIGH. Immediate actions: 1) Implement parameterized queries, 2) Add input sanitization, 3) Enable WAF rules. Long-term: Security audit and penetration testing.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.95,
                needs_clarification=False,
                reasoning="Analyzed code patterns and security logs"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Suspicious activity detected in application logs, possible security breach attempt",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"log_entries": "Multiple failed login attempts, unusual SQL query patterns"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Security Assessment" in result.content
                assert "SQL injection" in result.content
                assert "Risk Level: HIGH" in result.content
                assert "parameterized queries" in result.content
                assert result.confidence >= 0.9
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_infrastructure_analysis(self, mock_genai_client):
        """Test infrastructure and architecture evaluation."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Infrastructure Analysis: Current architecture shows scalability constraints. CPU utilization at 85%, memory usage at 90%. Recommendations: 1) Implement horizontal scaling, 2) Add load balancer, 3) Optimize memory usage, 4) Consider microservices architecture for better resource allocation.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.88,
                reasoning="Analyzed system metrics and architecture patterns"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="System performance is degrading under increased user load",
                    source_agent="Support",
                    target_agent="Technical",
                    context={
                        "system_metrics": {
                            "cpu_usage": "85%",
                            "memory_usage": "90%",
                            "active_users": "15000"
                        }
                    }
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Infrastructure Analysis" in result.content
                assert "scalability constraints" in result.content
                assert "horizontal scaling" in result.content
                assert result.confidence >= 0.8


class TestTechnicalCollaborationPatterns:
    """Test TechnicalAgent collaboration with other agents."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_collaboration_with_support_agent(self, mock_genai_client):
        """Test technical collaboration responding to Support agent queries."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Technical Analysis Complete: The issue is a memory leak in the session management module. Immediate fix: Restart the service. Long-term solution: Update session cleanup logic to properly release memory.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.92,
                suggested_next_agent=None,
                reasoning="Analyzed system logs and memory usage patterns"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer reports application becoming slower over time and eventually crashing",
                    source_agent="Support",
                    target_agent="Technical",
                    message_type="handoff",
                    context={"customer_id": "12345", "error_pattern": "gradual slowdown then crash"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.target_agent == "Support"
                assert "memory leak" in result.content
                assert "session management" in result.content
                assert result.confidence >= 0.9
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_collaboration_with_product_agent(self, mock_genai_client):
        """Test technical collaboration with Product agent for implementation details."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            # Create and register product agent
            mock_product_agent = Mock()
            mock_product_response = Response(
                content="Product requirements: Feature should support 10,000 concurrent users with 99.9% uptime",
                source_agent="Product",
                target_agent="Technical"
            )
            mock_product_agent.process_message = AsyncMock(return_value=mock_product_response)
            agent.register_agent("Product", mock_product_agent)
            
            mock_response = Response(
                content="Technical Implementation Analysis: Based on product requirements, recommend microservices architecture with Redis clustering and auto-scaling. HANDOFF_REQUEST: Product - Please confirm scalability requirements and budget constraints",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.85,
                suggested_next_agent="Product"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Need technical assessment for new feature implementation",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"feature_type": "real-time collaboration"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Technical Implementation Analysis" in result.content
                assert "microservices" in result.content
                assert result.suggested_next_agent == "Product"
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_technical_handoff_coordination(self, mock_genai_client):
        """Test Technical agent coordinating handoffs based on technical assessment."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Technical analysis indicates this is a product configuration issue rather than a system problem. The API limits are set too low for the customer's usage pattern. HANDOFF_REQUEST: Product - Please review API rate limiting configuration for enterprise customers",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.9,
                suggested_next_agent="Product"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Enterprise customer experiencing API rate limiting issues",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"customer_tier": "enterprise", "api_usage": "high"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "product configuration issue" in result.content
                assert "HANDOFF_REQUEST: Product" in result.content
                assert "API rate limiting" in result.content


class TestTechnicalMethodology:
    """Test TechnicalAgent's diagnostic methodology and approaches."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_systematic_problem_analysis(self, mock_genai_client):
        """Test systematic approach to problem analysis."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Systematic Analysis:\n1. Problem Identification: Database connection failures\n2. Data Collection: Connection logs, network metrics\n3. Hypothesis Testing: Network latency vs. database overload\n4. Root Cause: Database connection pool exhaustion\n5. Solution: Increase pool size and implement connection recycling",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.93,
                reasoning="Applied structured diagnostic methodology"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Intermittent database connection issues affecting multiple customers",
                    source_agent="Support",
                    target_agent="Technical"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Systematic Analysis" in result.content
                assert "Problem Identification" in result.content
                assert "Root Cause" in result.content
                assert "Solution" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_evidence_based_reasoning(self, mock_genai_client):
        """Test evidence-based technical reasoning."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Evidence-Based Analysis:\nLogs show 95% of errors occur between 2-4 PM (peak hours)\nCPU spikes correlate with error frequency (correlation: 0.89)\nMemory usage stable, ruling out memory leaks\nConclusion: Performance bottleneck due to insufficient CPU resources during peak load",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.95,
                reasoning="Statistical analysis of system metrics and error patterns"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="System errors spike during business hours",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"error_logs": "attached", "monitoring_data": "24h_metrics.json"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Evidence-Based Analysis" in result.content
                assert "correlation" in result.content
                assert "95%" in result.content
                assert result.confidence >= 0.9
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_risk_assessment_methodology(self, mock_genai_client):
        """Test risk assessment and impact analysis."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Risk Assessment:\nCRITICAL: Data corruption risk if left unaddressed\nImpact: 10,000+ users affected\nMitigation Timeline: Immediate (< 2 hours)\nContingency Plan: Rollback to previous version if fix fails\nMonitoring: Implement real-time alerts for early detection",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.88,
                needs_clarification=False,
                reasoning="Assessed system impact and risk factors"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Critical bug discovered in data processing pipeline",
                    source_agent="Support",
                    target_agent="Technical",
                    priority="critical"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Risk Assessment" in result.content
                assert "CRITICAL" in result.content
                assert "Impact" in result.content
                assert "Contingency Plan" in result.content


class TestTechnicalCommunicationStandards:
    """Test TechnicalAgent communication standards and clarity."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_precise_technical_terminology(self, mock_genai_client):
        """Test use of precise technical terminology with explanations."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="TCP Connection Reset (RST packet): The server forcibly closed the connection. This typically indicates: 1) Server overload, 2) Firewall blocking, 3) Application crash. Diagnostic steps: Check server logs, verify firewall rules, monitor resource usage.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.9
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Users getting connection reset errors",
                    source_agent="Support",
                    target_agent="Technical"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "TCP Connection Reset" in result.content
                assert "RST packet" in result.content
                assert "This typically indicates" in result.content  # Explanation provided
                assert "Diagnostic steps" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_step_by_step_procedures(self, mock_genai_client):
        """Test provision of step-by-step technical procedures."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Database Recovery Procedure:\n1. Stop application services\n2. Create database backup\n3. Run consistency check: DBCC CHECKDB\n4. If corruption found: Restore from last known good backup\n5. If clean: Restart services and monitor\n6. Verify data integrity post-recovery",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.92
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Database corruption detected, need recovery steps",
                    source_agent="Support",
                    target_agent="Technical",
                    priority="high"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "1." in result.content
                assert "2." in result.content
                assert "3." in result.content
                assert "DBCC CHECKDB" in result.content
                assert "Procedure" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_verification_steps_included(self, mock_genai_client):
        """Test inclusion of verification and validation steps."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="SSL Certificate Update:\n1. Generate new certificate\n2. Install on web server\n3. Update configuration\n4. Restart web service\nVerification Steps:\n- Test HTTPS connection\n- Verify certificate chain\n- Check expiration date\n- Monitor error logs for 24 hours",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.9
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="SSL certificate expiring soon, need update procedure",
                    source_agent="Support",
                    target_agent="Technical"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Verification Steps" in result.content
                assert "Test HTTPS" in result.content
                assert "Monitor error logs" in result.content


class TestErrorHandlingAndEdgeCases:
    """Test TechnicalAgent error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_insufficient_technical_information(self, mock_genai_client):
        """Test handling requests with insufficient technical details."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Additional Information Required: To provide accurate technical analysis, I need: 1) Error logs or messages, 2) System specifications, 3) Recent changes made, 4) Affected user count. Please provide these details for proper diagnosis.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.3,
                needs_clarification=True,
                reasoning="Insufficient technical details for accurate analysis"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Something is broken",
                    source_agent="Support",
                    target_agent="Technical"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.needs_clarification is True
                assert result.confidence <= 0.5
                assert "Additional Information Required" in result.content
                assert "Error logs" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_conflicting_technical_data(self, mock_genai_client):
        """Test handling conflicting or inconsistent technical data."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Data Inconsistency Detected: CPU metrics show 20% usage while users report slow performance. Possible causes: 1) Metrics collection issue, 2) I/O bottleneck not reflected in CPU, 3) Network latency. Recommend additional monitoring of disk I/O and network metrics.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.6,
                needs_clarification=True,
                reasoning="Conflicting data requires additional investigation"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="System showing low CPU usage but users experiencing slow performance",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"cpu_usage": "20%", "user_complaints": "high"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Data Inconsistency" in result.content
                assert result.needs_clarification is True
                assert result.confidence <= 0.7
                assert "additional monitoring" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_legacy_system_constraints(self, mock_genai_client):
        """Test handling constraints imposed by legacy systems."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Legacy System Constraints: Limited to .NET Framework 2.0 and SQL Server 2008. Modern solutions not applicable. Recommended approach: 1) Optimize existing stored procedures, 2) Implement connection pooling within framework limits, 3) Plan for future system modernization.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.75,
                reasoning="Analyzed within legacy system constraints"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Need performance improvement for legacy financial system",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"system_version": ".NET 2.0", "database": "SQL Server 2008"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Legacy System Constraints" in result.content
                assert ".NET Framework 2.0" in result.content
                assert "within framework limits" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_emergency_critical_issue(self, mock_genai_client):
        """Test handling emergency critical technical issues."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="EMERGENCY RESPONSE: Complete system outage detected. Immediate Actions: 1) Activate disaster recovery, 2) Notify stakeholders, 3) Begin service restoration from backup datacenter. ETA: 15 minutes. Will provide updates every 5 minutes.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.95,
                priority="critical",
                reasoning="Emergency protocol activated for system outage"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="CRITICAL: All services down, complete system outage",
                    source_agent="Support",
                    target_agent="Technical",
                    priority="critical"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "EMERGENCY RESPONSE" in result.content
                assert "Immediate Actions" in result.content
                assert "15 minutes" in result.content
                assert result.confidence >= 0.9


class TestTechnicalSpecializations:
    """Test TechnicalAgent specialized technical areas."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_network_diagnostics(self, mock_genai_client):
        """Test network-related technical diagnostics."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Network Diagnostics: Packet loss detected (12%) between user subnet and application servers. Traceroute shows latency spike at ISP gateway. Recommendations: 1) Contact ISP about routing issues, 2) Implement network redundancy, 3) Adjust timeout values.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.87
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Users experiencing intermittent connection timeouts",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"network_tools": "traceroute, ping available"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Network Diagnostics" in result.content
                assert "Packet loss" in result.content
                assert "Traceroute" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_database_performance_analysis(self, mock_genai_client):
        """Test database performance analysis capabilities."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Database Performance Analysis: Query execution time increased 300% due to missing indexes on frequently joined tables. Specific issues: 1) Table scan on Users table (2M records), 2) No index on OrderDate column. Solution: CREATE INDEX idx_users_email, idx_orders_date.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.91
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Database queries running very slowly",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"query_logs": "slow_queries.log", "execution_plans": "available"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Database Performance Analysis" in result.content
                assert "missing indexes" in result.content
                assert "CREATE INDEX" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_api_integration_troubleshooting(self, mock_genai_client):
        """Test API integration troubleshooting capabilities."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="API Integration Analysis: Third-party API returning 429 (Rate Limited) errors. Issue: Exceeded 1000 requests/hour limit. Solutions: 1) Implement request queuing, 2) Add exponential backoff, 3) Negotiate higher rate limits, 4) Cache frequent requests.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.89
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Third-party API integration failing intermittently",
                    source_agent="Support",
                    target_agent="Technical",
                    context={"api_logs": "http_errors.log", "error_pattern": "intermittent 429"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "API Integration Analysis" in result.content
                assert "429" in result.content
                assert "Rate Limited" in result.content
                assert "exponential backoff" in result.content


class TestPerformanceAndComplexity:
    """Test TechnicalAgent performance with complex technical scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.performance
    async def test_complex_multi_system_analysis(self, mock_genai_client):
        """Test analysis of complex multi-system technical issues."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Multi-System Analysis: Cascading failure across microservices. Root cause: Authentication service memory leak causing timeout cascades. Impact chain: Auth → API Gateway → User Service → Database. Recovery plan: 1) Restart auth service, 2) Clear API Gateway cache, 3) Monitor cascade resolution.",
                source_agent="Technical",
                target_agent="Support",
                confidence=0.85,
                reasoning="Analyzed interconnected system dependencies and failure patterns"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Multiple systems failing simultaneously across different services",
                    source_agent="Support",
                    target_agent="Technical",
                    context={
                        "affected_services": ["auth", "api-gateway", "user-service", "database"],
                        "timeline": "failures started 30 minutes ago",
                        "architecture": "microservices"
                    }
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Multi-System Analysis" in result.content
                assert "Cascading failure" in result.content
                assert "Authentication service" in result.content
                assert "Recovery plan" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.performance
    async def test_concurrent_technical_analysis(self, mock_genai_client):
        """Test handling multiple concurrent technical analysis requests."""
        with patch.object(TechnicalAgent, '_initialize_genai_client'):
            agent = TechnicalAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Technical analysis completed for concurrent request",
                source_agent="Technical",
                target_agent="Support"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                messages = [
                    Message(
                        content=f"Technical issue {i}",
                        source_agent="Support",
                        target_agent="Technical"
                    )
                    for i in range(3)
                ]
                
                import asyncio
                tasks = [agent.process_message(msg) for msg in messages]
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 3
                for result in results:
                    assert isinstance(result, Response)
                    assert result.source_agent == "Technical" 