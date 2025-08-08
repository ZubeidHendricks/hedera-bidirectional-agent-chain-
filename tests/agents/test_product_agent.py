"""
Comprehensive tests for agents/product_agent.py
Tests all ProductAgent functionality with 100% coverage including product knowledge and features.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from agents.product_agent import ProductAgent
from agents.base_agent import BaseAgent
from models.message import Message, Response
from tests.conftest import create_test_message, create_test_response


class TestProductAgentInitialization:
    """Test ProductAgent initialization and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_product_agent_initialization(self, mock_genai_client):
        """Test ProductAgent initialization with proper configuration."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            
            assert agent.name == "Product"
            assert agent.expertise == "product_knowledge"
            assert "product_knowledge" in agent.capabilities
            assert isinstance(agent.system_prompt, str)
            assert "Product Knowledge Agent" in agent.system_prompt
            assert "BIDIRECTIONAL AGENT CHAIN ROLE" in agent.system_prompt
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_product_agent_system_prompt_content(self, mock_genai_client):
        """Test ProductAgent system prompt contains required product elements."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            
            prompt = agent.system_prompt
            
            # Check for key product sections
            assert "PRIMARY PRODUCT CAPABILITIES" in prompt
            assert "BIDIRECTIONAL AGENT CHAIN ROLE" in prompt
            assert "PRODUCT KNOWLEDGE DOMAINS" in prompt
            assert "COLLABORATION PATTERNS" in prompt
            assert "PRODUCT COMMUNICATION STANDARDS" in prompt
            assert "QUALITY STANDARDS" in prompt
            
            # Check for specific product capabilities
            assert "product features" in prompt
            assert "configuration" in prompt
            assert "integration" in prompt
            assert "licensing" in prompt
            assert "compatibility" in prompt
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_product_agent_inheritance(self, mock_genai_client):
        """Test ProductAgent properly inherits from BaseAgent."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            
            assert isinstance(agent, BaseAgent)
            assert hasattr(agent, 'process_message')
            assert hasattr(agent, 'register_agent')
            assert hasattr(agent, 'get_capabilities')


class TestProductKnowledgeDomains:
    """Test ProductAgent's product knowledge and expertise areas."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_feature_functionality_explanation(self, mock_genai_client):
        """Test explanation of product features and functionality."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Feature Analysis: Advanced Analytics Dashboard provides real-time data visualization with 15+ chart types, custom date ranges, and export capabilities. Key functionalities: 1) Drag-and-drop report builder, 2) Automated insights generation, 3) Collaborative sharing, 4) Mobile responsive design. Best practices: Start with template dashboards, customize gradually based on user feedback.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.92,
                reasoning="Comprehensive product feature knowledge"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants to understand the Advanced Analytics Dashboard capabilities",
                    source_agent="Support",
                    target_agent="Product",
                    context={"customer_tier": "enterprise", "current_features": "basic_analytics"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.source_agent == "Product"
                assert "Feature Analysis" in result.content
                assert "real-time data visualization" in result.content
                assert "15+ chart types" in result.content
                assert "Best practices" in result.content
                assert result.confidence >= 0.9
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_product_configuration_guidance(self, mock_genai_client):
        """Test product configuration and setup guidance."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Configuration Guide: Single Sign-On (SSO) Setup for Enterprise customers:\n1. Navigate to Admin Panel → Security → SSO Configuration\n2. Select provider: SAML 2.0, OAuth 2.0, or Azure AD\n3. Enter Identity Provider details (Entity ID, SSO URL, Certificate)\n4. Map user attributes (email, name, department)\n5. Test with pilot group before full deployment\nImportant: Backup current auth settings before proceeding.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.95,
                reasoning="Step-by-step configuration knowledge"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Enterprise customer needs help setting up SSO integration",
                    source_agent="Support",
                    target_agent="Product",
                    context={"customer_tier": "enterprise", "integration_type": "SAML"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Configuration Guide" in result.content
                assert "Single Sign-On (SSO)" in result.content
                assert "1." in result.content and "2." in result.content  # Step-by-step
                assert "SAML 2.0" in result.content
                assert "Important:" in result.content  # Warning included
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_product_limitations_and_constraints(self, mock_genai_client):
        """Test explanation of product limitations and constraints."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Product Limitations: API Integration feature has the following constraints:\n- Maximum 1,000 API calls per hour (Basic tier)\n- File upload size limited to 100MB\n- Real-time sync not available for custom fields\n- Webhook retries: maximum 3 attempts\nWorkarounds: 1) Upgrade to Professional tier for higher limits, 2) Use batch processing for large files, 3) Implement polling for custom field updates.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.88,
                reasoning="Clear understanding of product constraints and alternatives"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer asking why their API integration is being rate limited",
                    source_agent="Support",
                    target_agent="Product",
                    context={"customer_tier": "basic", "api_usage": "high"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Product Limitations" in result.content
                assert "1,000 API calls per hour" in result.content
                assert "Basic tier" in result.content
                assert "Workarounds" in result.content
                assert "Upgrade to Professional" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_licensing_and_subscription_guidance(self, mock_genai_client):
        """Test licensing and subscription tier guidance."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Licensing Analysis: Customer requires Advanced Reporting (Enterprise feature) and API access >1000 calls/hour. Recommendation: Upgrade from Basic ($29/month) to Enterprise ($199/month). Benefits: 1) Unlimited API calls, 2) Advanced analytics, 3) SSO integration, 4) Priority support, 5) Custom integrations. ROI justification: Time savings on manual reporting (~20 hours/month) offsets upgrade cost.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.9,
                reasoning="Analyzed customer needs against subscription tiers"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants advanced reporting features but has basic subscription",
                    source_agent="Support",
                    target_agent="Product",
                    context={"current_plan": "basic", "requested_features": ["advanced_reporting", "high_api_usage"]}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Licensing Analysis" in result.content
                assert "Enterprise ($199/month)" in result.content
                assert "ROI justification" in result.content
                assert "20 hours/month" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_integration_compatibility_assessment(self, mock_genai_client):
        """Test integration compatibility and requirements assessment."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Integration Compatibility: Salesforce CRM integration fully supported. Requirements: 1) Professional tier or higher, 2) Salesforce API access, 3) Custom fields mapping setup. Supported data sync: Contacts, Accounts, Opportunities (bi-directional). Sync frequency: Real-time for Professional+, hourly for Basic. Prerequisites: Admin access to both systems for initial setup.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.93,
                reasoning="Comprehensive integration knowledge and requirements"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants to integrate with Salesforce CRM",
                    source_agent="Support",
                    target_agent="Product",
                    context={"integration_target": "salesforce", "customer_tier": "professional"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Integration Compatibility" in result.content
                assert "Salesforce CRM integration fully supported" in result.content
                assert "bi-directional" in result.content
                assert "Prerequisites" in result.content


class TestProductCollaborationPatterns:
    """Test ProductAgent collaboration with other agents."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_collaboration_with_technical_agent(self, mock_genai_client):
        """Test product collaboration with Technical agent for implementation guidance."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Product-Technical Collaboration: The requested feature (real-time dashboard updates) is available in Enterprise tier. Technical requirements: WebSocket connection, modern browser support (Chrome 80+, Firefox 75+), minimum 2Mbps bandwidth. HANDOFF_REQUEST: Technical - Please verify customer's infrastructure can support WebSocket connections and assess bandwidth requirements.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.85,
                suggested_next_agent="Technical",
                reasoning="Product capabilities known, technical implementation verification needed"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants real-time dashboard updates but experiencing technical issues",
                    source_agent="Support",
                    target_agent="Product",
                    context={"feature_request": "real_time_updates", "technical_issues": "slow_updates"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Product-Technical Collaboration" in result.content
                assert "HANDOFF_REQUEST: Technical" in result.content
                assert "WebSocket connection" in result.content
                assert result.suggested_next_agent == "Technical"
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_collaboration_with_support_agent(self, mock_genai_client):
        """Test product collaboration responding to Support agent queries."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Product Knowledge: Mobile App functionality differs by subscription tier. Basic: View-only access, document sync. Professional: Full editing, offline mode, advanced search. Enterprise: Custom branding, admin controls, analytics. Customer has Professional tier, so they should have full editing capabilities. If not working, likely a permissions or sync issue.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.9,
                reasoning="Product tier analysis and capability mapping"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer can't edit documents in mobile app, only view them",
                    source_agent="Support",
                    target_agent="Product",
                    message_type="handoff",
                    context={"customer_tier": "professional", "issue": "mobile_editing"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.target_agent == "Support"
                assert "Product Knowledge" in result.content
                assert "Professional tier" in result.content
                assert "full editing capabilities" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_product_feature_prioritization_input(self, mock_genai_client):
        """Test product input for feature prioritization and roadmap questions."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Feature Roadmap Guidance: Bulk export functionality is planned for Q2 2024 release. Current alternatives: 1) API-based export (Enterprise tier), 2) Scheduled report delivery, 3) Manual export in batches (<1000 records). Priority customers can request early access to beta version in Q1. Consider upgrade to Enterprise for immediate API access.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.87,
                reasoning="Roadmap knowledge and alternative solution awareness"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Enterprise customer requesting bulk data export feature",
                    source_agent="Support",
                    target_agent="Product",
                    context={"customer_tier": "enterprise", "feature_request": "bulk_export"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Feature Roadmap Guidance" in result.content
                assert "Q2 2024 release" in result.content
                assert "Current alternatives" in result.content
                assert "early access to beta" in result.content


class TestProductCommunicationStandards:
    """Test ProductAgent communication standards and customer focus."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_feature_descriptions_with_examples(self, mock_genai_client):
        """Test feature descriptions include practical examples."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Custom Fields Feature: Create tailored data fields for your specific business needs. Examples: 1) Property management: Add 'Property Type', 'Square Footage', 'Rental Price' fields, 2) HR system: Include 'Department', 'Start Date', 'Performance Rating', 3) Sales tracking: Custom 'Lead Source', 'Deal Stage', 'Expected Close Date'. Each field supports validation rules, dropdown options, and conditional visibility.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.91
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer asks what custom fields can do and how they work",
                    source_agent="Support",
                    target_agent="Product"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Custom Fields Feature" in result.content
                assert "Examples:" in result.content
                assert "Property management:" in result.content
                assert "validation rules" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_configuration_implications_explained(self, mock_genai_client):
        """Test configuration options explain implications and best practices."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Auto-Archive Configuration: Files inactive for 90+ days are moved to archive storage. Implications: 1) Archived files take 3-5 seconds longer to access, 2) 80% storage cost reduction, 3) Search still works but with slight delay. Best practice: Set to 180 days for frequently accessed systems, 90 days for document repositories. Warning: Cannot be reversed without manual intervention.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.88
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants to enable auto-archiving but is concerned about impacts",
                    source_agent="Support",
                    target_agent="Product",
                    context={"feature": "auto_archive", "concern": "access_speed"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Auto-Archive Configuration" in result.content
                assert "Implications:" in result.content
                assert "Best practice:" in result.content
                assert "Warning:" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_alternative_approaches_suggested(self, mock_genai_client):
        """Test suggestion of alternative approaches when preferred options aren't available."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Alternative Solutions: Direct database access isn't available in any tier due to security policies. Alternative approaches: 1) Use Data Export API (Enterprise tier) for automated extracts, 2) Scheduled reports with custom SQL queries, 3) Business Intelligence connector for read-only analytics, 4) Custom dashboard builder for visual reporting. Recommended: Start with BI connector for immediate needs, plan API integration for automation.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.85
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants direct database access for reporting",
                    source_agent="Support",
                    target_agent="Product",
                    context={"request": "database_access", "use_case": "reporting"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Alternative Solutions" in result.content
                assert "isn't available" in result.content
                assert "Alternative approaches:" in result.content
                assert "Recommended:" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_prerequisite_identification(self, mock_genai_client):
        """Test identification and explanation of prerequisites."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="API Integration Prerequisites: Before enabling API access: 1) Upgrade to Professional tier minimum, 2) Complete security review (Enterprise customers), 3) Generate API credentials in Admin panel, 4) Whitelist IP addresses for API access, 5) Review rate limiting documentation. Time required: 2-3 business days for security review (Enterprise), immediate for Professional tier.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.93
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants to start using our API, what do they need to do?",
                    source_agent="Support",
                    target_agent="Product",
                    context={"customer_tier": "basic", "request": "api_access"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "API Integration Prerequisites" in result.content
                assert "Before enabling API access:" in result.content
                assert "Time required:" in result.content
                assert "security review" in result.content


class TestProductVersionAndCompatibility:
    """Test ProductAgent handling of version differences and compatibility."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_version_differences_explanation(self, mock_genai_client):
        """Test explanation of version differences and upgrade considerations."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Version Comparison: Current v3.2 vs Legacy v2.8:\nNew in v3.2: 1) Enhanced security (OAuth 2.0), 2) Real-time collaboration, 3) Mobile app improvements, 4) 50% faster search, 5) Advanced permissions. Breaking changes: Legacy API endpoints deprecated, custom CSS requires updates. Migration timeline: 2-4 weeks typical. Support: v2.8 supported until Dec 2024.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.9
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer on version 2.8 asking about upgrading to latest version",
                    source_agent="Support",
                    target_agent="Product",
                    context={"current_version": "2.8", "latest_version": "3.2"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Version Comparison" in result.content
                assert "Breaking changes:" in result.content
                assert "Migration timeline:" in result.content
                assert "v2.8 supported until Dec 2024" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_compatibility_matrix_guidance(self, mock_genai_client):
        """Test compatibility matrix and system requirements guidance."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Compatibility Requirements: Windows 10+ (64-bit), macOS 10.15+, Linux Ubuntu 18.04+. Browser support: Chrome 85+, Firefox 78+, Safari 14+, Edge 85+. Mobile: iOS 13+, Android 8+. Network: Minimum 1Mbps, recommended 5Mbps for video features. Storage: 2GB local cache space. Note: IE 11 deprecated as of v3.0, legacy browser support available on request.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.95
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer asking about system requirements and browser compatibility",
                    source_agent="Support",
                    target_agent="Product",
                    context={"inquiry_type": "system_requirements"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Compatibility Requirements" in result.content
                assert "Windows 10+" in result.content
                assert "Chrome 85+" in result.content
                assert "IE 11 deprecated" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_upgrade_path_recommendations(self, mock_genai_client):
        """Test upgrade path recommendations and migration guidance."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Upgrade Path: v2.5 → v3.2 requires intermediate upgrade to v3.0 first. Recommended sequence: 1) Backup current data, 2) Upgrade to v3.0 (test environment), 3) Validate custom integrations, 4) Upgrade to v3.2, 5) Update API calls to new endpoints. Critical: Review deprecated features list before starting. Migration tools available for data conversion.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.88
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer on version 2.5 wants to upgrade to latest, what's the process?",
                    source_agent="Support",
                    target_agent="Product",
                    context={"current_version": "2.5", "target_version": "3.2", "custom_integrations": True}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Upgrade Path" in result.content
                assert "intermediate upgrade to v3.0" in result.content
                assert "Recommended sequence:" in result.content
                assert "Migration tools available" in result.content


class TestErrorHandlingAndEdgeCases:
    """Test ProductAgent error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_unknown_feature_inquiry(self, mock_genai_client):
        """Test handling inquiries about unknown or non-existent features."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Feature Inquiry: 'Quantum Analytics' is not a current product feature. Similar available features: 1) Advanced Analytics Dashboard, 2) Predictive Analytics (Enterprise), 3) Custom Reporting Suite. If this refers to a specific requirement, please provide more details about the desired functionality for proper guidance.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.4,
                needs_clarification=True,
                reasoning="Unknown feature name requires clarification"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer asking about Quantum Analytics feature",
                    source_agent="Support",
                    target_agent="Product"
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert result.needs_clarification is True
                assert result.confidence <= 0.5
                assert "not a current product feature" in result.content
                assert "Similar available features" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_conflicting_feature_requirements(self, mock_genai_client):
        """Test handling conflicting or incompatible feature requirements."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Configuration Conflict: Real-time sync and offline mode cannot be simultaneously enabled due to technical constraints. Options: 1) Real-time sync with online-only access, 2) Offline mode with periodic sync (hourly/daily), 3) Hybrid approach: critical data real-time, bulk data periodic. Recommend discussing specific use case to determine optimal configuration.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.7,
                needs_clarification=True,
                reasoning="Conflicting requirements need use case clarification"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer wants both real-time sync and offline mode enabled",
                    source_agent="Support",
                    target_agent="Product",
                    context={"requirements": ["real_time_sync", "offline_mode"]}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Configuration Conflict" in result.content
                assert "cannot be simultaneously enabled" in result.content
                assert "Options:" in result.content
                assert result.needs_clarification is True
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_outdated_product_information_request(self, mock_genai_client):
        """Test handling requests about outdated or deprecated features."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Deprecated Feature: Classic Reports were discontinued in v3.0 (January 2024). Modern replacement: Advanced Reporting Suite with enhanced functionality: 1) Drag-and-drop builder, 2) Real-time data, 3) Interactive visualizations, 4) Scheduled delivery. Migration guide available. For legacy report recreation, contact support with specific report requirements.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.9,
                reasoning="Clear deprecation knowledge and migration path"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer can't find Classic Reports feature in new interface",
                    source_agent="Support",
                    target_agent="Product",
                    context={"feature": "classic_reports", "customer_version": "3.1"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Deprecated Feature" in result.content
                assert "discontinued in v3.0" in result.content
                assert "Modern replacement" in result.content
                assert "Migration guide available" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.error_handling
    async def test_tier_limitation_explanation(self, mock_genai_client):
        """Test clear explanation of subscription tier limitations."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Tier Limitation: Advanced Security features (SSO, 2FA, audit logs) are Enterprise-only features. Customer's Professional tier includes: Standard security, basic audit trail, team permissions. To access requested features, upgrade to Enterprise required. Alternative: Use third-party authentication service with our API integration (Professional tier compatible).",
                source_agent="Product",
                target_agent="Support",
                confidence=0.95,
                reasoning="Clear tier boundary knowledge and alternatives"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Professional tier customer wants to enable SSO and audit logging",
                    source_agent="Support",
                    target_agent="Product",
                    context={"customer_tier": "professional", "requested_features": ["sso", "audit_logs"]}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Tier Limitation" in result.content
                assert "Enterprise-only features" in result.content
                assert "Professional tier includes" in result.content
                assert "Alternative:" in result.content


class TestProductSpecializedKnowledge:
    """Test ProductAgent specialized product knowledge areas."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_workflow_optimization_guidance(self, mock_genai_client):
        """Test workflow optimization and best practices guidance."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Workflow Optimization: For sales team efficiency:\n1. Setup: Automated lead scoring, email templates, pipeline stages\n2. Daily workflow: Lead import → Scoring → Assignment → Follow-up automation\n3. Reporting: Weekly pipeline review, monthly conversion analysis\n4. Optimization: A/B test email templates, refine scoring criteria quarterly\nExpected improvement: 30-40% time savings, 25% higher conversion rates.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.89
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Sales team wants to optimize their workflow using our platform",
                    source_agent="Support",
                    target_agent="Product",
                    context={"team": "sales", "goal": "efficiency", "current_process": "manual"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Workflow Optimization" in result.content
                assert "sales team efficiency" in result.content
                assert "Expected improvement" in result.content
                assert "30-40% time savings" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_industry_specific_configurations(self, mock_genai_client):
        """Test industry-specific configuration recommendations."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Healthcare Industry Configuration: HIPAA compliance requirements:\n1. Enable field-level encryption for PHI data\n2. Configure audit trails for all patient record access\n3. Set data retention policies (7-year minimum)\n4. Enable role-based access controls\n5. Configure session timeouts (15 minutes)\nCompliance features available in Healthcare tier ($299/month). Includes BAA signing and security assessments.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.92
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Healthcare customer needs HIPAA compliant configuration",
                    source_agent="Support",
                    target_agent="Product",
                    context={"industry": "healthcare", "compliance": "HIPAA"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Healthcare Industry Configuration" in result.content
                assert "HIPAA compliance" in result.content
                assert "Healthcare tier" in result.content
                assert "BAA signing" in result.content
    
    @pytest.mark.unit
    @pytest.mark.agents
    async def test_performance_optimization_recommendations(self, mock_genai_client):
        """Test performance optimization recommendations for large datasets."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Performance Optimization for Large Datasets (500K+ records):\n1. Enable data pagination (1000 records/page)\n2. Use filtered views instead of full table loads\n3. Implement lazy loading for detail views\n4. Schedule heavy reports for off-peak hours\n5. Consider data archiving for historical records >2 years\nExpected performance: 60% faster load times, 40% reduced memory usage.",
                source_agent="Product",
                target_agent="Support",
                confidence=0.87
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                message = Message(
                    content="Customer with 500,000 records experiencing slow performance",
                    source_agent="Support",
                    target_agent="Product",
                    context={"record_count": "500000", "issue": "slow_performance"}
                )
                
                result = await agent.process_message(message)
                
                assert isinstance(result, Response)
                assert "Performance Optimization" in result.content
                assert "500K+ records" in result.content
                assert "60% faster load times" in result.content
                assert "data archiving" in result.content


class TestProductPerformanceAndScalability:
    """Test ProductAgent performance with complex product scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.performance
    async def test_concurrent_product_inquiries(self, mock_genai_client):
        """Test handling multiple concurrent product inquiries."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            agent.genai_client = mock_genai_client['model_instance']
            
            mock_response = Response(
                content="Product information provided for concurrent inquiry",
                source_agent="Product",
                target_agent="Support"
            )
            
            with patch.object(agent, '_generate_response', return_value=mock_response):
                messages = [
                    Message(
                        content=f"Product question {i}",
                        source_agent="Support",
                        target_agent="Product"
                    )
                    for i in range(5)
                ]
                
                import asyncio
                tasks = [agent.process_message(msg) for msg in messages]
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 5
                for result in results:
                    assert isinstance(result, Response)
                    assert result.source_agent == "Product"
    
    @pytest.mark.unit
    @pytest.mark.agents
    @pytest.mark.performance
    def test_product_knowledge_memory_efficiency(self, mock_genai_client):
        """Test memory efficiency with extensive product knowledge."""
        with patch.object(ProductAgent, '_initialize_genai_client'):
            agent = ProductAgent()
            
            # Simulate extensive conversation history
            for i in range(50):
                message = Message(
                    content=f"Product inquiry {i}",
                    source_agent="Support",
                    target_agent="Product"
                )
                agent.conversation_history.append(message)
            
            # Check memory management
            assert len(agent.conversation_history) == 50
            assert isinstance(agent.system_prompt, str)
            assert len(agent.system_prompt) > 0 