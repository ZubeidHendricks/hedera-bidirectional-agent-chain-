#!/usr/bin/env python3
"""
Hedera Blockchain Data Generator
Generates structured CSV files for the Hedera RAG knowledge base
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import requests
from typing import Dict, List, Any

class HederaDataGenerator:
    """Generate comprehensive Hedera blockchain dataset for RAG knowledge base"""
    
    def __init__(self, output_dir: str = "agents/hedera_rag_kb"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🌐 Hedera Data Generator initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def generate_all_datasets(self):
        """Generate all Hedera datasets"""
        print("🚀 Starting Hedera dataset generation...")
        
        # Generate core datasets
        self.generate_developer_guides()
        self.generate_services_documentation()
        self.generate_api_examples()
        self.generate_use_cases()
        self.generate_tutorials()
        self.generate_sdk_documentation()
        
        print("✅ All Hedera datasets generated successfully!")
        print(f"📊 Files created in: {self.output_dir}")
    
    def generate_developer_guides(self):
        """Generate developer guides dataset"""
        data = [
            {
                "title": "Getting Started with Hedera",
                "description": "Complete guide to setting up your first Hedera application",
                "category": "Getting Started",
                "difficulty": "Beginner",
                "sdk_version": "2.50.0",
                "prerequisites": "Basic programming knowledge, Node.js or Java",
                "estimated_time": "2-3 hours",
                "technologies": "JavaScript/Java SDK, REST API",
                "code_language": "JavaScript, Java",
                "hedera_services": "HCS, HTS",
                "use_case": "Basic account and token operations"
            },
            {
                "title": "Hedera Consensus Service Deep Dive",
                "description": "Advanced guide to implementing consensus messaging with HCS",
                "category": "Consensus Service",
                "difficulty": "Advanced",
                "sdk_version": "2.50.0",
                "prerequisites": "Hedera basics, cryptography knowledge",
                "estimated_time": "4-6 hours",
                "technologies": "Hedera SDK, Mirror Node API",
                "code_language": "Java, Python",
                "hedera_services": "HCS",
                "use_case": "Audit trails, supply chain tracking"
            },
            {
                "title": "Smart Contracts 2.0 Development",
                "description": "Building and deploying smart contracts on Hedera",
                "category": "Smart Contracts",
                "difficulty": "Intermediate",
                "sdk_version": "2.50.0",
                "prerequisites": "Solidity, Ethereum development experience",
                "estimated_time": "3-5 hours",
                "technologies": "Solidity, Hardhat, Hedera SDK",
                "code_language": "Solidity, JavaScript",
                "hedera_services": "Smart Contracts 2.0, HTS",
                "use_case": "DeFi applications, NFT marketplaces"
            },
            {
                "title": "Token Service Integration Guide",
                "description": "Creating and managing fungible and non-fungible tokens",
                "category": "Token Service",
                "difficulty": "Intermediate",
                "sdk_version": "2.50.0",
                "prerequisites": "Hedera basics, token economics",
                "estimated_time": "2-4 hours",
                "technologies": "Hedera SDK, REST API",
                "code_language": "JavaScript, Go",
                "hedera_services": "HTS",
                "use_case": "Token creation, NFT minting"
            },
            {
                "title": "File Service for Data Storage",
                "description": "Storing and retrieving files on the Hedera network",
                "category": "File Service",
                "difficulty": "Beginner",
                "sdk_version": "2.50.0",
                "prerequisites": "Basic Hedera knowledge",
                "estimated_time": "1-2 hours",
                "technologies": "Hedera SDK",
                "code_language": "Python, Java",
                "hedera_services": "HFS",
                "use_case": "Document storage, data immutability"
            }
        ]
        
        df = pd.DataFrame(data)
        output_file = self.output_dir / "Hedera Dataset - developer_guides.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Generated developer guides: {output_file}")
        return df
    
    def generate_services_documentation(self):
        """Generate Hedera services documentation dataset"""
        data = [
            {
                "service_name": "Hedera Consensus Service (HCS)",
                "description": "Decentralized messaging and consensus ordering service",
                "category": "Consensus",
                "endpoints": "createTopic, submitMessage, getTopicInfo",
                "use_cases": "Audit trails, supply chain, messaging",
                "pricing": "Per message submission",
                "throughput": "10,000+ TPS",
                "finality": "3-5 seconds",
                "sdk_support": "Java, JavaScript, Go, Python",
                "mainnet_status": "Available",
                "testnet_status": "Available"
            },
            {
                "service_name": "Hedera Token Service (HTS)",
                "description": "Native token creation and management service",
                "category": "Tokens",
                "endpoints": "createToken, mintToken, transferToken, burnToken",
                "use_cases": "DeFi, NFTs, loyalty programs, CBDCs",
                "pricing": "Per token operation",
                "throughput": "10,000+ TPS",
                "finality": "3-5 seconds",
                "sdk_support": "Java, JavaScript, Go, Python",
                "mainnet_status": "Available",
                "testnet_status": "Available"
            },
            {
                "service_name": "Hedera File Service (HFS)",
                "description": "Decentralized file storage and versioning",
                "category": "Storage",
                "endpoints": "createFile, updateFile, deleteFile, getFileContents",
                "use_cases": "Smart contract bytecode, configuration files",
                "pricing": "Per storage operation",
                "throughput": "Depends on file size",
                "finality": "3-5 seconds",
                "sdk_support": "Java, JavaScript, Go, Python",
                "mainnet_status": "Available",
                "testnet_status": "Available"
            },
            {
                "service_name": "Smart Contracts 2.0",
                "description": "EVM-compatible smart contract execution",
                "category": "Computation",
                "endpoints": "createContract, callContract, updateContract",
                "use_cases": "DeFi protocols, DAOs, complex business logic",
                "pricing": "Gas-based pricing model",
                "throughput": "Depends on contract complexity",
                "finality": "3-5 seconds",
                "sdk_support": "Ethereum tooling, Hedera SDK",
                "mainnet_status": "Available",
                "testnet_status": "Available"
            }
        ]
        
        df = pd.DataFrame(data)
        output_file = self.output_dir / "Hedera Dataset - services_documentation.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Generated services documentation: {output_file}")
        return df
    
    def generate_api_examples(self):
        """Generate API examples dataset"""
        data = [
            {
                "api_name": "Account Creation",
                "endpoint": "/api/v1/accounts",
                "method": "POST",
                "description": "Create a new Hedera account with initial balance",
                "request_example": '{"publicKey": "302a300506032b6570032100...", "initialBalance": 1000}',
                "response_example": '{"accountId": "0.0.123456", "status": "SUCCESS"}',
                "sdk_language": "JavaScript",
                "service": "Cryptocurrency Service",
                "complexity": "Basic",
                "gas_cost": "0.05 HBAR"
            },
            {
                "api_name": "Token Transfer",
                "endpoint": "/api/v1/tokens/transfer",
                "method": "POST",
                "description": "Transfer tokens between accounts",
                "request_example": '{"tokenId": "0.0.789", "from": "0.0.123", "to": "0.0.456", "amount": 100}',
                "response_example": '{"transactionId": "0.0.123@1234567890.123456789", "status": "SUCCESS"}',
                "sdk_language": "Java",
                "service": "HTS",
                "complexity": "Basic",
                "gas_cost": "0.001 HBAR"
            },
            {
                "api_name": "Submit Consensus Message",
                "endpoint": "/api/v1/topics/{topicId}/messages",
                "method": "POST",
                "description": "Submit a message to a consensus topic",
                "request_example": '{"message": "SGVsbG8gSGVkZXJhIQ==", "submitKey": "..."}',
                "response_example": '{"consensusTimestamp": "1234567890.123456789", "sequenceNumber": 42}',
                "sdk_language": "Python",
                "service": "HCS",
                "complexity": "Intermediate",
                "gas_cost": "0.0001 HBAR"
            },
            {
                "api_name": "Smart Contract Call",
                "endpoint": "/api/v1/contracts/call",
                "method": "POST",
                "description": "Call a smart contract function",
                "request_example": '{"contractId": "0.0.999", "functionParameters": "...", "gas": 100000}',
                "response_example": '{"result": "0x123abc...", "gasUsed": 75000}',
                "sdk_language": "Go",
                "service": "Smart Contracts 2.0",
                "complexity": "Advanced",
                "gas_cost": "Variable"
            }
        ]
        
        df = pd.DataFrame(data)
        output_file = self.output_dir / "Hedera Dataset - api_examples.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Generated API examples: {output_file}")
        return df
    
    def generate_use_cases(self):
        """Generate use cases dataset"""
        data = [
            {
                "use_case": "Supply Chain Traceability",
                "industry": "Manufacturing",
                "description": "Track products from origin to consumer using HCS for audit trails",
                "hedera_services": "HCS, HTS",
                "implementation_complexity": "Medium",
                "estimated_cost": "$1,000-10,000/month",
                "key_benefits": "Transparency, compliance, fraud prevention",
                "target_audience": "Enterprises, manufacturers, retailers",
                "technical_requirements": "HCS topics, token tracking, Mirror Node API",
                "success_metrics": "Reduced fraud, improved compliance"
            },
            {
                "use_case": "Decentralized Finance (DeFi)",
                "industry": "Financial Services",
                "description": "Build lending, trading, and yield farming protocols",
                "hedera_services": "Smart Contracts 2.0, HTS",
                "implementation_complexity": "High",
                "estimated_cost": "$50,000-500,000",
                "key_benefits": "Low fees, fast finality, regulatory clarity",
                "target_audience": "DeFi protocols, financial institutions",
                "technical_requirements": "Solidity contracts, token standards, oracles",
                "success_metrics": "TVL, transaction volume, user adoption"
            },
            {
                "use_case": "NFT Marketplace",
                "industry": "Digital Assets",
                "description": "Create and trade non-fungible tokens with native HTS",
                "hedera_services": "HTS, Smart Contracts 2.0",
                "implementation_complexity": "Medium",
                "estimated_cost": "$10,000-100,000",
                "key_benefits": "Low minting costs, fast transactions, carbon negative",
                "target_audience": "Artists, collectors, gaming companies",
                "technical_requirements": "HTS NFTs, marketplace smart contracts, IPFS",
                "success_metrics": "NFTs minted, trading volume, creator revenue"
            },
            {
                "use_case": "Carbon Credit Trading",
                "industry": "Sustainability",
                "description": "Tokenize and trade carbon credits with transparent tracking",
                "hedera_services": "HTS, HCS, Smart Contracts 2.0",
                "implementation_complexity": "High",
                "estimated_cost": "$100,000-1,000,000",
                "key_benefits": "Verifiable credits, automated trading, sustainability",
                "target_audience": "Environmental organizations, corporations",
                "technical_requirements": "Token standards, verification oracles, trading logic",
                "success_metrics": "Credits tokenized, trades executed, CO2 offset"
            },
            {
                "use_case": "Digital Identity Verification",
                "industry": "Identity & Security",
                "description": "Secure, decentralized identity management system",
                "hedera_services": "HCS, HFS, Smart Contracts 2.0",
                "implementation_complexity": "High",
                "estimated_cost": "$25,000-250,000",
                "key_benefits": "Privacy-preserving, interoperable, user-controlled",
                "target_audience": "Identity providers, enterprises, individuals",
                "technical_requirements": "DID standards, credential storage, verification",
                "success_metrics": "Identities created, verifications performed, privacy scores"
            }
        ]
        
        df = pd.DataFrame(data)
        output_file = self.output_dir / "Hedera Dataset - use_cases.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Generated use cases: {output_file}")
        return df
    
    def generate_tutorials(self):
        """Generate tutorials dataset"""
        data = [
            {
                "tutorial_title": "Build Your First DApp on Hedera",
                "difficulty": "Beginner",
                "duration": "2 hours",
                "prerequisites": "Basic JavaScript knowledge",
                "learning_objectives": "Create account, deploy contract, interact with UI",
                "technologies": "React, Hedera SDK, MetaMask",
                "step_count": 8,
                "code_snippets": "Yes",
                "video_available": "Yes",
                "completion_rate": "85%"
            },
            {
                "tutorial_title": "Advanced HCS Message Ordering",
                "difficulty": "Advanced",
                "duration": "4 hours",
                "prerequisites": "Hedera fundamentals, cryptography basics",
                "learning_objectives": "Understand consensus ordering, implement message verification",
                "technologies": "Java SDK, Mirror Node API",
                "step_count": 12,
                "code_snippets": "Yes",
                "video_available": "No",
                "completion_rate": "62%"
            },
            {
                "tutorial_title": "NFT Collection Management",
                "difficulty": "Intermediate",
                "duration": "3 hours",
                "prerequisites": "Token basics, metadata standards",
                "learning_objectives": "Create NFT collection, implement royalties, batch operations",
                "technologies": "HTS, IPFS, React",
                "step_count": 10,
                "code_snippets": "Yes",
                "video_available": "Yes",
                "completion_rate": "74%"
            }
        ]
        
        df = pd.DataFrame(data)
        output_file = self.output_dir / "Hedera Dataset - tutorials.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Generated tutorials: {output_file}")
        return df
    
    def generate_sdk_documentation(self):
        """Generate SDK documentation dataset"""
        data = [
            {
                "sdk_name": "Hedera SDK for JavaScript",
                "version": "2.50.0",
                "language": "JavaScript/TypeScript",
                "platform": "Node.js, Browser",
                "installation": "npm install @hashgraph/sdk",
                "key_features": "Full API coverage, Promise-based, TypeScript support",
                "documentation_url": "https://docs.hedera.com/hedera/sdks-and-apis/sdks/javascript-sdk",
                "github_url": "https://github.com/hashgraph/hedera-sdk-js",
                "examples_count": 45,
                "last_updated": "2025-01-01",
                "download_count": "50,000+/month"
            },
            {
                "sdk_name": "Hedera SDK for Java",
                "version": "2.50.0",
                "language": "Java",
                "platform": "JVM",
                "installation": "Maven/Gradle dependency",
                "key_features": "Native Java integration, extensive examples, enterprise-ready",
                "documentation_url": "https://docs.hedera.com/hedera/sdks-and-apis/sdks/java-sdk",
                "github_url": "https://github.com/hashgraph/hedera-sdk-java",
                "examples_count": 60,
                "last_updated": "2025-01-01",
                "download_count": "25,000+/month"
            },
            {
                "sdk_name": "Hedera SDK for Go",
                "version": "2.50.0",
                "language": "Go",
                "platform": "Go runtime",
                "installation": "go get github.com/hashgraph/hedera-sdk-go/v2",
                "key_features": "Idiomatic Go, high performance, concurrent operations",
                "documentation_url": "https://docs.hedera.com/hedera/sdks-and-apis/sdks/go-sdk",
                "github_url": "https://github.com/hashgraph/hedera-sdk-go",
                "examples_count": 35,
                "last_updated": "2025-01-01",
                "download_count": "15,000+/month"
            }
        ]
        
        df = pd.DataFrame(data)
        output_file = self.output_dir / "Hedera Dataset - sdk_documentation.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Generated SDK documentation: {output_file}")
        return df

def main():
    """Generate all Hedera datasets"""
    print("🌐 Hedera Blockchain Data Generator")
    print("=" * 50)
    
    generator = HederaDataGenerator()
    generator.generate_all_datasets()
    
    print(f"\n📊 Data generation complete!")
    print(f"Files created in: agents/hedera_rag_kb/")
    print(f"Next steps:")
    print(f"1. Review the generated CSV files")
    print(f"2. Run: python create_hedera_embeddings.py")
    print(f"3. Start the agent system: python main.py")

if __name__ == "__main__":
    main()