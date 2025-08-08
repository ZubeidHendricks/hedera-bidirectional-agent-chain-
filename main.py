import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from chain.orchestrator import ChainOrchestrator
from models.message import UserRequest


REQUIRED_ENV_VARS = ["GOOGLE_API_KEY"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
CORS_ORIGINS = eval(os.getenv("CORS_ALLOW_ORIGINS", '["*"]'))
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Global orchestrator instance
orchestrator: Optional[ChainOrchestrator] = None


class QueryRequest(BaseModel):
    """Enhanced query request model with comprehensive options"""
    query: str = Field(..., description="User's query or request", min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous", description="Unique user identifier")
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    expected_response_format: str = Field(default="text", pattern="^(text|structured|json)$")
    max_processing_time: Optional[int] = Field(default=60, ge=10, le=300, description="Max processing time in seconds")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context information")

class QueryResponse(BaseModel):
    """Comprehensive query response model"""
    result_id: str
    request_id: str
    response: str
    agents_involved: list
    conversation_flow: list
    processing_time: float
    success: bool
    confidence_score: float
    performance_metrics: Dict[str, Any]
    timestamp: str
    error_details: Optional[str] = None

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    orchestrator: Dict[str, Any]
    agents: Dict[str, Any]
    network_health: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with proper startup and shutdown"""
    print("🚀 Starting Bidirectional Agent Chain Server...")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Debug Mode: {DEBUG_MODE}")
    print(f"   Host: {SERVER_HOST}:{SERVER_PORT}")
    
    global orchestrator
    try:
        orchestrator = ChainOrchestrator()
        
        health_status = await orchestrator.get_agent_status()
        if not health_status.get("network_health", False):
            print("⚠️ Warning: Some agents may not be fully healthy")
        else:
            print("✅ All agents initialized and healthy")
            
        print("🌟 Server startup complete!")
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        raise
    
    yield 
    
    # Cleanup
    print("🔄 Shutting down server...")
    print("✅ Server shutdown complete")

app = FastAPI(
    title="Bidirectional Agent Chain API",
    description="""
    ## Modern Bidirectional Agent Chaining System
    
    A sophisticated multi-agent system that leverages LangGraph and Google's Gemini API to provide 
    intelligent, collaborative responses through specialized agents:
    
    - **Support Agent**: Customer communication and coordination
    - **Technical Agent**: System diagnostics and technical solutions  
    - **Product Agent**: Product knowledge and feature expertise
    
    ### Key Features:
    - 🔄 Bidirectional agent communication
    - 🧠 LangGraph-powered orchestration
    - 🎯 Intelligent agent routing and collaboration
    - 📊 Performance monitoring and analytics
    - 🔍 Comprehensive conversation tracking
    - ⚡ Async processing with timeout management
    
    ### Architecture:
    Built on modern AI orchestration patterns with object-oriented design principles
    and micro-architecture for scalability and maintainability.
    """,
    version="2.0.0",
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def get_orchestrator() -> ChainOrchestrator:
    """Dependency to get the orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    return orchestrator


@app.post("/query", response_model=QueryResponse, status_code=200)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    orchestrator_instance: ChainOrchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """
    Process a user query through the bidirectional agent chain.
    
    The system intelligently routes queries through specialized agents,
    enabling collaborative problem-solving and comprehensive responses.
    """
    try:
        user_request = UserRequest(
            query=request.query,
            user_id=request.user_id,
            priority=request.priority,
            expected_response_format=request.expected_response_format,
            max_processing_time=request.max_processing_time,
            context=request.context
        )
        
        print(f"\n{'='*50}")
        print(f"🔵 NEW QUERY: {request.query}")
        print(f"   User: {request.user_id}")
        print(f"   Priority: {request.priority}")
        print(f"{'='*50}")
        
        result = await orchestrator_instance.process_request(user_request)
        
        print(f"\n{'='*50}")
        print(f"✅ RESPONSE GENERATED")
        print(f"   Confidence: {result.get('confidence_score', 0.0):.2f}")
        print(f"   Processing Time: {result.get('processing_time', 0.0):.2f}s")
        print(f"   Agents: {', '.join(result.get('agents_involved', []))}")
        print(f"{'='*50}")
        
        background_tasks.add_task(log_request_completion, user_request.request_id, result["success"])
        
        return QueryResponse(**result)
        
    except Exception as e:
        print(f"❌ API Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to process query",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(
    orchestrator_instance: ChainOrchestrator = Depends(get_orchestrator)
) -> HealthCheckResponse:
    """
    Comprehensive health check for the bidirectional agent chain system.
    
    Returns detailed status information about:
    - System orchestrator health
    - Individual agent status and performance
    - Network connectivity between agents
    - Performance metrics and configuration
    """
    try:
        status = await orchestrator_instance.get_agent_status()
        
        return HealthCheckResponse(
            status="healthy" if status.get("network_health", False) else "degraded",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            orchestrator=status.get("orchestrator", {}),
            agents=status.get("agents", {}),
            network_health=status.get("network_health", False)
        )
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            orchestrator={"error": str(e)},
            agents={},
            network_health=False
        )

@app.get("/conversation/{request_id}")
async def get_conversation_history(
    request_id: str,
    orchestrator_instance: ChainOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Retrieve detailed conversation history for a specific request.
    
    Provides complete audit trail of agent interactions, decision points,
    and collaborative processes for the specified request.
    """
    try:
        history = orchestrator_instance.get_conversation_history(request_id)
        
        if not history:
            raise HTTPException(
                status_code=404, 
                detail=f"No conversation history found for request: {request_id}"
            )
        
        return {
            "request_id": request_id,
            "conversation_flow": history,
            "total_steps": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrieving conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )

@app.get("/metrics")
async def get_performance_metrics(
    orchestrator_instance: ChainOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Get comprehensive performance metrics for the agent chain system.
    
    Includes:
    - Agent-specific performance statistics
    - Orchestrator metrics and configuration
    - System-wide performance indicators
    - Resource utilization information
    """
    try:
        metrics = orchestrator_instance.get_performance_metrics()
        
        return {
            **metrics,
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "environment": ENVIRONMENT,
                "debug_mode": DEBUG_MODE,
                "server_host": SERVER_HOST,
                "server_port": SERVER_PORT
            }
        }
        
    except Exception as e:
        print(f"❌ Error retrieving metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )

@app.get("/agents")
async def get_agent_details(
    orchestrator_instance: ChainOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Get detailed information about all agents in the system.
    
    Returns comprehensive agent profiles including:
    - Capabilities and expertise areas
    - Connection topology
    - Health status and performance metrics
    - Configuration and model information
    """
    try:
        agent_info = {}
        
        for name, agent in orchestrator_instance.agents.items():
            agent_info[name.lower()] = {
                **agent.get_agent_info(),
                "health": await agent.health_check()
            }
            
            if hasattr(agent, 'get_diagnostic_capabilities'):
                agent_info[name.lower()]["diagnostic_capabilities"] = agent.get_diagnostic_capabilities()
            elif hasattr(agent, 'get_product_capabilities'):
                agent_info[name.lower()]["product_capabilities"] = agent.get_product_capabilities()
            elif hasattr(agent, 'get_collaboration_stats'):
                agent_info[name.lower()]["collaboration_stats"] = agent.get_collaboration_stats()
        
        return {
            "agents": agent_info,
            "network_topology": {
                "total_agents": len(orchestrator_instance.agents),
                "connections": "fully_bidirectional",
                "orchestration": "langgraph_state_based"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error retrieving agent details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent details: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with system information and quick start guide"""
    return {
        "service": "Bidirectional Agent Chain API",
        "version": "2.0.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "architecture": "LangGraph + Google Gemini",
        "endpoints": {
            "query": "POST /query - Process user queries through agent chain",
            "health": "GET /health - Comprehensive system health check", 
            "conversation": "GET /conversation/{id} - Get conversation history",
            "metrics": "GET /metrics - System performance metrics",
            "agents": "GET /agents - Detailed agent information",
            "docs": "GET /docs - Interactive API documentation" if DEBUG_MODE else "Disabled in production"
        },
        "quick_start": {
            "example_query": {
                "method": "POST",
                "url": "/query",
                "body": {
                    "query": "My application is running slowly and users are reporting timeouts",
                    "user_id": "test_user",
                    "priority": "high"
                }
            }
        },
        "timestamp": datetime.now().isoformat()
    }

async def log_request_completion(request_id: str, success: bool):
    """Background task to log request completion"""
    status = "SUCCESS" if success else "FAILED"
    print(f"📋 Request {request_id} completed: {status}")


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request, exc):
    return JSONResponse(
        status_code=408,
        content={
            "error": "Request timeout",
            "message": "The request took too long to process",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Bidirectional Agent Chain Server...")
    print("📋 Configuration:")
    print(f"   - Environment: {ENVIRONMENT}")
    print(f"   - Debug Mode: {DEBUG_MODE}")
    print(f"   - API Documentation: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"   - CORS Origins: {CORS_ORIGINS}")
    print("\n🔗 Quick Test Commands:")
    print(f"   Health Check: curl http://{SERVER_HOST}:{SERVER_PORT}/health")
    print(f"   Test Query: curl -X POST http://{SERVER_HOST}:{SERVER_PORT}/query \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"query\": \"My app crashes when uploading large files\", \"user_id\": \"test_user\"}'")
    
    if DEBUG_MODE:
        uvicorn.run(
            "main:app",
            host=SERVER_HOST,
            port=SERVER_PORT,
            reload=True,
            log_level="debug",
            access_log=True
        )
    else:
        uvicorn.run(
            app,
            host=SERVER_HOST,
            port=SERVER_PORT,
            reload=False,
            log_level="info",
            access_log=False
        )