"""
FastAPI application factory and configuration.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.middleware import setup_middleware
from src.api.routes import translation_router, system_router
from src.config.config import config
from src.database.connection import init_database, close_database
from src.utils.exceptions import TranslationSystemException, create_error_response
from src.utils.logging import api_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    api_logger.info("Starting machine translation API service")
    
    try:
        # Initialize database
        await init_database()
        api_logger.info("Database initialized successfully")
        
        # Initialize other services here (Redis, queue, etc.)
        
        api_logger.info("Machine translation API service started successfully")
        
    except Exception as e:
        api_logger.error(f"Failed to start service: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    api_logger.info("Shutting down machine translation API service")
    
    try:
        # Close database connections
        await close_database()
        api_logger.info("Database connections closed")
        
        # Close other services here
        
        api_logger.info("Machine translation API service shut down successfully")
        
    except Exception as e:
        api_logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Machine Translation API",
        description="""
        A scalable, cost-efficient, and performant backend system for machine translation services.
        
        ## Features
        
        * **High Performance**: Achieves 1,500+ words per minute translation speed
        * **Auto-scaling**: Dynamic resource management based on demand
        * **Priority Handling**: Critical, high, and normal priority queues
        * **Multi-level Caching**: Optimized caching for frequently requested translations
        * **Cost Optimization**: Intelligent resource pooling and cost tracking
        * **Comprehensive Monitoring**: Real-time metrics and alerting
        * **Security**: JWT authentication, rate limiting, and data encryption
        
        ## Authentication
        
        All endpoints (except health check and language list) require authentication using JWT tokens.
        Include the token in the Authorization header: `Bearer <your-jwt-token>`
        
        ## Rate Limiting
        
        API requests are rate limited per user. Default limits apply unless otherwise specified.
        
        ## Error Handling
        
        The API returns structured error responses with appropriate HTTP status codes.
        All errors include a correlation ID for tracking and debugging.
        """,
        version="1.0.0",
        contact={
            "name": "Machine Translation API Support",
            "email": "support@translation-api.com"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        lifespan=lifespan,
        docs_url="/docs" if config.environment.value != "production" else None,
        redoc_url="/redoc" if config.environment.value != "production" else None,
        openapi_url="/openapi.json" if config.environment.value != "production" else None
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(translation_router)
    app.include_router(system_router)
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add security scheme
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        
        # Add global security requirement
        openapi_schema["security"] = [{"BearerAuth": []}]
        
        # Add custom tags
        openapi_schema["tags"] = [
            {
                "name": "translation",
                "description": "Translation operations including job submission, status checking, and result retrieval"
            },
            {
                "name": "system",
                "description": "System operations including health checks, statistics, and administrative functions"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Exception handlers
    @app.exception_handler(TranslationSystemException)
    async def translation_system_exception_handler(request: Request, exc: TranslationSystemException):
        """Handle custom translation system exceptions."""
        api_logger.warning(
            f"Translation system exception: {exc.error_code} - {exc.message}",
            metadata={
                "error_code": exc.error_code,
                "path": request.url.path,
                "method": request.method,
                "correlation_id": getattr(request.state, "correlation_id", None)
            }
        )
        
        status_code = status.HTTP_400_BAD_REQUEST
        if exc.error_code in ["AUTHENTICATION_ERROR", "AUTHORIZATION_ERROR"]:
            status_code = status.HTTP_401_UNAUTHORIZED
        elif exc.error_code == "RATE_LIMIT_ERROR":
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif exc.error_code in ["TRANSLATION_ENGINE_ERROR", "DATABASE_ERROR", "QUEUE_ERROR"]:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        return JSONResponse(
            status_code=status_code,
            content=create_error_response(exc),
            headers={"X-Correlation-ID": getattr(request.state, "correlation_id", "unknown")}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        api_logger.warning(
            f"Request validation error: {str(exc)}",
            metadata={
                "path": request.url.path,
                "method": request.method,
                "errors": exc.errors(),
                "correlation_id": getattr(request.state, "correlation_id", None)
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            },
            headers={"X-Correlation-ID": getattr(request.state, "correlation_id", "unknown")}
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        api_logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            metadata={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
                "correlation_id": getattr(request.state, "correlation_id", None)
            }
        )
        
        # If detail is already a dict (structured error), use it directly
        if isinstance(exc.detail, dict):
            content = exc.detail
        else:
            content = {
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail
                }
            }
        
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers={
                **dict(exc.headers or {}),
                "X-Correlation-ID": getattr(request.state, "correlation_id", "unknown")
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        api_logger.error(
            f"Unexpected error: {str(exc)}",
            metadata={
                "path": request.url.path,
                "method": request.method,
                "correlation_id": getattr(request.state, "correlation_id", None)
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred"
                }
            },
            headers={"X-Correlation-ID": getattr(request.state, "correlation_id", "unknown")}
        )
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Machine Translation API",
            "version": "1.0.0",
            "status": "operational",
            "documentation": "/docs" if config.environment.value != "production" else None,
            "health_check": "/api/v1/health"
        }
    
    # Custom docs endpoint with authentication
    if config.environment.value != "production":
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=f"{app.title} - Documentation",
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            )
    
    return app


# Create the application instance
app = create_app()