"""
FastAPI middleware for the machine translation API.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logging import api_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.time()
        
        # Extract request info
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request
        api_logger.info(
            f"Request started: {method} {url}",
            event="request_started",
            metadata={
                "correlation_id": correlation_id,
                "method": method,
                "url": url,
                "client_ip": client_ip,
                "user_agent": user_agent
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            api_logger.info(
                f"Request completed: {method} {url} - {response.status_code}",
                event="request_completed",
                metrics={
                    "processing_time_ms": round(process_time * 1000, 2),
                    "status_code": response.status_code
                },
                metadata={
                    "correlation_id": correlation_id,
                    "method": method,
                    "url": url
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            api_logger.error(
                f"Request failed: {method} {url} - {str(e)}",
                event="request_failed",
                metrics={
                    "processing_time_ms": round(process_time * 1000, 2)
                },
                metadata={
                    "correlation_id": correlation_id,
                    "method": method,
                    "url": url,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Remove server header for security
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware (simplified implementation)."""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean up old entries
        self.clients = {
            ip: timestamps for ip, timestamps in self.clients.items()
            if any(t > current_time - self.period for t in timestamps)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            # Filter recent requests
            recent_requests = [
                t for t in self.clients[client_ip]
                if t > current_time - self.period
            ]
            
            if len(recent_requests) >= self.calls:
                api_logger.warning(
                    f"Rate limit exceeded for {client_ip}",
                    event="rate_limit_exceeded",
                    metadata={
                        "client_ip": client_ip,
                        "requests_count": len(recent_requests),
                        "limit": self.calls,
                        "period": self.period
                    }
                )
                
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(self.period)}
                )
            
            self.clients[client_ip] = recent_requests + [current_time]
        else:
            self.clients[client_ip] = [current_time]
        
        return await call_next(request)


def setup_cors_middleware(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-Process-Time"]
    )


def setup_middleware(app):
    """Setup all middleware for the FastAPI app."""
    # Add custom middleware (order matters - last added is executed first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, calls=1000, period=60)  # 1000 requests per minute
    
    # Setup CORS
    setup_cors_middleware(app)