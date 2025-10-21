"""
FastAPI dependencies for the machine translation API.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.database.connection import get_db_session
from src.database.repositories import JobRepository, CacheRepository, MetricsRepository
from src.services.auth_service import AuthService
from src.utils.exceptions import AuthenticationError, AuthorizationError
from src.utils.logging import api_logger

# Security scheme
security = HTTPBearer()


async def get_job_repository(session=Depends(get_db_session)) -> JobRepository:
    """Get job repository dependency."""
    return JobRepository(session)


async def get_cache_repository(session=Depends(get_db_session)) -> CacheRepository:
    """Get cache repository dependency."""
    return CacheRepository(session)


async def get_metrics_repository(session=Depends(get_db_session)) -> MetricsRepository:
    """Get metrics repository dependency."""
    return MetricsRepository(session)


async def get_auth_service() -> AuthService:
    """Get authentication service dependency."""
    return AuthService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        user_data = await auth_service.authenticate(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_data
        
    except AuthenticationError as e:
        api_logger.warning(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        api_logger.error(f"Authentication error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


async def get_current_active_user(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get current active user (not disabled)."""
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return current_user


async def check_rate_limit(
    current_user: dict = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """Check rate limiting for current user."""
    try:
        user_id = current_user["user_id"]
        
        # Check rate limit
        within_limit = await auth_service.check_rate_limit(user_id)
        
        if not within_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        return current_user
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Rate limit check error: {str(e)}", exc_info=True)
        # Don't block requests if rate limiting fails
        return current_user


async def validate_api_key(
    api_key: str,
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[str]:
    """Validate API key and return user ID."""
    try:
        user_id = await auth_service.validate_api_key(api_key)
        return user_id
    except Exception as e:
        api_logger.error(f"API key validation error: {str(e)}")
        return None


# Optional authentication for public endpoints
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[dict]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user_data = await auth_service.authenticate(token)
        return user_data
    except Exception:
        # Silently fail for optional authentication
        return None


# Admin user dependency
async def get_admin_user(
    current_user: dict = Depends(get_current_active_user)
) -> dict:
    """Get current user if they have admin privileges."""
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


# Pagination dependency
class PaginationParams:
    """Pagination parameters."""
    
    def __init__(
        self,
        page: int = 1,
        per_page: int = 20
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page number must be >= 1"
            )
        
        if per_page < 1 or per_page > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Items per page must be between 1 and 100"
            )
        
        self.page = page
        self.per_page = per_page
        self.offset = (page - 1) * per_page


def get_pagination_params(
    page: int = 1,
    per_page: int = 20
) -> PaginationParams:
    """Get pagination parameters."""
    return PaginationParams(page=page, per_page=per_page)


# Language validation dependency
async def validate_language_pair(
    source_language: str,
    target_language: str
) -> tuple[str, str]:
    """Validate language pair is supported."""
    # This would typically check against a service or database
    # For now, we'll use a simple list
    supported_languages = {
        "en", "es", "fr", "de", "zh", "ja", "ko", "pt", "it", "ru"
    }
    
    source_lang = source_language.lower()
    target_lang = target_language.lower()
    
    if source_lang not in supported_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source language '{source_language}' is not supported"
        )
    
    if target_lang not in supported_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target language '{target_language}' is not supported"
        )
    
    if source_lang == target_lang:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source and target languages cannot be the same"
        )
    
    return source_lang, target_lang


# Content validation dependency
def validate_content_length(content: str) -> str:
    """Validate content length."""
    if not content or not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content cannot be empty"
        )
    
    word_count = len(content.split())
    if word_count > 15000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content exceeds maximum word limit of 15,000 words (current: {word_count})"
        )
    
    return content


# Priority validation dependency
def validate_priority(priority: str) -> str:
    """Validate priority level."""
    valid_priorities = {"normal", "high", "critical"}
    
    if priority not in valid_priorities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority '{priority}'. Must be one of: {', '.join(valid_priorities)}"
        )
    
    return priority