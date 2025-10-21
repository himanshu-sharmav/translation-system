"""
Authentication service for JWT token management and user validation.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import jwt
from passlib.context import CryptContext

from src.config.config import config
from src.database.connection import get_db_session
from src.database.models import User, RateLimit
from src.database.repositories.base import BaseRepository
from src.utils.exceptions import AuthenticationError, AuthorizationError, RateLimitError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "auth-service")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for handling authentication and authorization."""
    
    def __init__(self):
        self.secret_key = config.security.jwt_secret_key
        self.algorithm = "HS256"
        self.token_expiration_hours = config.security.jwt_expiration_hours
        self.rate_limit_per_minute = config.security.rate_limit_per_minute
    
    async def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by JWT token."""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Extract user information
            user_id = payload.get("sub")
            if not user_id:
                raise AuthenticationError("Invalid token: missing user ID")
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise AuthenticationError("Token has expired")
            
            # Get user from database
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"user_id": user_id})
                
                if not user:
                    raise AuthenticationError("User not found")
                
                if not user.is_active:
                    raise AuthenticationError("User account is disabled")
                
                # Update last login
                user.last_login = datetime.utcnow()
                await user_repo.update(user)
                await session.commit()
            
            logger.info(
                f"User authenticated successfully",
                metadata={
                    "user_id": user_id,
                    "token_exp": exp
                }
            )
            
            return {
                "user_id": user.user_id,
                "email": user.email,
                "is_active": user.is_active,
                "is_admin": user_id.startswith("admin_"),  # Simple admin check
                "rate_limit_per_minute": user.rate_limit_per_minute
            }
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            raise AuthenticationError("Authentication failed")
    
    async def generate_token(self, user_id: str) -> str:
        """Generate JWT token for user."""
        try:
            # Check if user exists
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"user_id": user_id})
                
                if not user:
                    raise AuthenticationError("User not found")
                
                if not user.is_active:
                    raise AuthenticationError("User account is disabled")
            
            # Create token payload
            now = datetime.utcnow()
            exp = now + timedelta(hours=self.token_expiration_hours)
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": exp,
                "type": "access_token"
            }
            
            # Generate token
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            logger.info(
                f"Token generated for user",
                metadata={
                    "user_id": user_id,
                    "expires_at": exp.isoformat()
                }
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Token generation error: {str(e)}", exc_info=True)
            raise AuthenticationError("Failed to generate token")
    
    async def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        try:
            if not api_key or len(api_key) < 32:
                return None
            
            # Hash the API key for comparison
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"api_key": api_key})
                
                if not user:
                    logger.warning(f"Invalid API key attempted")
                    return None
                
                if not user.is_active:
                    logger.warning(f"API key used for disabled user: {user.user_id}")
                    return None
                
                # Update last login
                user.last_login = datetime.utcnow()
                await user_repo.update(user)
                await session.commit()
                
                logger.info(
                    f"API key validated successfully",
                    metadata={"user_id": user.user_id}
                )
                
                return user.user_id
                
        except Exception as e:
            logger.error(f"API key validation error: {str(e)}", exc_info=True)
            return None
    
    async def check_rate_limit(self, user_id: str, endpoint: str = "default") -> bool:
        """Check if user is within rate limits."""
        try:
            current_time = datetime.utcnow()
            window_start = current_time.replace(second=0, microsecond=0)
            window_end = window_start + timedelta(minutes=1)
            
            async with get_db_session() as session:
                # Get user's rate limit
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"user_id": user_id})
                
                if not user:
                    raise AuthenticationError("User not found")
                
                user_rate_limit = user.rate_limit_per_minute
                
                # Get or create rate limit entry
                rate_limit_repo = BaseRepository(session, RateLimit)
                rate_limit_entry = await rate_limit_repo.find_one_by({
                    "user_id": user_id,
                    "endpoint": endpoint,
                    "window_start": window_start
                })
                
                if rate_limit_entry:
                    # Check if limit exceeded
                    if rate_limit_entry.request_count >= user_rate_limit:
                        logger.warning(
                            f"Rate limit exceeded",
                            metadata={
                                "user_id": user_id,
                                "endpoint": endpoint,
                                "request_count": rate_limit_entry.request_count,
                                "limit": user_rate_limit
                            }
                        )
                        return False
                    
                    # Increment request count
                    rate_limit_entry.request_count += 1
                    await rate_limit_repo.update(rate_limit_entry)
                else:
                    # Create new rate limit entry
                    rate_limit_entry = RateLimit(
                        user_id=user_id,
                        endpoint=endpoint,
                        request_count=1,
                        window_start=window_start,
                        window_end=window_end
                    )
                    await rate_limit_repo.create(rate_limit_entry)
                
                await session.commit()
                
                logger.debug(
                    f"Rate limit check passed",
                    metadata={
                        "user_id": user_id,
                        "endpoint": endpoint,
                        "request_count": rate_limit_entry.request_count,
                        "limit": user_rate_limit
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}", exc_info=True)
            # Default to allowing request if rate limiting fails
            return True
    
    async def create_user(self, user_id: str, email: str, password: str = None) -> Dict[str, Any]:
        """Create a new user account."""
        try:
            # Generate API key
            api_key = self.generate_api_key()
            
            # Hash password if provided
            password_hash = None
            if password:
                password_hash = pwd_context.hash(password)
            
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                
                # Check if user already exists
                existing_user = await user_repo.find_one_by({"user_id": user_id})
                if existing_user:
                    raise AuthenticationError("User already exists")
                
                # Check if email already exists
                if email:
                    existing_email = await user_repo.find_one_by({"email": email})
                    if existing_email:
                        raise AuthenticationError("Email already registered")
                
                # Create user
                user = User(
                    user_id=user_id,
                    email=email,
                    api_key=api_key,
                    is_active=True,
                    rate_limit_per_minute=self.rate_limit_per_minute
                )
                
                created_user = await user_repo.create(user)
                await session.commit()
                
                logger.info(
                    f"User created successfully",
                    metadata={
                        "user_id": user_id,
                        "email": email
                    }
                )
                
                return {
                    "user_id": created_user.user_id,
                    "email": created_user.email,
                    "api_key": api_key,
                    "is_active": created_user.is_active,
                    "rate_limit_per_minute": created_user.rate_limit_per_minute
                }
                
        except Exception as e:
            logger.error(f"User creation error: {str(e)}", exc_info=True)
            raise AuthenticationError("Failed to create user")
    
    async def update_user_rate_limit(self, user_id: str, new_limit: int) -> bool:
        """Update user's rate limit."""
        try:
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"user_id": user_id})
                
                if not user:
                    raise AuthenticationError("User not found")
                
                user.rate_limit_per_minute = new_limit
                await user_repo.update(user)
                await session.commit()
                
                logger.info(
                    f"User rate limit updated",
                    metadata={
                        "user_id": user_id,
                        "new_limit": new_limit
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Rate limit update error: {str(e)}", exc_info=True)
            return False
    
    async def disable_user(self, user_id: str) -> bool:
        """Disable user account."""
        try:
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"user_id": user_id})
                
                if not user:
                    raise AuthenticationError("User not found")
                
                user.is_active = False
                await user_repo.update(user)
                await session.commit()
                
                logger.info(
                    f"User disabled",
                    metadata={"user_id": user_id}
                )
                
                return True
                
        except Exception as e:
            logger.error(f"User disable error: {str(e)}", exc_info=True)
            return False
    
    async def cleanup_old_rate_limits(self, hours: int = 24) -> int:
        """Clean up old rate limit entries."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            async with get_db_session() as session:
                rate_limit_repo = BaseRepository(session, RateLimit)
                
                # Delete old rate limit entries
                deleted_count = await rate_limit_repo.bulk_delete({
                    "window_end": {"lt": cutoff_time}
                })
                
                await session.commit()
                
                logger.info(
                    f"Cleaned up old rate limit entries",
                    metadata={
                        "deleted_count": deleted_count,
                        "cutoff_hours": hours
                    }
                )
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Rate limit cleanup error: {str(e)}", exc_info=True)
            return 0
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        try:
            async with get_db_session() as session:
                user_repo = BaseRepository(session, User)
                user = await user_repo.find_one_by({"user_id": user_id})
                
                if not user:
                    return None
                
                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "is_active": user.is_active,
                    "rate_limit_per_minute": user.rate_limit_per_minute,
                    "created_at": user.created_at,
                    "last_login": user.last_login
                }
                
        except Exception as e:
            logger.error(f"Get user info error: {str(e)}", exc_info=True)
            return None